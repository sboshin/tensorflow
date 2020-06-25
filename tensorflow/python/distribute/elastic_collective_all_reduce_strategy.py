# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class ElasticCollectiveAllReduceStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import difflib
import weakref
import threading
from six.moves import queue as Queue

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.eager import tape
from tensorflow.python.framework import c_api_util
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.framework import dtypes
from tensorflow.python.data.experimental.ops import distribute


# TODO(yuefengz): support in-graph replication.
@tf_export("distribute.experimental.ElasticMultiWorkerMirroredStrategy", v1=[])
class ElasticCollectiveAllReduceStrategy(collective_all_reduce_strategy.CollectiveAllReduceStrategy):
  """A distribution strategy for synchronous training on multiple workers.

  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it creates copies of all variables in the
  model on each device across all workers.

  It uses CollectiveOps's implementation of multi-worker all-reduce to
  to keep variables in sync. A collective op is a single op in the
  TensorFlow graph which can automatically choose an all-reduce algorithm in
  the TensorFlow runtime according to hardware, network topology and tensor
  sizes.

  By default it uses all local GPUs or CPU for single-worker training.

  When 'TF_CONFIG' environment variable is set, it parses cluster_spec,
  task_type and task_id from 'TF_CONFIG' and turns into a multi-worker strategy
  which mirrored models on GPUs of all machines in a cluster. In the current
  implementation, it uses all GPUs in a cluster and it assumes all workers have
  the same number of GPUs.

  You can also pass a `distribute.cluster_resolver.ClusterResolver` instance
  when instantiating the strategy. The task_type, task_id etc. will be parsed
  from the resolver instance instead of from the `TF_CONFIG` env var.

  It supports both eager mode and graph mode. However, for eager mode, it has to
  set up the eager context in its constructor and therefore all ops in eager
  mode have to run after the strategy object is created.

  """
  # TODO(anjalisridhar): Update our guides with examples showing how we can use
  # the cluster_resolver argument.

  def __init__(
      self,
      communication=cross_device_ops_lib.CollectiveCommunication.AUTO,
      cluster_resolver=None):
    """Creates the strategy.

    Args:
      communication: optional Enum of type
        `distribute.experimental.CollectiveCommunication`.  This provides a way
        for the user to override the choice of collective op communication.
        Possible values include `AUTO`, `RING`, and `NCCL`.
      cluster_resolver: optional `distribute.cluster_resolver.ClusterResolver`
        object. The default ClusterResolver that is used is the
        TFConfigClusterResolver which is instantiated from the TF_CONFIG env
        var.
    """
    # TODO(b/150151677): consider move communication to CollectiveHints.
    super(collective_all_reduce_strategy.CollectiveAllReduceStrategy, self).__init__(
        ElasticCollectiveAllReduceExtended(
            self,
            communication=communication,
            cluster_resolver=cluster_resolver))

    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MultiWorkerMirroredStrategy")
    # pylint: disable=protected-access
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended._num_gpus_per_worker)

  @classmethod
  def _from_local_devices(cls, devices):
    """A convenience method to create an object with a list of devices."""
    obj = cls()
    obj.extended._initialize_local(TFConfigClusterResolver(), devices=devices)  # pylint: disable=protected-access
    return obj

  # def scope(self):  # pylint: disable=useless-super-delegation
  #   """Returns a context manager selecting this Strategy as current.

  #   Inside a `with strategy.scope():` code block, this thread
  #   will use a variable creator set by `strategy`, and will
  #   enter its "cross-replica context".

  #   In `MultiWorkerMirroredStrategy`, all variables created inside
  #   `strategy.scope() will be mirrored on all replicas of each worker.
  #   Moreover, it also sets a default device scope so that ops without
  #   specified devices will end up on the correct worker.

  #   Returns:
  #     A context manager to use for creating variables with this strategy.
  #   """
  #   return super(CollectiveAllReduceStrategy, self).scope()


class ElasticCollectiveAllReduceExtended(collective_all_reduce_strategy.CollectiveAllReduceExtended):
  """Implementation of CollectiveAllReduceStrategy."""

  def __init__(self,
               container_strategy,
               communication,
               cluster_resolver):
    self._cluster_resolver = cluster_resolver or TFConfigClusterResolver()
    distribute_lib.StrategyExtendedV1.__init__(self, container_strategy)
    assert isinstance(
        communication,
        cross_device_ops_lib.CollectiveCommunication)
    self._communication = communication
    self._initialize_strategy(self._cluster_resolver)
    self._cfer_fn_cache = weakref.WeakKeyDictionary()
    assert isinstance(self._get_cross_device_ops(),
                      cross_device_ops_lib.CollectiveAllReduce)
    self.var_list = []
    self._iterators = []

  def _initialize_strategy(self, cluster_resolver):
    if cluster_resolver.cluster_spec().as_dict():
      self._initialize_multi_worker(cluster_resolver)
    else:
      exit(1)
      self._initialize_local(cluster_resolver)

  def _get_variable_creator_initial_value(self,
                                          replica_id,
                                          device,
                                          primary_var,
                                          **kwargs):
    return super(collective_all_reduce_strategy.CollectiveAllReduceExtended,
                   self)._get_variable_creator_initial_value(
                       replica_id=replica_id,
                       device=device,
                       primary_var=primary_var,
                       **kwargs)

  def var_sync(self):
    #broadcast values from chief to workers
    group_key = self._collective_keys.get_group_key([device])
    group_size = self._num_workers
    collective_instance_key = (
        self._collective_keys.get_variable_instance_key())
    if self._is_chief:
      bcast_send = collective_ops.broadcast_send(
                  initial_value, initial_value.shape, initial_value.dtype,
                  group_size, group_key, collective_instance_key)
      with ops.control_dependencies([bcast_send]):
        return array_ops.identity(initial_value)



  def _create_variable(self, next_creator, **kwargs):
    """Create a mirrored variable. See `DistributionStrategy.scope`."""
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      devices = self._devices
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      devices = colocate_with._devices  # pylint: disable=protected-access

    def _real_mirrored_creator(**kwargs):  # pylint: disable=g-missing-docstring
      value_list = []
      for i, d in enumerate(devices):
        with ops.device(d):
          kwargs["initial_value"] = self._get_variable_creator_initial_value(
              replica_id=i,
              device=d,
              primary_var=value_list[0] if value_list else None,
              **kwargs)
          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = value_list[0].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)
          with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
            # Don't record operations (e.g. other variable reads) during
            # variable creation.
            with tape.stop_recording():
              v = next_creator(**kwargs)
          assert not isinstance(v, values.DistributedVariable)
          value_list.append(v)
          #if(v not in self.var_list):
          #  self.var_list.append(v)
      return value_list

    return values.create_mirrored_variable(self._container_strategy(),
                                           _real_mirrored_creator,
                                           values.MirroredVariable,
                                           values.SyncOnReadVariable, **kwargs)

  def _get_collective_ops_info(self):
    group_size = self._num_workers
    group_key = 10000+group_size #always use 1 for Strategy comms
    collective_instance_key = 99 #always use 99 for Strategy communication
    logging.info("EXECUTING EAGERLY? "+ str(context.context().executing_eagerly()))
    if self._is_chief:
      if(hasattr(self, "_collective_keys")):
        collective_init = [self._collective_keys.get_op_instance_key()+1, self._collective_keys._group_key]
      else:
        collective_init = [cross_device_utils.OP_INSTANCE_KEY_START_NUMBER, 1]
      collective_init = ops.convert_to_tensor(
              collective_init, dtype=dtypes.int32)
      bcast_send = collective_ops.broadcast_send(
                  collective_init, collective_init.shape, collective_init.dtype,
                  group_size, group_key, collective_instance_key)
      del bcast_send
    else:
      collective_init = ops.convert_to_tensor(
              [0,0], dtype=dtypes.int32)
      collective_init = collective_ops.broadcast_recv(collective_init.shape,
                                                   dtypes.int32,
                                                   group_size, group_key,
                                                   collective_instance_key)
    return collective_init

  def _initialize_multi_worker(self, cluster_resolver, update_server=False):
    """Initializes the object for multi-worker training."""
    logging.info("start init collective")
    cluster_spec = multi_worker_util.normalize_cluster_spec(
        cluster_resolver.cluster_spec())
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if task_type is None or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`.")
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id

    self._num_workers = multi_worker_util.worker_count(cluster_spec, task_type)
    if not self._num_workers:
      raise ValueError("No `worker`, `chief` or `evaluator` tasks can be found "
                       "in `cluster_spec`.")

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)

    self._worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)

    if(not update_server):
      if (ops.executing_eagerly_outside_functions() and
          not getattr(self, "_local_or_standalone_client_mode", False)):
        context.context().configure_collective_ops(
            collective_leader=multi_worker_util.collective_leader(
                cluster_spec, task_type, task_id),
            scoped_allocator_enabled_ops=("CollectiveReduce",),
            device_filters=("/job:%s/task:%d" % (task_type, task_id),))
        self._collective_ops_configured = True

    # Starting a std server in eager mode and in independent worker mode.
    if (update_server or context.executing_eagerly() and
        not getattr(self, "_std_server_started", False) and
        not getattr(self, "_local_or_standalone_client_mode", False)):
      # Checking _local_or_standalone_client_mode as well because we should not
      # create the std server in standalone client mode.
      config_proto = config_pb2.ConfigProto()
      config_proto = self._update_config_proto(config_proto)

      if hasattr(cluster_resolver, "port"):
        port = cluster_resolver.port
      else:
        port = 0
      server_def = tensorflow_server_pb2.ServerDef(
          cluster=cluster_spec.as_cluster_def(),
          default_session_config=config_proto,
          job_name=task_type,
          task_index=task_id,
          protocol=cluster_resolver.rpc_layer or "grpc",
          port=port)
      #if(update_server):
      #  context.context().update_server_def(server_def)
      #else:
      logging.info("before collective")
      logging.info(threading.active_count())
      
      
      context.context().enable_collective_ops(server_def)
      
      self._std_server_started = True
      # The `ensure_initialized` is needed before calling
      # `context.context().devices()`.
      logging.info("before ensuring")
      print("print | before ensuring ")
      context.context().ensure_initialized()
      logging.info(
          "Enabled multi-worker collective ops with available devices: %r",
          context.context().devices())
      logging.info(threading.active_count())

    # TODO(yuefengz): The `num_gpus` is only for this particular task. It
    # assumes all workers have the same number of GPUs. We should remove this
    # assumption by querying all tasks for their numbers of GPUs.
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    if num_gpus:
      local_devices = tuple("%s/device:GPU:%d" % (self._worker_device, i)
                            for i in range(num_gpus))
    else:
      local_devices = (self._worker_device,)

    #op_col_key = self._get_collective_ops_info().numpy()
    if(not update_server):
      # if(hasattr(self, "_collective_keys")):
      #   del self._collective_keys
      #   del self._cross_device_ops
      #   del self._input_workers
      #   #context.context().remove_function(self._loaded_fn)
      #   if(self._container_strategy() in mirrored_run._cfer_fn_cache):
      #     del mirrored_run._cfer_fn_cache[self._container_strategy()]
        

      
      
      #print("OP_COLLECTIVE KEY IS ",op_col_key)
      #self._collective_keys = cross_device_utils.CollectiveKeys(group_key_start=op_col_key[1], op_instance_key_start=op_col_key[0])
      self._collective_keys = cross_device_utils.CollectiveKeys()
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
          num_workers=self._num_workers,
          num_gpus_per_worker=num_gpus,
          collective_keys=self._collective_keys,
          communication=self._communication)
    super(ElasticCollectiveAllReduceExtended, self)._initialize_single_worker(
        local_devices)
    host_device = device_util.get_host_for_device(self._worker_device)
    self._input_workers = input_lib.InputWorkers(
        [(host_device, self.worker_devices)])
    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = "/job:%s/task:%d" % (task_type, task_id)

    # Save the num_gpus_per_worker and rpc_layer for configure method.
    self._num_gpus_per_worker = num_gpus
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()

    logging.info(
        "MultiWorkerMirroredStrategy with cluster_spec = %r, task_type = %r, "
        "task_id = %r, num_workers = %r, local_devices = %r, "
        "communication = %s", cluster_spec.as_dict(), task_type,
        task_id, self._num_workers, local_devices,
        self._communication)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the object.

    Args:
      session_config: a `tf.compat.v1.ConfigProto`
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type, such as "worker".
      task_id: the current task id.

    Raises:
      ValueError: if `task_type` is not in the `cluster_spec`.
    """
    if cluster_spec:
      # Use the num_gpus_per_worker recorded in constructor since _configure
      # doesn't take num_gpus.
      cluster_resolver = SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_id=task_id,
          num_accelerators={"GPU": self._num_gpus_per_worker},
          rpc_layer=self._rpc_layer)
      self._initialize_multi_worker(cluster_resolver, True)
      assert isinstance(self._get_cross_device_ops(),
                        cross_device_ops_lib.CollectiveAllReduce)

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _experimental_distribute_dataset(self, dataset):
    self._input_dataset = dataset
    input_context = self._make_input_context()
    self._input_dataset_dist = input_lib.get_distributed_dataset(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync,
        input_context=input_context)
    return self._input_dataset_dist

  @property
  def _num_replicas_in_sync(self):
    return len(self.worker_devices) * self._num_workers
  def get_elastic_iterator(self, dataset):
    it = iter(dataset)
    self._iterators.append(it)
    return it
    
    
  def update_cluster(self):

    def find_iterator(iterator):
      bfs_q = Queue.Queue()
      bfs_q.put(iterator)
      visited = []
      while not bfs_q.empty():
        it = bfs_q.get()
        print(it)
        print("\n")
        print(dir(it), it.__dict__)
        visited.append(it)

        if hasattr(it, "_iterator_resource"):
          print(dir(it._iterator_resource), it._iterator_resource.ref())
        if hasattr(it, "_iterators"):
          for input_iters in it._iterators:
            if input_iters not in visited:
              bfs_q.put(input_iters)
        elif hasattr(it, "_iterator"):
          bfs_q.put(it._iterator)
        
      return it


    #context._reset_context()
    #context.ensure_initialized()
    self._initialize_multi_worker(self._cluster_resolver, True)
    context.context().update_group_size(self._num_workers)
    new_ds = self._experimental_distribute_dataset(self._input_dataset)
    new_it = iter(new_ds)
    new_it = find_iterator(new_it)
    # if(isinstance(self._input_dataset_dist._cloned_datasets[0],distribute._AutoShardDataset)):
    #   in_dataset = self._input_dataset_dist._cloned_datasets[0]._input_dataset
    #   print(in_dataset._as_serialized_graph())
    #   self._input_dataset_dist._cloned_datasets[0] = distribute._AutoShardDataset(self._input_dataset, self._num_workers, 0)
    

    for iterator in self._iterators:
      bot_it = find_iterator(iterator)
      from tensorflow.core.framework import graph_pb2
      from tensorflow.python.data.experimental.ops import distribute_options
      graph_def = graph_pb2.GraphDef().FromString(bot_it._dataset._as_serialized_graph(external_state_policy=distribute_options
                                    .ExternalStatePolicy.FAIL).numpy())
      d1 = str(graph_def)
      graph_def = graph_pb2.GraphDef().FromString(new_it._dataset._as_serialized_graph(external_state_policy=distribute_options
                                    .ExternalStatePolicy.FAIL).numpy())
      d2 = str(graph_def)
      
      for i, s in enumerate(difflib.ndiff(d1, d2)):
        if s[0]==' ': continue
        elif s[0]=='-':
            print(u'Delete "{}" from position {}'.format(s[-1],i))
        elif s[0]=='+':
            print(u'Add "{}" to position {}'.format(s[-1],i))
      bot_it.reshard_iterator(new_it._dataset)
      