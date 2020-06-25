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
"""Tests for CollectiveAllReduceStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import os
from multiprocessing import Process, Queue
import time


from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.distribute.elastic_collective_all_reduce_strategy import ElasticCollectiveAllReduceStrategy
from tensorflow.python.distribute.cluster_resolver.elastic_cluster_resolver import ElasticClusterResolver
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.mixed_precision.experimental import test_util as mp_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import training_util
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scale_optimizer
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import gradient_descent


class BadClusterBase(test.TestCase):
  """ Base class for testing elastic multi node and dataset, using a bad cluster resolution"""

  def add_node(self, task_id):
    #import shutil
    cluster_folder = self._cluster_folder
    port = 5000+int(task_id)
    with open(cluster_folder+"/localhost:%d"%(port), 'w') as fp:
      fp.write("%d"%(port))
    time.sleep(1)
  
  def remove_node(self, task_id):
    cluster_folder = self._cluster_folder
    port = 5000+int(task_id)
    os.remove(cluster_folder+"/localhost:%d"%(port))

  def setUp(self):
    self._cluster_folder = self.get_temp_dir()
    self._procs = {}
    self._exclude_procs=[]
    import tensorflow_datasets as tfds
    self.dataset_dir = "/tmp/mnist_dataset"
    self.mnist = tfds.builder('mnist', data_dir=self.dataset_dir)
    self.mnist.download_and_prepare()

  def setUpModel(self):
    """Constructs the ML model used to predict handwritten digits."""

    image = keras.layers.Input(shape=(28, 28, 1))

    y = keras.layers.Conv2D(filters=32,
                              kernel_size=5,
                              padding='same',
                              activation='relu')(image)
    y = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(y)
    y = keras.layers.Conv2D(filters=32,
                              kernel_size=5,
                              padding='same',
                              activation='relu')(y)
    y = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(y)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(1024, activation='relu')(y)
    y = keras.layers.Dropout(0.4)(y)

    probs = keras.layers.Dense(10, activation='softmax')(y)

    model = keras.models.Model(image, probs, name='mnist')

    return model

  def setUpDataset(self, batch_size):
    import tensorflow_datasets as tfds
    @tfds.decode.make_decoder(output_dtype=dtypes.float32)
    def decode_image(example, feature):
      """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
      return math_ops.cast(feature.decode_example(example), dtype=dtypes.float32) / 255
    
    mnist_train, mnist_test = self.mnist.as_dataset(
        split=['train', 'test'],
        decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
        as_supervised=True)
    options = dataset_ops.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    train_input_dataset = mnist_train.repeat().shuffle(
        buffer_size=50000).batch(batch_size)
    eval_input_dataset = mnist_test.repeat().batch(batch_size)

    self.train_input_dataset = train_input_dataset.with_options(options)
    self.eval_input_dataset = eval_input_dataset.with_options(options)

  def start_node(self, target, params):
    task_id = params["task_id"]
    if(task_id in self._procs and self._procs[task_id] is not None):
      raise ValueError("Task already assigned to that task id")
    t = Process(target=target, kwargs=params)
    t.start()
    self._procs[task_id] = t

  def end_node(self, task_id):
    if(task_id in self._procs):
      self._procs[task_id].terminate()
      self._exclude_procs.append(task_id)
    else:
      raise ValueError("Task %d not started"%(task_id))

  def join_all(self):
    for ii in self._procs:
      self._procs[ii].join()
      print(self._procs[ii])
      if(self._procs[ii].exitcode != 0 and ii not in self._exclude_procs):
        raise RuntimeError("Process %d failed with exitcode %d but wasn't in exclude list"%(ii, self._procs[ii].exitcode))

  def _run_test_fn(self, task_id, batch_size):
    def remove_my_node(signum, frame):
      self.remove_node(task_id)
      exit(1)
    from signal import signal, SIGABRT, SIGKILL, SIGTERM
    for sig in (SIGABRT, SIGTERM):
      signal(sig, remove_my_node)

    def get_cluster_spec():
      cluster_folder = self._cluster_folder
      cluster_list = sorted(os.listdir(cluster_folder))
      cluster_spec = {}
      cluster_spec["cluster"] = {"worker": cluster_list}
      return cluster_spec

    
    self.add_node(task_id)
    strategy = ElasticCollectiveAllReduceStrategy(cluster_resolver=ElasticClusterResolver(get_cluster_spec, task_id, task_type="worker"))
    self.setUpDataset(batch_size)
    with strategy.scope():
      lr_schedule = learning_rate_schedule.ExponentialDecay(
          0.05, decay_steps=100000, decay_rate=0.96)
      optimizer = gradient_descent.SGD(learning_rate=lr_schedule)
      model = self.setUpModel()
      model.compile(
          optimizer=optimizer,
          run_eagerly=False,
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])
  
    model.fit(
        self.train_input_dataset,
        epochs=1,
        steps_per_epoch=200)

  def _run_dataset_fn(self, task_id, batch_size, ret_queue):
    import time
    def remove_my_node(signum, frame):
      self.remove_node(task_id)
      exit(1)
    from signal import signal, SIGABRT, SIGKILL, SIGTERM
    for sig in (SIGABRT, SIGTERM):
      signal(sig, remove_my_node)

    start = time.time()
    def get_cluster_spec():
      cluster_folder = self._cluster_folder
      cluster_list = sorted(os.listdir(cluster_folder))
      cluster_spec = {}
      cluster_spec["cluster"] = {"worker": cluster_list}
      print("Cluster spec is ", cluster_spec)
      return cluster_spec

    self.add_node(task_id)
    strategy = ElasticCollectiveAllReduceStrategy(cluster_resolver=ElasticClusterResolver(get_cluster_spec, task_id, task_type="worker"))
    self.setUpDataset(batch_size)
    dds = strategy.experimental_distribute_dataset(self.train_input_dataset)
    it = strategy.extended.get_elastic_iterator(dds)
    i = next(it)
    first_batch = len(i[0])
    logging.warning("time to first sleep is %f"%(time.time() - start))
    time.sleep(30)
    strategy.extended.update_cluster()
    i = next(it)
    i = next(it)
    i = next(it)
    last_batch = len(i[0])
    ret_queue.put((first_batch, last_batch))
    return first_batch, last_batch

class ElasticCollectiveAllReduceStrategyTestBase(
    BadClusterBase):

  def test_growth_iterator(self):
    #Start with 2 nodes, wait 30ish seconds, add a 3rd
    ret_queue = Queue()
    batch_size = 300
    params = {"task_id":None, "batch_size":batch_size, "ret_queue":ret_queue}
    for ii in range(2):
      params["task_id"] = ii
      self.start_node(self._run_dataset_fn, params)
    
    time.sleep(10)
    params["task_id"] = 2
    self.start_node(self._run_dataset_fn, params)
    time.sleep(10)

    self.join_all()
    first_batches = []
    last_batches = []
    while not ret_queue.empty():
      t = ret_queue.get()
      first_batches.append(t[0])
      last_batches.append(t[1])

    assert set(first_batches) == set([batch_size/2, batch_size/2, batch_size/3])
    assert set(last_batches) == set([batch_size/3, batch_size/3, batch_size/3])
    
    

  def test_shrink_iterator(self):
    #Start with 2 nodes, wait 30ish seconds, add a 3rd
    batch_size = 300
    ret_queue = Queue()
    params = {"task_id":None, "batch_size":batch_size, "ret_queue":ret_queue}
    for ii in range(3):
      params["task_id"] = ii
      self.start_node(self._run_dataset_fn, params)
    
    time.sleep(10)
    self.end_node(2)
    time.sleep(10)

    self.join_all()
    first_batches = []
    last_batches = []
    while not ret_queue.empty():
      t = ret_queue.get()
      first_batches.append(t[0])
      last_batches.append(t[1])

    assert set(first_batches) == set([batch_size/3, batch_size/3, batch_size/3])
    assert set(last_batches) == set([batch_size/2, batch_size/2])
  
  def test_grow_keras(self):
    #Start with 2 nodes, wait 30ish seconds, add a 3rd
    batch_size = 300
    params = {"task_id":None, "batch_size":batch_size}
    for ii in range(2):
      params["task_id"] = ii
      self.start_node(self._run_test_fn, params)
    
    time.sleep(10)
    params["task_id"] = 2
    self.start_node(self._run_test_fn, params)
    time.sleep(10)

    self.join_all()
    

  def test_shrink_keras(self):
    #Start with 2 nodes, wait 30ish seconds, add a 3rd
    batch_size = 300
    params = {"task_id":None, "batch_size":batch_size}
    for ii in range(3):
      params["task_id"] = ii
      self.start_node(self._run_test_fn, params)
    
    time.sleep(10)
    self.end_node(2)
    
    self.join_all()

if __name__ == '__main__':
  test.main()
