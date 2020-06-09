# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
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
# Lint as: python3

import os
import tempfile

from absl.testing import parameterized
from tapas.datasets import dataset
import tensorflow.compat.v1 as tf



tf.disable_v2_behavior()


def write_tf_example(filename, data_format, features):
  example = tf.train.Example(features=tf.train.Features(feature=features))
  if data_format == "tfrecord":
    with tf.io.TFRecordWriter(filename) as writer:
      writer.write(example.SerializeToString())
  else:
    raise ValueError("Unsupported data_format: {}".format(data_format))


class DatasetTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(DatasetTest, self).setUp()

    # We add a prefix because the dataset API matches files and then sort them
    # lexicographically.
    self._file1 = tempfile.mktemp(prefix="1", suffix="test")
    self._file2 = tempfile.mktemp(prefix="2", suffix="test-00010-of-00020")
    self._file_patterns = [
        self._file1,
        # We use a ? to check that glob mechanism works.
        self._file2.replace("00010-of-00020", "000?0-of-00020")
    ]

    # Creates empty files to avoid errors in tearDown when self.cached_session()
    # is executed.
    open(self._file1, "a").close()
    open(self._file2, "a").close()

    self._file_patterns = [self._file1, self._file2]

  def tearDown(self):
    super(DatasetTest, self).tearDown()

    os.remove(self._file1)
    os.remove(self._file2)

  @parameterized.named_parameters(
      ("train_f1_f2", "tfrecord", True, dict(batch_size=2), (True, True)),
      ("train_f1", "tfrecord", True, dict(batch_size=1), (True, False)),
      ("train_f2", "tfrecord", True, dict(batch_size=1), (False, True)),
      ("test_f1_f2", "tfrecord", False, dict(batch_size=2, cycle_length=1),
       (True, True)),
      ("test_f1", "tfrecord", False, dict(batch_size=1, cycle_length=1),
       (True, False)),
      ("test_f2", "tfrecord", False, dict(batch_size=1, cycle_length=1),
       (False, True)))
  def test_read_dataset(self, data_format, is_training, params,
                        include_patterns):
    write_tf_example(
        self._file1, data_format, {
            "name":
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"one"])),
            "number":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        })
    write_tf_example(
        self._file2, data_format, {
            "name":
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"two"])),
            "number":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        })

    feature_types = {
        "name": tf.io.FixedLenFeature([], tf.string),
        "number": tf.io.FixedLenFeature([], tf.int64),
    }

    parse_fn = dataset.build_parser_function(feature_types, params)

    def filter_fn(xs):
      return [x for (x, include) in zip(xs, include_patterns) if include]

    patterns = filter_fn(self._file_patterns)
    ds = dataset.read_dataset(
        parse_fn,
        "dataset",
        patterns,
        data_format,
        compression_type="",
        is_training=is_training,
        params=params,
    )
    feature_tuple = tf.data.make_one_shot_iterator(ds).get_next()

    with self.cached_session() as sess:
      feature_tuple = sess.run(feature_tuple)

    if params["batch_size"] == 1:
      self.assertIsInstance(feature_tuple, dict)
    else:
      self.assertLen(feature_tuple, params["batch_size"])

    if not is_training:
      expected_names = filter_fn([b"one", b"two"])
      expected_numbers = filter_fn([1, 2])
      self.assertSequenceEqual(list(feature_tuple["name"]), expected_names)
      self.assertSequenceEqual(list(feature_tuple["number"]), expected_numbers)

  @parameterized.named_parameters(
      ("tfrecord", "tfrecord"))
  def test_read_dataset_test_shape_is_fully_known(self, data_format):
    write_tf_example(self._file1, data_format, {
        "number": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
    })
    feature_types = {
        "number": tf.io.FixedLenFeature([], tf.int64),
    }
    params = {"batch_size": 5}
    parse_fn = dataset.build_parser_function(feature_types, params)
    ds = dataset.read_dataset(
        parse_fn,
        "dataset",
        file_patterns=[self._file1],
        data_format=data_format,
        compression_type="",
        is_training=True,
        params=params,
    )
    feature_tuple = tf.data.make_one_shot_iterator(ds).get_next()
    feature_tuple["number"].shape.assert_is_fully_defined()


if __name__ == "__main__":
  tf.test.main()
