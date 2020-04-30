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
"""Contains dataset utility functions."""

from typing import Iterable, Text, Callable, Any, Mapping, Optional
import tensorflow.compat.v1 as tf


ParseExampleFn = Callable[[Any], Mapping[Text, Any]]


def read_dataset(
    parse_examples_fn,
    name,
    file_patterns,
    data_format,
    compression_type,
    is_training,
    params,
    max_eval_count = 50000,
):
  """Returns an input_fn that can be used with the tf.Estimator API."""
  with tf.variable_scope(name):
    batch_size = params["batch_size"]
    # This is used mainly by the test to remove any source of randomness.
    cycle_length = params.get("cycle_length", 64)

    dataset = tf.data.Dataset.list_files(file_patterns, shuffle=is_training)

    if is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      if data_format == "tfrecord":
        buffer_size = 8 * 1024 * 1024  # 8 MiB per file
        return tf.data.TFRecordDataset(
            filename,
            buffer_size=buffer_size,
            compression_type=compression_type,
        )
      raise ValueError("Unsupported data_format: {}".format(data_format))

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            fetch_dataset, sloppy=is_training, cycle_length=cycle_length))

    if is_training:
      dataset = dataset.shuffle(1024)
    else:
      if max_eval_count is not None:
        dataset = dataset.take(max_eval_count)

    parse_fn = parse_examples_fn

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            parse_fn,
            batch_size=batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=is_training))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def build_parser_function(feature_types,
                          params):
  """Returns a parse function that can be used by read_dataset."""
  del params

  def parse_examples(serialized_examples):
    features = tf.io.parse_single_example(serialized_examples, feature_types)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(features.keys()):
      t = features[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
        features[name] = t
    return features

  return parse_examples
