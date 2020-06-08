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
"""Beam functions for converting interactions to TF examples."""

import os
import random
from typing import Iterable, Tuple, Text, Optional, List, Union

import apache_beam as beam

from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from tapas.utils import tf_example_utils
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

_NS = "main"
_KeyInteraction = Tuple[Text, interaction_pb2.Interaction]
_KeyInteractionTable = Tuple[Text, Tuple[interaction_pb2.Interaction,
                                         Optional[interaction_pb2.Table]]]


def fingerprint(key):
  return "%08X" % abs(tf_example_utils.fingerprint(key))


def prepand_fingerprint(key):
  return "%s_%s" % (fingerprint(key), key)


def _has_valid_shape(table):
  """Returns true if table has a rectangular shape."""
  if not table.columns:
    return False
  if not table.rows:
    return False
  num_columns = len(table.columns)
  for row in table.rows:
    if len(row.cells) != num_columns:
      return False
  return True


def check_table_id_fn(
    key_interaction):
  """Adds interaction, table and question id if missing."""
  key, interaction = key_interaction

  if not _has_valid_shape(interaction.table):
    beam.metrics.Metrics.counter(_NS, "Tables empty or of ragged shape").inc()
    return

  if interaction.id and interaction.table.table_id and all(
      bool(q.id) for q in interaction.questions):
    yield key_interaction
  else:
    new_interaction = interaction_pb2.Interaction()
    new_interaction.CopyFrom(interaction)
    for question in new_interaction.questions:
      if not question.id:
        question.id = key
        beam.metrics.Metrics.counter(_NS, "Question Ids added").inc()
    if not new_interaction.table.table_id:
      new_interaction.table.table_id = key
      beam.metrics.Metrics.counter(_NS, "Table Ids added").inc()
    if not new_interaction.id:
      new_interaction.id = key
      beam.metrics.Metrics.counter(_NS, "Interaction Ids added").inc()
    yield key, new_interaction


def check_tale_size_fn(key_interaction, min_num_rows,
                       min_num_columns):
  """Filters tables with too few rows or columns."""
  beam.metrics.Metrics.counter(_NS, "Inputs").inc()
  _, interaction = key_interaction
  num_rows = len(interaction.table.rows)
  if num_rows < min_num_rows:
    beam.metrics.Metrics.counter(_NS, "Inputs: Too few rows").inc()
    return
  num_cols = len(interaction.table.columns)
  if num_cols < min_num_columns:
    beam.metrics.Metrics.counter(_NS, "Inputs: Too few columns").inc()
    return
  yield key_interaction


def add_numeric_values_fn(element):
  key, interaction = element
  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(interaction)
  number_annotation_utils.add_numeric_values(new_interaction)
  return key, new_interaction


def duplicate_fn(key_interaction,
                 dupe_factor):
  key, interaction = key_interaction
  for dupe_index in range(dupe_factor):
    new_id = "%s_%d" % (key, dupe_index)
    yield new_id, interaction


def _parse_interaction(text_proto_line):
  interaction = text_format.Parse(text_proto_line,
                                  interaction_pb2.Interaction())
  return (interaction.id, interaction)


class ToTensorflowExample(beam.DoFn):
  """Class for converting pretraining examples."""

  def __init__(self, config):
    self._config = config

  def start_bundle(self):
    self._converter = tf_example_utils.ToPretrainingTensorflowExample(
        self._config)

  def process(
      self,
      element):
    beam.metrics.Metrics.counter(_NS, "Interactions").inc()

    key, (interaction, random_table) = element

    seed = tf_example_utils.fingerprint(
        "%s_%d_%d" %
        (key, self._config.random_seed, self._config.max_seq_length))
    rng = random.Random(seed)
    example = self._converter.convert(rng, interaction, random_table)
    if example:
      beam.metrics.Metrics.counter(_NS, "Examples").inc()
      yield prepand_fingerprint(key), example


_MAX_INT = 2**32 - 1


def to_numpy_seed(obj):
  return tf_example_utils.fingerprint(repr(obj)) % _MAX_INT


def _partition_fn(example,
                  partition_count, num_splits):
  """Partitions example into train/test based on hash of table id."""
  assert partition_count == 2

  if isinstance(example, interaction_pb2.Interaction):
    example_id = example.table.table_id
  elif isinstance(example, tf.train.Example):
    example_id = example.features.feature["table_id"].bytes_list.value[0]
  else:
    raise ValueError(f"Unexpected type: {type(example)}")

  shard = to_numpy_seed(example_id) % num_splits
  if shard == 0:
    return 1
  return 0


def split_by_table_id_and_write(
    examples,
    output_dir,
    train_suffix = ".tfrecord",
    test_suffix = ".tfrecord",
    num_splits = 100,
    proto_message=interaction_pb2.Interaction,
):
  """Split interactions into train and test and write them to disc."""
  train, test = (
      examples
      | "Partition" >> beam.Partition(_partition_fn, 2, num_splits))

  for name, suffix, data in zip(
      ["train", "test"],
      [train_suffix, test_suffix],
      [train, test],
  ):
    output_file = os.path.join(output_dir, name + suffix)
    _ = (
        data
        | "WriteTFRecordsExamples_%s" % name >> beam.io.WriteToTFRecord(
            file_path_prefix=output_file,
            shard_name_template="",
            coder=beam.coders.ProtoCoder(proto_message)))


def pair_with_none_fn(
    element):
  key, interaction = element
  yield key, (interaction, None)


def build_pretrain_data_pipeline(
    input_files,
    output_dir,
    config,
    dupe_factor,
    min_num_rows,
    min_num_columns,
):
  """Maps pre-training interactions to TF examples."""

  def pipeline(root):

    lines = []
    for filename in input_files:
      lines.append(
          root | f"Read {filename}" >> beam.io.textio.ReadFromText(filename))

    examples = (
        lines | "Flatten" >> beam.Flatten()
        | "Pre-Shuffle" >> beam.transforms.util.Reshuffle()
        | "Parse" >> beam.Map(_parse_interaction)
        | "CheckTableId" >> beam.FlatMap(check_table_id_fn)
        | "CheckTableSize" >> beam.FlatMap(check_tale_size_fn, min_num_rows,
                                           min_num_columns)
        | "AddNumericValues" >> beam.Map(add_numeric_values_fn)
        | "Duplicate" >> beam.FlatMap(duplicate_fn, dupe_factor)
        | "PairWithNone" >> beam.FlatMap(pair_with_none_fn)
        | "ToTensorflowExample" >> beam.ParDo(ToTensorflowExample(config))
        | "DropKey" >> beam.Map(lambda id_ex: id_ex[1])
        | "Post-Shuffle" >> beam.transforms.util.Reshuffle())

    split_by_table_id_and_write(
        examples,
        output_dir,
        proto_message=tf.train.Example,
    )

  return pipeline
