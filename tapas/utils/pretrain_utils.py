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
from typing import Iterable, Tuple, Text, Optional, Union

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

_Proto = Union[interaction_pb2.Interaction, interaction_pb2.Table,
               tf.train.Example]


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


def _parse_text_proto(
    text_proto_line,
    proto_message,
):
  message = text_format.Parse(text_proto_line, proto_message())
  return (_get_input_id(message), message)


def _parse_text_interaction(text_proto_line,):
  output = _parse_text_proto(text_proto_line, interaction_pb2.Interaction)
  assert isinstance(output[1], interaction_pb2.Interaction)
  return output


def _proto_to_text(message):
  return text_format.MessageToString(message, as_one_line=True)


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


def partition_fn(
    example,
    partition_count,
    num_splits,
):
  """Partitions example into train/test based on hash of table id."""
  assert partition_count == 2
  example_id = _get_table_id(example[1])
  shard = to_numpy_seed(example_id) % num_splits
  if shard == 0:
    return 1
  return 0


def write_proto_outputs(output_file, name, data, proto_message):
  """Write protos to a container."""
  if output_file.endswith((".txtpb.gz", ".txtpb")):
    _ = (
        data
        | "DropKey_%s" % name >> beam.Values()
        | "ToTextProto" % name >> beam.Map(
            _proto_to_text,
            proto_message=proto_message,
        )
        | "WriteTextExamples_%s" % name >> beam.io.WriteToText(output_file))
    return
  elif output_file.endswith(".tfrecord"):
    _ = (
        data
        | "DropKey_%s" % name >> beam.Values()
        | "WriteTFRecordsExamples_%s" % name >> beam.io.WriteToTFRecord(
            file_path_prefix=output_file,
            shard_name_template="",
            coder=beam.coders.ProtoCoder(proto_message)))
    return
  raise ValueError(f"Unsupported output format: {output_file}")


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
      | "Partition" >> beam.Partition(partition_fn, 2, num_splits))

  for name, suffix, data in zip(
      ["train", "test"],
      [train_suffix, test_suffix],
      [train, test],
  ):
    output_file = os.path.join(output_dir, name + suffix)
    write_proto_outputs(output_file, name, data, proto_message)


def pair_with_none_fn(
    element):
  key, interaction = element
  yield key, (interaction, None)


def _get_input_id(element):
  if isinstance(element, interaction_pb2.Interaction):
    return str(element.id)
  if isinstance(element, interaction_pb2.Table):
    return str(element.table_id)
  if isinstance(element, tf.train.Example):
    question_id = element.features.feature["question_id"].bytes_list.value[0]
    return question_id.decode("utf8")
  raise ValueError(f"Cannot extract id: {type(element)}")


def _get_table_id(element):
  if isinstance(element, interaction_pb2.Interaction):
    return str(element.table.table_id)
  if isinstance(element, interaction_pb2.Table):
    return str(element.table_id)
  if isinstance(element, tf.train.Example):
    table_id = element.features.feature["table_id"].bytes_list.value[0]
    return table_id.decode("utf8")
  raise ValueError(f"Cannot extract table id: {type(element)}")


def read_inputs(root, input_file, name, proto_message):
  """Reads interaction or table protos."""
  if input_file.endswith((".txtpb.gz", ".txtpb")):
    return (root | f"Read {name}" >> beam.io.textio.ReadFromText(input_file)
            | "Pre-Shuffle" >> beam.transforms.util.Reshuffle()
            | "Parse Text Proto" >> beam.Map(_parse_text_interaction))
  value_coder = beam.coders.ProtoCoder(proto_message)
  if input_file.endswith(".tfrecord") or input_file.endswith(".tfrecords"):
    return (
        root | "Read Inputs %s" % name >> beam.io.ReadFromTFRecord(
            file_pattern=input_file, coder=value_coder, validate=False)
        | "Key Inputs_%s" % name >>
        beam.Map(lambda interaction: (_get_input_id(interaction), interaction))
        | "Reshuffle_%s" % name >> beam.transforms.util.Reshuffle())
  raise ValueError(f"Unsupported input format: {input_file}")


def read_interactions(root, input_file, name):
  return read_inputs(
      root, input_file, name, proto_message=interaction_pb2.Interaction)
