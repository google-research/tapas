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
"""Pipeline for generating synthetic statements for tables."""

import collections
import enum
import os
import random
from typing import Text, Iterable, Tuple, Any, Optional
import apache_beam as beam
import numpy as np
from tapas.protos import interaction_pb2
from tapas.utils import beam_utils
from tapas.utils import contrastive_statements
from tapas.utils import pretrain_utils
from tapas.utils import synthesize_entablement
from tapas.utils import tf_example_utils
import tensorflow.compat.v1 as tf

_KeyInteraction = Tuple[Text, interaction_pb2.Interaction]
_NS = "main"


class Mode(enum.Enum):
  SYNTHETIC = 1
  CONTRASTIVE = 2
  ALL = 3


class FlumeCounter(synthesize_entablement.Counter):

  def count(self, message):
    beam.metrics.Metrics.counter(_NS, message).inc()




def synthesize_fn(
    key_interaction,
    config,
    add_opposite_table,
    use_fake_table,
):
  """Synthesizes up to 4 statements."""
  key, interaction = key_interaction
  rng = np.random.RandomState(beam_utils.to_numpy_seed(key))
  for new_interaction in synthesize_entablement.synthesize_from_interaction(
      config, rng, interaction, FlumeCounter(), add_opposite_table):
    if use_fake_table:
      _clear_table(new_interaction)
    yield new_interaction.id, new_interaction


def shard_interaction_fn(
    key_interaction):
  """Prepares tables for further processing."""
  beam.metrics.Metrics.counter(_NS, "Interactions").inc()
  key, interaction = key_interaction
  num_cols = len(interaction.table.columns)
  if num_cols < 2:
    beam.metrics.Metrics.counter(_NS, "Invalid number of cols").inc()
    return

  num_rows = len(interaction.table.rows)
  if num_rows < 2:
    beam.metrics.Metrics.counter(_NS, "Invalid number of rows").inc()
    return

  interaction = beam_utils.rekey(interaction)
  # Save the key for debugging.
  interaction.table.document_title = key

  beam.metrics.Metrics.counter(_NS, "Interactions: Valids").inc()
  for new_interaction in beam_utils.get_row_sharded_interactions(
      interaction,
      max_num_cells=50,
  ):
    if len(new_interaction.table.rows) < 2:
      continue
    beam.metrics.Metrics.counter(_NS, "Interactions: Shards").inc()
    yield key, new_interaction


def _to_id(obj):
  return hex(beam_utils.to_numpy_seed(obj))


def _count(message):
  beam.metrics.Metrics.counter(_NS, message).inc()


def _clear_table(interaction):
  table = interaction.table
  del table.columns[:]
  del table.rows[:]
  table.columns.add().text = ""
  table.rows.add().cells.add().text = ""


def _to_contrastive_statements_fn(
    key_interaction,
    use_fake_table,
    drop_without_support_rate,
):
  """Converts pretraining interaction to contrastive interaction."""

  # Make a copy since beam functions should not manipulate inputs.
  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(key_interaction[1])
  interaction = new_interaction

  iid = interaction.table.table_id
  rng = random.Random(beam_utils.to_numpy_seed(iid))

  generated_statements = set()

  for result in contrastive_statements.get_contrastive_statements(
      rng, interaction, count_fn=_count):

    has_support, statement, contrastive_statement = result

    beam.metrics.Metrics.counter(_NS, "Pairs").inc()

    if not has_support and rng.random() < drop_without_support_rate:
      beam.metrics.Metrics.counter(
          _NS, "Pairs: Down-sampled pairs without support").inc()
      continue

    if contrastive_statement in generated_statements:
      beam.metrics.Metrics.counter(_NS, "Pairs: Duplicates").inc()
      continue

    generated_statements.add(contrastive_statement)

    new_interaction = interaction_pb2.Interaction()
    new_interaction.CopyFrom(interaction)
    del new_interaction.questions[:]

    new_interaction.id = _to_id((
        iid,
        (statement, contrastive_statement),
    ))

    if use_fake_table:
      _clear_table(new_interaction)

    new_interaction.table.table_id = new_interaction.id

    new_question = new_interaction.questions.add()
    new_question.id = _to_id((iid, statement))
    new_question.original_text = statement
    new_question.answer.class_index = 1

    new_question = new_interaction.questions.add()
    new_question.id = _to_id((iid, contrastive_statement))
    new_question.original_text = contrastive_statement
    new_question.answer.class_index = 0

    beam.metrics.Metrics.counter(_NS, "Pairs emitted").inc()
    yield new_interaction.id, new_interaction


class ToClassifierTensorflowExample(beam.DoFn):
  """Class for converting finetuning examples."""

  def __init__(
      self,
      config,
  ):
    self._config = config

  def start_bundle(self):
    self._converter = tf_example_utils.ToClassifierTensorflowExample(
        self._config)

  def process(
      self,
      key_interaction,
  ):
    _, interaction = key_interaction
    for index, question in enumerate(interaction.questions):
      beam.metrics.Metrics.counter(_NS, "Input question").inc()
      try:
        example = self._converter.convert(interaction, index)
        yield question.id, example
        beam.metrics.Metrics.counter(_NS, "Conversion success").inc()
      except ValueError as e:
        beam.metrics.Metrics.counter(_NS, "Conversion error").inc()
        beam.metrics.Metrics.counter(_NS, str(e)).inc()


def build_pipeline(
    mode,
    config,
    use_fake_table,
    add_opposite_table,
    drop_without_support_rate,
    input_file,
    output_dir,
    output_suffix,
    conversion_config,
    num_splits = 100,
):
  """Builds the pipeline."""

  def _pipeline(root):
    """Pipeline."""

    interactions = (
        pretrain_utils.read_interactions(root, input_file, name="input")
        | "Preprocess" >> beam.FlatMap(shard_interaction_fn))

    data = []

    if mode in [Mode.CONTRASTIVE, Mode.ALL]:

      data.append(
          interactions
          | "ToContrastivePairs" >> beam.FlatMap(
              _to_contrastive_statements_fn,
              use_fake_table=use_fake_table,
              drop_without_support_rate=drop_without_support_rate,
          ))

    if mode in [Mode.SYNTHETIC, Mode.ALL]:

      data.append(interactions
                  | "Synthesize" >> beam.FlatMap(
                      synthesize_fn,
                      config=config,
                      add_opposite_table=add_opposite_table,
                      use_fake_table=use_fake_table,
                  ))

    if not data:
      raise ValueError(f"Unknown mode: {mode}")

    output_data = (data | "Flatten" >> beam.Flatten())
    proto_message = interaction_pb2.Interaction

    if conversion_config is not None:
      pretrain_utils.write_proto_outputs(
          os.path.join(output_dir, "interactions") + output_suffix,
          "interactions",
          output_data,
          interaction_pb2.Interaction,
      )
      output_data = (
          output_data | "ToExamples" >> beam.ParDo(
              ToClassifierTensorflowExample(conversion_config)))
      proto_message = tf.train.Example

    pretrain_utils.split_by_table_id_and_write(
        (output_data | "Reshuffle" >> beam.transforms.util.Reshuffle()),
        output_dir,
        train_suffix=output_suffix,
        test_suffix=output_suffix,
        num_splits=num_splits,
        proto_message=proto_message)

  return _pipeline
