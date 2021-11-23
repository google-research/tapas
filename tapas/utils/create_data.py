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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

import enum
import os
import random
from typing import Iterable, List, Optional, Sequence, Text, Tuple

from absl import logging
import apache_beam as beam
from tapas.protos import annotated_text_pb2
from tapas.protos import interaction_pb2
from tapas.protos import negative_retrieval_examples_pb2
from tapas.retrieval import tf_example_utils as retrieval_utils
from tapas.utils import pretrain_utils
from tapas.utils import text_utils
from tapas.utils import tf_example_utils
import tensorflow.compat.v1 as tf


_fingerprint = pretrain_utils.fingerprint
_prepand_fingerprint = pretrain_utils.prepand_fingerprint

_NS = 'main'
_KeyInteraction = Tuple[Text, interaction_pb2.Interaction]
_KeyInteractionTable = Tuple[Text, Tuple[interaction_pb2.Interaction,
                                         Optional[interaction_pb2.Table]]]

_NegativeRetrievalExamples = negative_retrieval_examples_pb2.NegativeRetrievalExamples

_annotated_cell = annotated_text_pb2.AnnotatedText.annotated_cell_ext
_annotated_question = annotated_text_pb2.AnnotatedText.annotated_question_ext

_read_inputs = pretrain_utils.read_inputs
read_interactions = pretrain_utils.read_interactions


class ConverterImplType(enum.Enum):
  PYTHON = 1


def _get_key(example):
  key, _ = example
  return key


def _key_with_random_key(
    example,
    seed,
    num_random_table_bins,
    num_corpus_bins,
):
  """Computes a new random key (train/test split aware) for random pairing."""
  key, _ = example
  seed = seed + tf_example_utils.fingerprint(key)
  rng = random.Random(seed)
  # Add train/test information so that random buckets don't cross over.
  partition = pretrain_utils.partition_fn(example, 2, num_corpus_bins)
  return f'{partition}_{rng.randint(0, num_random_table_bins)}', example


def _get_samples(key, rate):
  """Maps a sampling rate into an integer sample count deterministically.

  Args:
    key: Key to use for hashing function.
    rate: Sample rate.

  Returns:
    sample_count. If `rate` is an integer, the function returns `rate`. For the
      decimal part we sample a random number from [0, 1) and add one if sampled]
      float is below the mantissa.
  """
  int_rate = int(rate)
  if int_rate == rate:  # Not really needed but adding to be safe.
    return int_rate
  decimal_part = rate - int_rate
  return int_rate + int(random.Random(key).random() < decimal_part)


def _merge_random_interactions(
    key_interactions
):
  """Merges random table with interaction pair."""
  _, interactions = key_interactions

  first_key = None
  first_interaction = None
  prev_table = None

  for key, interaction in interactions:
    beam.metrics.Metrics.counter(_NS, 'Interactions').inc()

    if first_interaction is None or first_key is None or prev_table is None:
      first_key = key
      first_interaction = interaction
      prev_table = interaction.table
      continue

    if prev_table.table_id != interaction.table.table_id:
      yield key, (interaction, prev_table)
    else:
      beam.metrics.Metrics.counter(
          _NS, 'Interactions without random interaction').inc()
      yield key, (interaction, None)

  assert first_key and first_interaction and prev_table
  if first_interaction.table.table_id != prev_table.table_id:
    yield first_key, (first_interaction, prev_table)
  else:
    beam.metrics.Metrics.counter(
        _NS, 'Interactions without random interaction').inc()
    yield first_key, (first_interaction, None)


def insert_fake_table_fn(key_interaction):
  """Replaces all tables with a fake table."""
  key, interaction = key_interaction

  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(interaction)
  del new_interaction.table.rows[:]
  del new_interaction.table.columns[:]
  new_interaction.table.columns.add()
  new_interaction.table.rows.add().cells.add()
  return key, new_interaction


def _add_text_fn(element):
  key, interaction = element
  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(interaction)
  text_utils.filter_invalid_unicode_from_table(new_interaction.table)
  for question in new_interaction.questions:
    question.text = text_utils.normalize_for_match(question.original_text)
  return key, new_interaction


def build_pretraining_pipeline(
    input_file,
    output_dir,
    output_suffix,
    config,
    dupe_factor,
    min_num_rows,
    min_num_columns,
    num_random_table_bins = 1_000,
    add_random_table = False,
    num_corpus_bins = 1_000,
    add_numeric_values = True,
):
  """Pipeline that maps interactions to TF examples.

  Args:
    input_file: Input in Interaction proto format.
    output_dir: Output directory.
    output_suffix: Extension to use to save the train/test files.
    config: Pretrain config.
    dupe_factor: Number of duplicates per input.
    min_num_rows: Min num of rows required to create example.
    min_num_columns: Min num of cols required to create example.
    num_random_table_bins: This is for sampling negative examples smaller number
      will result in a higher number of negatives but might be slow.
    add_random_table: If true, pair 50% of examples with random tables.
    num_corpus_bins: Number of bins for random sharding. Determines test size.
    add_numeric_values: If true, parse numeric values (needed for ranks).

  Returns:
    The pipeline function.
  """

  def _pipeline(root):
    """Pipeline."""

    interactions = read_interactions(root, input_file, name='input')

    interactions = (
        interactions
        | 'CheckTableId' >> beam.FlatMap(pretrain_utils.check_table_id_fn)
        | 'CheckTableSize' >> beam.FlatMap(pretrain_utils.check_tale_size_fn,
                                           min_num_rows, min_num_columns))
    if add_numeric_values:
      interactions = (
          interactions |
          'AddNumericValues' >> beam.Map(pretrain_utils.add_numeric_values_fn))
    else:
      interactions = (interactions | 'AddText' >> beam.Map(_add_text_fn))
    interactions = (
        interactions
        | 'Duplicate' >> beam.FlatMap(pretrain_utils.duplicate_fn, dupe_factor))

    if add_random_table:
      interactions = (
          interactions | 'Key With Random Key' >> beam.Map(
              _key_with_random_key, config.random_seed, num_random_table_bins,
              num_corpus_bins)
          | 'GroupByKey' >> beam.GroupByKey()
          |
          'MergeRandomInteractions' >> beam.FlatMap(_merge_random_interactions))
    else:
      interactions = (
          interactions
          | 'PairWithNone' >> beam.FlatMap(pretrain_utils.pair_with_none_fn))

    examples = (
        interactions
        | f'ToTensorflowExample_{config.max_seq_length}' >> beam.ParDo(
            pretrain_utils.ToTensorflowExample(config)))

    pretrain_utils.split_by_table_id_and_write(
        examples,
        output_dir=output_dir,
        train_suffix=output_suffix,
        test_suffix=output_suffix,
        proto_message=tf.train.Example,
        num_splits=num_corpus_bins,
    )

  return _pipeline


class ToClassifierTensorflowExample(beam.DoFn):
  """Class for converting finetuning examples."""

  def __init__(
      self,
      config,
      name,
      convert_impl_value,
  ):
    self._config = config
    self._name = name
    self._convert_impl_value = convert_impl_value

  def start_bundle(self):
    convert_impl = ConverterImplType(self._convert_impl_value)
    if convert_impl == ConverterImplType.PYTHON:
      self._converter = tf_example_utils.ToClassifierTensorflowExample(
          self._config)
    else:
      raise ValueError(f'Unsupported implementation: {convert_impl.name}')

  def process(
      self,
      element):
    key, interaction = element
    for index, question in enumerate(interaction.questions):
      beam.metrics.Metrics.counter(_NS, 'Input question').inc()
      try:
        example = self._converter.convert(interaction, index)
        # The default value of 1 means no up-sumpling nor downsampling
        rate = self._config.label_sampling_rate.get(
            (self._name, question.answer.class_index), 1)
        fingerprinted_key = _prepand_fingerprint(f'{key}_{index}')
        beam.metrics.Metrics.counter(_NS, 'Conversion success').inc()
        # We upsample if rate is bigger than 1 and downsample when it's smaller.
        for _ in range(_get_samples(fingerprinted_key, rate)):
          beam.metrics.Metrics.counter(_NS, 'Example emitted').inc()
          yield fingerprinted_key, example

      except ValueError as e:
        beam.metrics.Metrics.counter(_NS, 'Conversion error').inc()
        beam.metrics.Metrics.counter(_NS, str(e)).inc()


def _to_interaction_fn(
    element):
  key, table = element
  interaction = interaction_pb2.Interaction()
  interaction.table.CopyFrom(table)
  interaction.id = table.table_id
  return key, interaction


def read_tables(root, input_file, name):
  inputs = _read_inputs(
      root, input_file, name, proto_message=interaction_pb2.Table)
  return inputs | f'to_interaction {name}' >> beam.Map(_to_interaction_fn)


def read_only_tables(root, input_file, name):
  return _read_inputs(
      root, input_file, name, proto_message=interaction_pb2.Table)


def build_classifier_pipeline(
    input_files,
    output_files,
    config,
    use_fake_tables = False,
    converter_impl = ConverterImplType.PYTHON,
):
  """Pipeline for converting finetuning examples."""

  if len(output_files) != len(input_files):
    raise ValueError(f'Size mismatch: {output_files} {input_files}')

  def _pipeline(root):
    """Pipeline."""

    for (input_file, output_file) in zip(input_files, output_files):
      name = os.path.basename(input_file)


      interactions = read_interactions(root, input_file, name)
      if use_fake_tables:
        interactions = (
            interactions
            | f'InsertFakeTable_{name}' >> beam.Map(insert_fake_table_fn))
      examples = (
          interactions
          | f'CheckTableId_{name}' >> beam.FlatMap(
              pretrain_utils.check_table_id_fn)
          | f'AddNumericValues_{name}' >> beam.Map(
              pretrain_utils.add_numeric_values_fn)
          | f'ToClassifierTensorflowExample_{name}' >> beam.ParDo(
              ToClassifierTensorflowExample(
                  config,
                  name,
                  convert_impl_value=converter_impl.value,
              )))

      pretrain_utils.write_proto_outputs(output_file, f'WriteExamples_{name}',
                                         examples, tf.train.Example)

  return _pipeline


class ToRetrievalTensorflowExample(beam.DoFn):
  """Class for converting retrieval examples."""

  def __init__(
      self,
      config,
      convert_impl_value,
  ):
    self._config = config
    self._convert_impl_value = convert_impl_value

  def start_bundle(self):
    convert_impl = ConverterImplType(self._convert_impl_value)
    if convert_impl == ConverterImplType.PYTHON:
      self._converter = retrieval_utils.ToRetrievalTensorflowExample(
          self._config)
    else:
      raise ValueError(f'Unsupported implementation: {convert_impl.name}')

  def process(
      self,
      element):
    key, interaction = element

    if not interaction.questions:
      new_interaction = interaction_pb2.Interaction()
      new_interaction.CopyFrom(interaction)
      new_question = new_interaction.questions.add()
      new_question.original_text = ''
      new_question.text = ''
      new_question.id = 'FAKE'
      interaction = new_interaction
      beam.metrics.Metrics.counter(
          _NS, 'Fake Questions added for table only example').inc()

    for index in range(len(interaction.questions)):
      beam.metrics.Metrics.counter(_NS, 'Input question').inc()
      question = interaction.questions[index]
      nr_ext = _NegativeRetrievalExamples.negative_retrieval_examples_ext
      if nr_ext not in question.Extensions:
        negative_examples = [None]
      else:
        negative_examples = question.Extensions[nr_ext].examples
        beam.metrics.Metrics.counter(
            _NS,
            f'Input question with {len(negative_examples):03d} negative examples',
        ).inc()

      for negative_example in negative_examples:
        if negative_example is not None:
          beam.metrics.Metrics.counter(_NS, 'Negative Example').inc()
        try:
          example = self._converter.convert(
              interaction,
              index,
              negative_example,
          )
          beam.metrics.Metrics.counter(_NS, 'Conversion success').inc()
        except ValueError as e:
          beam.metrics.Metrics.counter(_NS, 'Conversion error').inc()
          beam.metrics.Metrics.counter(_NS, str(e)).inc()
        # Make sure examples with the same table are not grouped together.
        title = interaction.table.document_title
        new_key = f'{key}#{question.original_text}#{title}'
        if negative_example is not None:
          new_key = f'{new_key}_{negative_example.table.table_id}'
        yield _prepand_fingerprint(new_key), example


class InputFormat(enum.Enum):
  INTERACTION = 1
  TABLE = 2


def _read_inputs_with_format(
    root,
    input_file,
    name,
    input_format,
):
  if input_format == InputFormat.INTERACTION:
    return read_interactions(root, input_file, name)
  elif input_format == InputFormat.TABLE:
    return read_tables(root, input_file, name)


def build_retrieval_pipeline(
    input_files,
    input_format,
    output_files,
    config,
    converter_impl = ConverterImplType.PYTHON,
):
  """Pipeline for converting finetuning examples."""

  if len(output_files) != len(input_files):
    raise ValueError(f'Size mismatch: {output_files} {input_files}')

  def _pipeline(root):
    """Pipeline."""

    for (input_file, output_file) in zip(input_files, output_files):
      name = os.path.basename(input_file)


      examples = (
          _read_inputs_with_format(root, input_file, name, input_format)
          | f'CheckTableId_{name}' >> beam.FlatMap(
              pretrain_utils.check_table_id_fn)
          | f'AddNumericValues_{name}' >> beam.Map(
              pretrain_utils.add_numeric_values_fn)
          | f'ToRetrievalTensorflowExample_{name}' >> beam.ParDo(
              ToRetrievalTensorflowExample(
                  config,
                  convert_impl_value=converter_impl.value,
              )))

      pretrain_utils.write_proto_outputs(output_file, f'WriteExamples_{name}',
                                         examples, tf.train.Example)

  return _pipeline
