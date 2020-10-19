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
"""Utilities for data conversions for different tasks / datasets."""

import collections
import os
from typing import Mapping, Text, Iterable, Optional

from tapas.protos import interaction_pb2
from tapas.utils import interaction_utils_parser
from tapas.utils import pruning_utils
from tapas.utils import sqa_utils
from tapas.utils import tabfact_utils
from tapas.utils import tasks
from tapas.utils import wikisql_utils
from tapas.utils import wtq_utils

import tensorflow.compat.v1 as tf

_Mode = interaction_utils_parser.SupervisionMode


def get_interaction_dir(output_dir):
  return os.path.join(output_dir, 'interactions')


def _to_tfrecord(
    interactions,
    output_dir,
    token_selector,
):
  """Dump interactions to TFRecord files."""
  interactions_dir = get_interaction_dir(output_dir)
  tf.io.gfile.makedirs(interactions_dir)
  for name, examples in interactions.items():
    tfrecord_file = os.path.join(interactions_dir, f'{name}.tfrecord')
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
      for interaction in examples:
        if token_selector is not None:
          interaction = token_selector.annotated_interaction(interaction)
        writer.write(interaction.SerializeToString())


def get_train_filename(task):
  if task in [
      tasks.Task.WIKISQL, tasks.Task.WIKISQL_SUPERVISED, tasks.Task.TABFACT
  ]:
    return 'train'
  if task in [tasks.Task.SQA, tasks.Task.WTQ]:
    return 'random-split-1-train'
  raise ValueError(f'Unknown task: {task.name}')


def get_dev_filename(task):
  if task in [
      tasks.Task.WIKISQL, tasks.Task.WIKISQL_SUPERVISED, tasks.Task.TABFACT
  ]:
    return 'dev'
  if task in [tasks.Task.SQA, tasks.Task.WTQ]:
    return 'random-split-1-dev'
  raise ValueError(f'Unknown task: {task.name}')


def get_test_filename(task):
  if task in [
      tasks.Task.WIKISQL, tasks.Task.WIKISQL_SUPERVISED, tasks.Task.SQA,
      tasks.Task.WTQ, tasks.Task.TABFACT
  ]:
    return 'test'
  raise ValueError(f'Unknown task: {task.name}')


def get_supervision_modes(task):
  """Gets the correct supervision mode for each task."""
  if task == tasks.Task.WIKISQL:
    # We tried using REMOVE_ALL_STRICT but didn't find it to improve results.
    return {
        'train.tsv': _Mode.REMOVE_ALL,
        'dev.tsv': _Mode.NONE,
        'test.tsv': _Mode.NONE
    }
  if task == tasks.Task.WTQ:
    # Remove ambiguous examples at training time.
    # (Examples where the answer occurs in multiple cells.)
    modes = {}
    modes['train.tsv'] = _Mode.REMOVE_ALL_STRICT
    modes['test.tsv'] = _Mode.REMOVE_ALL
    for i in range(1, 5 + 1):
      modes[f'random-split-{i}-train.tsv'] = _Mode.REMOVE_ALL_STRICT
      modes[f'random-split-{i}-dev.tsv'] = _Mode.REMOVE_ALL
    return modes
  if task in [
      tasks.Task.SQA,
      tasks.Task.WIKISQL_SUPERVISED,
      tasks.Task.TABFACT,
  ]:
    return collections.defaultdict(lambda: _Mode.NONE)
  raise ValueError(f'Unknown task: {task.name}')


def create_interactions(
    task,
    input_dir,
    output_dir,
    token_selector,
):
  """Converts original task data to interactions.

  Interactions will be written to f'{output_dir}/interactions'. Other files
  might be written as well.

  Args:
    task: The current task.
    input_dir: Data with original task data.
    output_dir: Outputs are written to this directory.
    token_selector: Optional helper class to keep more relevant tokens in input.
  """

  def to_tfrecord(
      interactions):
    """Helper function that binds output dir and token_selector arguments."""
    _to_tfrecord(interactions, output_dir, token_selector)

  if task == tasks.Task.SQA:
    tsv_dir = input_dir
  elif task == tasks.Task.WTQ:
    wtq_utils.convert(input_dir, output_dir)
    tsv_dir = output_dir
  elif task == tasks.Task.WIKISQL:
    wikisql_utils.convert(input_dir, output_dir)
    tsv_dir = output_dir
  elif task == tasks.Task.WIKISQL_SUPERVISED:
    wikisql_utils.convert(input_dir, output_dir)
    tsv_dir = output_dir
  elif task == tasks.Task.TABFACT:
    to_tfrecord(tabfact_utils.convert(input_dir))
    return
  else:
    raise ValueError(f'Unknown task: {task.name}')
  sqa_utils.create_interactions(
      get_supervision_modes(task),
      tsv_dir,
      get_interaction_dir(output_dir),
      token_selector,
  )
