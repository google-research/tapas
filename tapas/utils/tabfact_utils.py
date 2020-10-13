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
# coding=utf-8
r"""Utils for converting WTQ to SQA format."""

import collections
import csv
import json
import os
import re
from typing import Any, Iterable, List, Mapping, Set, Text, Tuple

from absl import logging
import six
from tapas.protos import interaction_pb2
import tensorflow.compat.v1 as tf


_TABLE_DIR_NAME = 'table_csv'  # Name that the table folder has in SQA.


FEATURE_ALL = 'ALL'
FEATURE_TRUE = 'TRUE'
FEATURE_FALSE = 'FALSE'
FEATURE_SUPERLATIVE = 'SUPER'
FEATURE_COMPARATIVE = 'COMP'
FEATURE_NEGATIVE = 'NEG'
FEATURE_PERCENTAGE = 'PERC'
FEATURE_AGGREGATION = 'AGG'

_SUPERLATIVE_WORDS = {
    'lowest', 'highest', 'best', 'worst', 'least', 'most', 'oldest', 'newest',
    'fewest', 'last', 'first', 'earliest', 'latest', 'greatest', 'smallest'
}
_COMPARATIVE_WORDS = {
    'than', 'less', 'more', 'better', 'worse', 'higher', 'lower', 'shorter',
    'same'
}
_NEGATIVE_WORDS = {'not', 'any', 'none', 'no', 'never'}
_AGGREGATION_WORDS = {
    'only', 'there', 'total', 'count', 'average', 'sum', 'amount'
}
_PERCENTAGE_RE = re.compile(r'\d[\d.]* ?%')

_DEFAULT_FEATURE_LIST = [
    FEATURE_TRUE, FEATURE_FALSE, FEATURE_SUPERLATIVE, FEATURE_COMPARATIVE,
    FEATURE_NEGATIVE, FEATURE_PERCENTAGE, FEATURE_AGGREGATION
]

_ExperimentMetric = Mapping[Text, Iterable[Tuple[Text, float]]]


def get_features_single(question, label):
  """Add a feature bag of words features as well as some hard-coded ones."""
  tokens = question.split()
  features = set(tokens)
  features.add(FEATURE_ALL)
  features.add(FEATURE_TRUE if label else FEATURE_FALSE)
  if features & _NEGATIVE_WORDS:
    features.add(FEATURE_NEGATIVE)
  if features & _SUPERLATIVE_WORDS:
    features.add(FEATURE_SUPERLATIVE)
  if features & _COMPARATIVE_WORDS:
    features.add(FEATURE_COMPARATIVE)
  if features & _AGGREGATION_WORDS:
    features.add(FEATURE_AGGREGATION)
  if re.search(_PERCENTAGE_RE, question):
    features.add(FEATURE_PERCENTAGE)

  return features, label


def get_features(questions):
  return [
      get_features_single(question, label)
      for question, label in zip(questions[0], questions[1])
  ]


def get_interesting_features(accuracy,
                             heardoom,
                             num_features):
  """Find the most intersting features in terms of headroom and saliency."""

  def _get_feature_score(feature):
    # Compute a heauristic to rank features according to both the headroom but
    # also the error rate in that particular group compared to baseline.
    metrics = zip(accuracy[FEATURE_ALL], accuracy[feature], heardoom[feature])
    return max(exp_room * (overall_acc - exp_acc)
               for (_, overall_acc), (_, exp_acc), (_, exp_room) in metrics)

  non_default_features = (
      feature for feature in accuracy if feature not in _DEFAULT_FEATURE_LIST)
  results = sorted(non_default_features, key=_get_feature_score, reverse=True)

  return (_DEFAULT_FEATURE_LIST + list(results))[:num_features]


def _log_stats(counter, file_name):
  """Logs counters to file."""
  logging.info('-------- STATS for: %s --------', file_name)
  logging.info('Questions: %d', counter['questions'])
  logging.info('----------------')


def _convert_data(
    all_questions,
    input_file,
    tables,
):
  """Converts TabFact data to interactions format."""
  logging.info('Converting data from: %s...', input_file)

  counter = collections.Counter()  # Counter for stats.

  with tf.io.gfile.GFile(input_file) as file_in:
    for table_id in json.load(file_in):
      questions, labels, _ = all_questions[table_id]
      for i, (text, label) in enumerate(zip(questions, labels)):
        # The extra zeros are there to match SQA id format.
        question_id = f'{table_id}_{i}-0'
        question = interaction_pb2.Question(
            id=f'{question_id}_0',
            original_text=text,
            answer=interaction_pb2.Answer(class_index=label))
        table = interaction_pb2.Table()
        table.CopyFrom(tables[table_id])
        yield interaction_pb2.Interaction(
            id=question_id, questions=[question], table=table)

        counter['questions'] += 1
        if counter['questions'] % 1000 == 0:
          logging.info('Processed %d questions...', counter['questions'])

    _log_stats(counter, input_file)


def read_all_tables(input_dir):
  """Read tables from the original format."""
  csv_path = os.path.join(input_dir, 'data', 'all_csv')
  results = {}
  for table_id in tf.io.gfile.listdir(csv_path):
    with tf.io.gfile.GFile(os.path.join(csv_path, table_id)) as f:
      results[table_id] = f.read()
  return results


def _convert_table(table_id, table_text):
  """Parses a table from # separated values format into proto format."""
  rows = []
  with six.StringIO(table_text) as csv_in:
    for index, row in enumerate(csv.reader(csv_in, delimiter='#')):
      cells = [interaction_pb2.Cell(text=text) for text in row]
      if index == 0:
        columns = cells
      else:
        rows.append(interaction_pb2.Cells(cells=cells))
  return interaction_pb2.Table(
      table_id=f'{_TABLE_DIR_NAME}/{table_id}', columns=columns, rows=rows)


def _convert_tables(input_dir):
  """Read tables the original format file and parse into proto format."""
  return {
      table_id: _convert_table(table_id, table_text)
      for table_id, table_text in read_all_tables(input_dir).items()
  }


def read_questions(input_dir):
  collected_data = {}
  for i in [1, 2]:
    input_file = os.path.join(input_dir,
                              f'collected_data/r{i}_training_all.json')
    with tf.gfile.Open(input_file) as f:
      collected_data.update(json.load(f))
  return collected_data


def convert(
    input_dir,):
  """Converts from TabFact to SQA format.

  Args:
    input_dir: The original TabFact data.

  Returns:
    Iterables of interactions for each split in train, dev and test.
  """
  tables = _convert_tables(input_dir)
  questions = read_questions(input_dir)
  splits = {
      'train': 'train_id.json',
      'test': 'test_id.json',
      'dev': 'val_id.json',
  }
  return {
      split: _convert_data(questions, os.path.join(input_dir, 'data', file),
                           tables) for split, file in splits.items()
  }
