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
"""Converts WikiSQL dataset to SQA format."""

import collections
import csv
import enum
import functools
import json
import os
import re
from typing import Iterable, Tuple, Text, Any, Mapping, MutableMapping, List, Optional

from absl import logging
import dataclasses
from tapas.utils import file_utils
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf

_TABLE_DIR_NAME = 'table_csv'  # Name that the table folder has in SQA.

_TYPE_CONVERTER = {
    'text': lambda x: x,
    'real': text_utils.convert_to_float,
}

_TOKENIZER = re.compile(r'\w+|[^\w\s]+', re.UNICODE | re.MULTILINE | re.DOTALL)

_DATASETS = ['train', 'test', 'dev']

_NAN = float('nan')


class _Aggregation(enum.Enum):
  """Aggregations as defined by WikiSQL. Indexes match the data."""
  NONE = 0
  MAX = 1
  MIN = 2
  COUNT = 3
  SUM = 4
  AVERAGE = 5


class _Operator(enum.Enum):
  """The boolean operators used by WikiSQL. Indexes match the data."""
  EQUALS = 0
  GREATER = 1
  LESSER = 2


@dataclasses.dataclass
class _Condition:
  """Represents an SQL where clauses (e.g A = "a" or B > 5)."""
  column: Text
  operator: _Operator
  cmp_value: Any


def _iterate_jsonl_file(
    input_dir,
    suffix):
  """Reads all WikiSQL jsonl files."""
  for split in _DATASETS:
    filename = os.path.join(input_dir, split) + suffix
    with tf.io.gfile.GFile(filename, 'r') as fileobj:
      for index, line in enumerate(fileobj):
        try:
          yield split, index, json.loads(line)
        except Exception as e:
          raise e


def _convert_tables(input_dir, output_dir):
  table_dir = os.path.join(output_dir, _TABLE_DIR_NAME)
  for _, _, table in _iterate_jsonl_file(
      input_dir=input_dir, suffix='.tables.jsonl'):
    with tf.io.gfile.GFile(os.path.join(table_dir, table['id'] + '.csv'),
                           'wb') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(table['header'])
      for row in table['rows']:
        writer.writerow(row)


def _parse_value(table, column,
                 cell_value):
  """Convert numeric values to floats and keeps everything else as string."""
  types = table['types']
  return _TYPE_CONVERTER[types[column]](cell_value)


def _parse_table(table):
  """Runs the type converter over the table cells."""
  types = table['types']
  table['real_rows'] = table['rows']
  typed_rows = []
  for row in table['rows']:
    typed_row = []
    for column, cell_value in enumerate(row):
      typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
    typed_rows.append(typed_row)
  table['rows'] = typed_rows


def _load_tables(input_dir):
  tables = {}
  for _, _, table in _iterate_jsonl_file(
      input_dir=input_dir, suffix='.tables.jsonl'):
    _parse_table(table)
    tables[table['id']] = table
  return tables


def _compare(operator, src, tgt):
  if operator == _Operator.EQUALS:
    return src == tgt
  elif operator == _Operator.GREATER:
    return src > tgt
  elif operator == _Operator.LESSER:
    return src < tgt
  raise ValueError(f'Unknown operator: {operator}')


def _is_string(x):
  return isinstance(x, str)


def _normalize_for_match(x):
  return [t for t in _TOKENIZER.findall(x.lower())]


def _respect_conditions(table, row,
                        conditions):
  """True if 'row' satisfies all 'conditions'."""
  for cond in conditions:
    table_value = row[cond.column]

    cmp_value = _parse_value(table, cond.column, cond.cmp_value)

    if _is_string(table_value) and _is_string(cmp_value):
      table_value = _normalize_for_match(table_value)
      cmp_value = _normalize_for_match(cmp_value)

    if not isinstance(table_value, type(cmp_value)):
      raise ValueError('Type difference {} != {}'.format(
          type(table_value), type(cmp_value)))

    if not _compare(cond.operator, table_value, cmp_value):
      return False
  return True


def _get_answer_coordinates(
    table,
    example):
  """Retrieves references coordinates by executing SQL."""
  # MAX and MIN are automatically supported by the model.
  aggregation_op_index = example['sql']['agg']
  if aggregation_op_index >= 3:
    aggregation_op = _Aggregation(aggregation_op_index)
  else:
    aggregation_op = _Aggregation.NONE

  target_column = example['sql']['sel']
  conditions = [
      _Condition(column, _Operator(operator), cmp_value)
      for column, operator, cmp_value in example['sql']['conds']
  ]

  indices = []
  for row in range(len(table['rows'])):
    if _respect_conditions(table, table['rows'][row], conditions):
      indices.append((row, target_column))

  if not indices:
    return [], aggregation_op

  if len(indices) == 1:
    return indices, aggregation_op

  # Parsing of MIN/MAX.
  if aggregation_op_index in (1, 2):
    operators = {2: min, 1: max}
    values = [
        (table['rows'][i][j], index) for index, (i, j) in enumerate(indices)
    ]
    reduced = functools.reduce(operators[example['sql']['agg']], values)

    ret = [indices[reduced[1]]]
    return ret, _Aggregation.NONE

  return indices, aggregation_op


def _get_float_answer(table,
                      answer_coordinates,
                      aggregation_op):
  """Applies operation to produce reference float answer."""
  if not answer_coordinates:
    if aggregation_op == _Aggregation.COUNT:
      return 0.0
    else:
      return _NAN

  # Count can support non numeric answers.
  if aggregation_op == _Aggregation.COUNT:
    return float(len(answer_coordinates))

  # If we have just one answer, if float returns it or try a conversion.
  values = [table['rows'][i][j] for (i, j) in answer_coordinates]
  if len(answer_coordinates) == 1:
    try:
      return text_utils.convert_to_float(values[0])
    except ValueError as e:
      if aggregation_op != _Aggregation.NONE:
        raise e

  if aggregation_op == _Aggregation.NONE:
    return None

  # Other aggregation only support numeric values. Bail out if we have strings.
  if not all((isinstance(v, (int, float)) for v in values)):
    return None

  if aggregation_op == _Aggregation.SUM:
    return float(sum(values))
  elif aggregation_op == _Aggregation.AVERAGE:
    return sum(values) / len(answer_coordinates)
  else:
    raise ValueError(f'Unknown aggregation: {aggregation_op}')


def _get_aggregation_name(aggregation):
  if aggregation == _Aggregation.NONE:
    return ''
  return aggregation.name


def _get_answer_text(table,
                     answer_coordinates,
                     float_answer):
  if float_answer is not None:
    return [str(float_answer)]
  return [str(table['real_rows'][r][c]) for r, c in answer_coordinates]


def _convert_questions(input_dir, output_dir):
  """Executes question conversion."""
  header = ('id', 'annotator', 'position', 'question', 'table_file',
            'answer_coordinates', 'answer_text', 'aggregation', 'float_answer')

  tables = _load_tables(input_dir)

  with tf.io.gfile.GFile(os.path.join(output_dir, 'train.tsv'),
                         'wb') as train_file:
    with tf.io.gfile.GFile(os.path.join(output_dir, 'test.tsv'),
                           'wb') as test_file:
      with tf.io.gfile.GFile(os.path.join(output_dir, 'dev.tsv'),
                             'wb') as dev_file:
        writers = {
            'train':
                csv.DictWriter(train_file, delimiter='\t', fieldnames=header),
            'test':
                csv.DictWriter(test_file, delimiter='\t', fieldnames=header),
            'dev':
                csv.DictWriter(dev_file, delimiter='\t', fieldnames=header),
        }

        counters = collections.defaultdict(int)

        for _, writer in writers.items():
          writer.writeheader()

        for tag, index, example in _iterate_jsonl_file(
            input_dir, suffix='.jsonl'):
          counters[f'{tag}-input'] += 1
          table = tables[example['table_id']]
          try:
            answer_coordinates, aggregation_op = \
                _get_answer_coordinates(table, example)
            float_answer = _get_float_answer(table, answer_coordinates,
                                             aggregation_op)
          except ValueError as exc:
            # Treat this as warning but emit empty result.
            answer_coordinates = []
            aggregation_op = _Aggregation.NONE
            float_answer = None
            counters[f'{tag}-warnings'] += 1
            counters[f'{tag}-warnings[{str(exc)}]'] += 1

          writers[tag].writerow({
              'id':
                  f'{tag}-{index}',
              'annotator':
                  '0',
              'position':
                  '0',
              'question':
                  example['question'],
              'table_file':
                  f"table_csv/{example['table_id']}.csv",
              'answer_coordinates': [str(ac) for ac in answer_coordinates],
              'answer_text':
                  _get_answer_text(table, answer_coordinates, float_answer),
              'aggregation':
                  _get_aggregation_name(aggregation_op),
              'float_answer':
                  '' if float_answer is None else str(float_answer),
          })

          if not answer_coordinates:
            counters[f'{tag}-empty'] += 1
          else:
            counters[f'{tag}-correct'] += 1
          counters[f'{tag}-output'] += 1

  for tag in _DATASETS:
    relative_count = 100 * counters[f'{tag}-output'] / counters[f'{tag}-input']
    logging.info(f'Split: {tag} converted {relative_count}')


def _create_dirs(output_dir):
  file_utils.make_directories(os.path.join(output_dir, _TABLE_DIR_NAME))


def convert(input_dir, output_dir):
  """Converts WikiSQL data to SQA format.

  Args:
    input_dir: WikiSQL original data.
    output_dir: Where output files are written to see below.

  This will create the following file structure in 'output_dir':
    - {dev,test,train}.tsv
    - table_csv/?-???????-?.csv (26530 files)
  """
  _create_dirs(output_dir)
  _convert_tables(input_dir, output_dir)
  _convert_questions(input_dir, output_dir)
