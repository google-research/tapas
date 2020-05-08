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
"""Heuristics for finding interpretations that produce float answers."""

import collections
import itertools
import math
import random

from typing import List, Any
from tapas.protos import interaction_pb2


_MAX_INDICES_TO_EXPLORE = 10
_MAX_NUM_TRIALS = 100
_FLOAT_TOLERANCE = 1.0e-2
_MAX_NUM_CANDIDATES = 500

_Answer = interaction_pb2.Answer

Candidate = collections.namedtuple('Candidate',
                                   ['agg_function', 'column', 'rows'])


def _is_integer(float_value):
  """True if 'float_value' is an integer (such as 5.0)."""
  return math.ceil(float_value) == float_value


def _clip_candidates(rng,
                     candidates):
  """Trims canddiates to the size specified by '_MAX_NUM_CANDIDATES'."""
  if len(candidates) > _MAX_NUM_CANDIDATES:
    rng.shuffle(candidates)
    candidates = candidates[:_MAX_NUM_CANDIDATES]
  candidates.sort()
  return candidates


def _find_count_candidates(rng, table,
                           float_value):
  """Finds count candidates.

  If 'float_value' is integer select random sets of rows for all columns.

  Args:
   rng: random number generator.
   table: the question table (only dimension matter).
   float_value: the expected answer.

  Returns:
    random list of count candidates that produce the right answer.
  """
  candidates = []
  if not _is_integer(float_value):
    # Answer is a true float.
    return []
  int_value = int(float_value)

  num_rows = len(table.rows)
  if num_rows < int_value:
    # Answer cannot be produced by counting columns.
    return []

  num_columns = len(table.rows[0].cells)

  solutions = set()

  rows = list(range(num_rows))
  for _ in range(_MAX_NUM_TRIALS):
    rng.shuffle(rows)
    solutions.add(tuple(sorted(rows[:int_value])))

  for rows in solutions:
    for column in range(num_columns):
      candidates.append(Candidate(_Answer.COUNT, column, rows))

  return _clip_candidates(rng, candidates)


def get_correct_powerset(indices, values, is_correct_fn):
  """Produces powerset of indices and checks each set if 'is_correct_fn'."""
  selections = set()
  for subset in itertools.chain.from_iterable(
      itertools.combinations(indices, r) for r in range(1,
                                                        len(indices) + 1)):
    if is_correct_fn(subset, values):
      selections.add(tuple(subset))
  return selections


def _find_numeric_cell_combinations(rng, table, is_correct_fn,
                                    agg_function):
  """Finds candidates that require a numeric operation (sum or average).

  Args:
   rng: random number generator.
   table: the question table (only dimension matter).
   is_correct_fn: Returns true if indices produce the right values.
   agg_function: The numeric operation

  Returns:
    random list of candidates that produce the right answer.
  """
  candidates = []
  if not table.rows:
    return []
  num_columns = len(table.rows[0].cells)
  for column in range(num_columns):
    values = [None] * len(table.rows)
    rows = []
    for row, cells in enumerate(table.rows):
      cell = cells.cells[column]
      if (cell.numeric_value.HasField('float_value') and
          cell.numeric_value.float_value != float('INF') and
          not math.isnan(cell.numeric_value.float_value)):
        values[row] = cell.numeric_value.float_value
        rows.append(row)

    if not rows:
      continue

    if len(rows) > _MAX_INDICES_TO_EXPLORE:
      selections = set()
      for _ in range(_MAX_NUM_TRIALS):
        rng.shuffle(rows)
        row_subset = sorted(rows[:_MAX_INDICES_TO_EXPLORE])
        selections.update(
            get_correct_powerset(row_subset, values, is_correct_fn))
    else:
      selections = get_correct_powerset(rows, values, is_correct_fn)

    for rows in selections:
      candidates.append(Candidate(agg_function, column, rows))
  return _clip_candidates(rng, candidates)


def _find_sum_candidates(rng, table,
                         float_value):

  def _is_correct(indices, values):
    return (math.fabs(sum(values[i] for i in indices) - float_value) <
            _FLOAT_TOLERANCE)

  return _find_numeric_cell_combinations(rng, table, _is_correct, _Answer.SUM)


def _find_average_candidates(rng, table,
                             float_value):

  def _is_correct(indices, values):
    length = float(len(indices))
    return (math.fabs(sum(values[i] for i in indices) / length - float_value) <
            _FLOAT_TOLERANCE)

  return _find_numeric_cell_combinations(rng, table, _is_correct,
                                         _Answer.AVERAGE)


def find_candidates(rng, table,
                    question):
  """Heuristically finds candidates that yield the correct float answer.

  This function attempts to find interpretations that produce the respective
  float answer.

  Args:
    rng: random number generator.
    table: Table with numeric value annotations.
    question: Question, model will attempt to produce float answer.

  Returns:
    Plausible candidates that produce the float answer.
  """
  if (not question.answer.HasField('float_value') or
      math.isnan(question.answer.float_value) or
      question.answer.float_value == float('INF')):
    return []

  candidates = []
  if question.answer.answer_coordinates:
    # The interpretation that selects the answer from the table.
    if len(question.answer.answer_coordinates) != 1:
      raise ValueError('Expected a single answer coordinate.')
    candidates.append(
        Candidate(_Answer.NONE,
                  question.answer.answer_coordinates[0].column_index,
                  (question.answer.answer_coordinates[0].row_index,)))

  candidates += _find_count_candidates(rng, table, question.answer.float_value)
  candidates += _find_sum_candidates(rng, table, question.answer.float_value)
  candidates += _find_average_candidates(rng, table,
                                         question.answer.float_value)

  return candidates
