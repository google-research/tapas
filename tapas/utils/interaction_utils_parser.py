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
"""This module implements a simple parser for Interaction.

It will parse answer_texts to populate other fields (e.g. answer_coordinates,
float_value) that are used to derive TF examples.

Please note that exceptions in this module are concise and not parameterized,
since they are used as counter names in a BEAM pipeline.
"""

import enum
from typing import Callable, List, Text

import frozendict
import numpy as np
import scipy.optimize
from tapas.protos import interaction_pb2
from tapas.utils import text_utils


class SupervisionMode(enum.Enum):
  # Don't filter out any supervised information.
  NONE = 0
  # Remove all the supervised signals and recompute them by parsing answer
  # texts.
  REMOVE_ALL = 2


def _find_matching_coordinates(table, answer_text,
                               normalize):
  normalized_text = normalize(answer_text)
  for row_index, row in enumerate(table.rows):
    for column_index, cell in enumerate(row.cells):
      if normalized_text == normalize(cell.text):
        yield (row_index, column_index)


def _compute_cost_matrix_inner(table,
                               answer,
                               normalize):
  """Returns a cost matrix M where the value M[i,j] contains a matching cost from answer i to cell j.

  The matrix is a binary matrix and -1 is used to indicate a possible match from
  a given answer_texts to a specific cell table. The cost matrix can then be
  usedto compute the optimal assignments that minimizes the cost using the
  hungarian algorithm (see scipy.optimize.linear_sum_assignment).

  Args:
    table: a Table message.
    answer: an Answer message.
    normalize: a function that normalizes a string.

  Raises:
    ValueError if:
      - we cannot correctly construct the cost matrix or the text-cell
      assignment is ambiguous.
      - we cannot find a matching cell for a given answer_text.

  Returns:
    A numpy matrix with shape (num_answer_texts, num_rows * num_columns).
  """
  max_candidates = 0
  num_cells = len(table.rows) * len(table.columns)
  num_candidates = np.zeros((len(table.rows), len(table.columns)))
  cost_matrix = np.zeros((len(answer.answer_texts), num_cells))

  for index, answer_text in enumerate(answer.answer_texts):
    found = False
    for row, column in _find_matching_coordinates(table, answer_text,
                                                  normalize):
      found = True
      cost_matrix[index, (row * len(table.columns)) + column] = -1
      num_candidates[row, column] += 1
      max_candidates = max(max_candidates, num_candidates[row, column])
    if not found:
      raise ValueError("Can't find text in table")

  # TODO(piccinno): Shall we allow ambiguous assignments?
  if max_candidates > 1:
    raise ValueError("Assignment is ambiguous")

  return cost_matrix


def _compute_cost_matrix(table,
                         answer):
  for index, normalize_fn in enumerate(text_utils.STRING_NORMALIZATIONS):
    try:
      return _compute_cost_matrix_inner(table, answer, normalize_fn)
    except ValueError:
      if index == len(text_utils.STRING_NORMALIZATIONS) - 1:
        raise


def _parse_answer_coordinates(table,
                              answer):
  """Populates answer_coordinates using answer_texts.

  Args:
    table: a Table message, needed to compute the answer coordinates.
    answer: an Answer message that will be modified on success.

  Raises:
    ValueError if the conversion fails.
  """
  cost_matrix = _compute_cost_matrix(table, answer)
  row_indices, column_indices = scipy.optimize.linear_sum_assignment(
      cost_matrix)
  for _ in row_indices:
    answer.answer_coordinates.add()
  for row_index in row_indices:
    flatten_position = column_indices[row_index]
    row_coordinate = flatten_position // len(table.columns)
    column_coordinate = flatten_position % len(table.columns)
    answer.answer_coordinates[row_index].row_index = row_coordinate
    answer.answer_coordinates[row_index].column_index = column_coordinate


def _parse_answer_float(answer):
  if len(answer.answer_texts) > 1:
    raise ValueError("Cannot convert to multiple answers to single float")
  float_value = text_utils.convert_to_float(answer.answer_texts[0])
  answer.float_value = float_value


def _has_single_float_answer_equal_to(question,
                                      target):
  """Returns true if the question has a single answer whose value equals to target."""
  if len(question.answer.answer_texts) != 1:
    return False
  try:
    float_value = text_utils.convert_to_float(question.answer.answer_texts[0])
    # In general answer_float is derived by applying the same conver_to_float
    # function at interaction creation time, hence here we use exact match to
    # avoid any false positive.
    return text_utils.to_float32(float_value) == text_utils.to_float32(target)
  except ValueError:
    return False


def _parse_question(table,
                    original_question,
                    clear_fields):
  """Parses question's answer_texts fields to possibly populate additional fields.

  Args:
    table: a Table message, needed to compute the answer coordinates.
    original_question: a Question message containing answer_texts.
    clear_fields: A list of strings indicating which fields need to be cleared
      and possibly repopulated.

  Returns:
    A Question message with answer_coordinates or float_value field populated.

  Raises:
    ValueError if we cannot parse correctly the question message.
  """
  question = interaction_pb2.Question()
  question.CopyFrom(original_question)

  # If we have a float value signal we just copy its string representation to
  # the answer text (if multiple answers texts are present OR the answer text
  # cannot be parsed to float OR the float value is different), after clearing
  # this field.
  if "float_value" in clear_fields and question.answer.HasField("float_value"):
    if not _has_single_float_answer_equal_to(question,
                                             question.answer.float_value):
      del question.answer.answer_texts[:]
      float_value = float(question.answer.float_value)
      if float_value.is_integer():
        number_str = str(int(float_value))
      else:
        number_str = str(float_value)
      question.answer.answer_texts.append(number_str)

  if not question.answer.answer_texts:
    raise ValueError("No answer_texts provided")

  for field_name in clear_fields:
    question.answer.ClearField(field_name)

  error_message = ""

  if not question.answer.answer_coordinates:
    try:
      _parse_answer_coordinates(table, question.answer)
    except ValueError as exc:
      error_message += "[answer_coordinates: {}]".format(str(exc))

  if not question.answer.HasField("float_value"):
    try:
      _parse_answer_float(question.answer)
    except ValueError as exc:
      error_message += "[float_value: {}]".format(str(exc))

  # Raises an exception if we cannot set any of the two fields.
  if not question.answer.answer_coordinates and not question.answer.HasField(
      "float_value"):
    raise ValueError("Cannot parse answer: {}".format(error_message))

  return question


# TODO(piccinno): Use some sort of introspection here to get the field names of
# the proto.
_CLEAR_FIELDS = frozendict.frozendict({
    SupervisionMode.REMOVE_ALL: [
        "answer_coordinates", "float_value", "aggregation_function"
    ]
})


def parse_question(table,
                   question, mode):
  """Parses answer_text field of question to populate additional fields needed to create TF examples.

  Args:
    table: a Table message, needed to compute the answer coordinates.
    question: a Question message, that will be modified (even on unsuccesful
      parsing).
    mode: See SupervisionMode enum for more information.

  Returns:
    A Question message with answer_coordinates or float_value field populated.

  Raises:
    ValueError if we cannot parse correctly the question message.
  """
  if mode == SupervisionMode.NONE:
    return question

  clear_fields = _CLEAR_FIELDS.get(mode, None)
  if clear_fields is None:
    raise ValueError(f"Mode {mode.name} is not supported")

  return _parse_question(table, question, clear_fields)
