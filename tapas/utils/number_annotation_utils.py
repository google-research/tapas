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
"""Helper functions for handling numeric Quik annotations and relations."""

import collections
import itertools
from typing import Text, Union, Tuple, Iterable, List, Any, Dict, Callable, Optional, Set
from absl import logging
from tapas.protos import interaction_pb2

from tapas.utils import constants
from tapas.utils import number_utils
from tapas.utils import text_utils


MAX_QUESTION_NUMERIC_VALUES = 8
_PrimitiveNumericValue = Union[
    float, Tuple[Optional[float], Optional[float], Optional[float]]]
_SortKeyFn = Callable[[interaction_pb2.NumericValue], Tuple[float, Ellipsis]]

_DATE_TUPLE_SIZE = 3


def _get_value_type(numeric_value):
  if numeric_value.HasField('float_value'):
    return constants.NUMBER_TYPE
  elif numeric_value.HasField('date'):
    return constants.DATE_TYPE
  raise ValueError('Unknown type: %s' % numeric_value)


def _get_value_as_primitive_value(
    numeric_value):
  """Maps a NumericValue proto to a float or tuple of float."""
  if numeric_value.HasField('float_value'):
    return numeric_value.float_value
  if numeric_value.HasField('date'):
    date = numeric_value.date
    value_tuple = [None, None, None]
    # All dates fields are cased to float to produce a simple primitive value.
    if date.HasField('year'):
      value_tuple[0] = float(date.year)
    if date.HasField('month'):
      value_tuple[1] = float(date.month)
    if date.HasField('day'):
      value_tuple[2] = float(date.day)
    return tuple(value_tuple)
  raise ValueError('Unknown type: %s' % numeric_value)


def _get_all_types(
    numeric_values):
  return {_get_value_type(value) for value in numeric_values}


def get_numeric_sort_key_fn(
    numeric_values):
  """Creates a function that can be used as a sort key or to compare the values.

  Maps to primitive types and finds the biggest common subset.

  Consider the values "05/05/2010" and "August 2007".
  With the corresponding primitive values (2010.,5.,5.) and (2007.,8., None).

  These values can be compared by year and date so we map to the sequence
  (2010., 5.), (2007., 8.).

  If we added a third value "2006" with primitive value (2006., None, None),
  we could only compare by the year so we would map to (2010.,), (2007.,)
  and (2006.,).

  Args:
   numeric_values: Values to compare.

  Returns:
   A function that can be used as a sort key function (mapping numeric values
   to a comparable tuple).

  Raises:
    ValueError if values don't have a common type or are not comparable.
  """
  value_types = _get_all_types(numeric_values)
  if len(value_types) != 1:
    raise ValueError('No common value type in %s' % numeric_values)

  value_type = next(iter(value_types))
  if value_type == constants.NUMBER_TYPE:
    # Primitive values are simple floats, nothing to do here.
    return _get_value_as_primitive_value

  # The type can only be Date at this point which means the primitive type
  # is a float triple.
  valid_indexes = set(range(_DATE_TUPLE_SIZE))

  for numeric_value in numeric_values:
    value = _get_value_as_primitive_value(numeric_value)
    assert isinstance(value, tuple)
    for tuple_index, inner_value in enumerate(value):
      if inner_value is None:
        valid_indexes.discard(tuple_index)

  if not valid_indexes:
    raise ValueError('No common value in %s' % numeric_values)

  def _sort_key_fn(numeric_value):
    value = _get_value_as_primitive_value(numeric_value)
    return tuple(value[index] for index in valid_indexes)

  return _sort_key_fn


def _consolidate_numeric_values(
    row_index_to_values,
    min_consolidation_fraction,
    debug_info):
  """Finds the most common numeric values in a column and returns them.

  Args:
   row_index_to_values: For each row index all the values in that cell.
   min_consolidation_fraction: Fraction of cells that need to have consolidated
     value.
   debug_info: Additional information only used for logging.

  Returns:
   For each row index the first value that matches the most common value.
   Rows that don't have a matching value are dropped. Empty list if values can't
   be consolidated.
  """
  type_counts = collections.Counter()
  for numeric_values in row_index_to_values.values():
    type_counts.update(_get_all_types(numeric_values))
  if not type_counts:
    return {}
  max_count = max(type_counts.values())
  if max_count < len(row_index_to_values) * min_consolidation_fraction:
    logging.log_every_n(logging.INFO, 'Can\'t consolidate types: %s %s %d', 100,
                        debug_info, row_index_to_values, max_count)
    return {}

  valid_types = set()
  for value_type, count in type_counts.items():
    if count == max_count:
      valid_types.add(value_type)
  if len(valid_types) > 1:
    assert constants.DATE_TYPE in valid_types
    max_type = constants.DATE_TYPE
  else:
    max_type = next(iter(valid_types))

  new_row_index_to_value = {}
  for index, values in row_index_to_values.items():
    # Extract the first matching value.
    for value in values:
      if _get_value_type(value) == max_type:
        new_row_index_to_value[index] = value
        break

  return new_row_index_to_value


def _get_numeric_values(text):
  """Parses text and returns numeric values."""
  numeric_spans = number_utils.parse_text(text)
  return itertools.chain(*(span.values for span in numeric_spans))


def _get_column_values(
    table,
    col_index):
  """Parses text in column and returns a dict mapping row_index to values."""
  index_to_values = {}
  for row_index, row in enumerate(table.rows):
    text = text_utils.normalize_for_match(row.cells[col_index].text)
    index_to_values[row_index] = list(_get_numeric_values(text))
  return index_to_values


def get_numeric_relation(
    value,
    other_value,
    sort_key_fn):
  """Compares two values and returns their relation or None."""
  value = sort_key_fn(value)
  other_value = sort_key_fn(other_value)
  if value == other_value:
    return constants.Relation.EQ
  if value < other_value:
    return constants.Relation.LT
  if value > other_value:
    return constants.Relation.GT
  return None


def add_numeric_table_values(table,
                             min_consolidation_fraction=0.7,
                             debug_info = None):
  """Parses text in table column-wise and adds the consolidated values.

  Consolidation refers to finding values with a common types (date or number).

  Args:
   table: Table to annotate.
   min_consolidation_fraction: Fraction of cells in a column that need to have
     consolidated value.
   debug_info: Additional information used for logging.
  """
  for row in table.rows:
    for cell in row.cells:
      cell.ClearField('numeric_value')

  for col_index, column in enumerate(table.columns):
    column_values = _consolidate_numeric_values(
        _get_column_values(table, col_index),
        min_consolidation_fraction=min_consolidation_fraction,
        debug_info=(debug_info, column))

    for row_index, numeric_value in column_values.items():
      table.rows[row_index].cells[col_index].numeric_value.CopyFrom(
          numeric_value)


def add_numeric_values_to_questions(interaction
                                   ):
  """Adds numeric value spans to all questions."""
  for question in interaction.questions:
    question.text = text_utils.normalize_for_match(question.original_text)
    question.annotations.CopyFrom(
        interaction_pb2.NumericValueSpans(
            spans=number_utils.parse_text(question.text)))


def add_numeric_values(interaction):
  add_numeric_table_values(interaction.table)
  add_numeric_values_to_questions(interaction)
