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
"""Functions for creating synthetic tables and statements."""

import abc
import enum
import functools
import itertools
from typing import List, Text, Optional, Tuple, Set, Iterable, Union, Mapping, Callable

import dataclasses
import numpy as np
from tapas.protos import interaction_pb2
from tapas.utils import text_utils


Number = Union[int, float]


@dataclasses.dataclass(frozen=True)
class SynthesizationConfig:
  """Specifies the sampling parameters.

    prob_statement_replacement: Chance to replace complex expr. with constant.
    prob_stop_value_row_expansion: Chance to stop adding new where clause.
    prob_count_aggregation: Chance to create COUNT aggregation.
    attempts: Number of retries for places that can fail
  """
  prob_statement_replacement: float = 0.5
  prob_stop_value_row_expansion: float = 0.5
  prob_count_aggregation: float = 0.2
  attempts: int = 100


class SynthesizationError(Exception):
  """Occurs when sampling an erroneous statement.

     Sampling should be retried if this happens.
  """
  pass


@dataclasses.dataclass(frozen=True)
class CellValue:
  value: Text
  rank: Optional[int]


@dataclasses.dataclass(frozen=True)
class Column:
  name: Text
  values: List[CellValue]


@dataclasses.dataclass(frozen=True)
class TableRow:
  cells: List[CellValue]


# TODO Consider turing Table into a DataFrame.
@dataclasses.dataclass(frozen=True)
class Table:
  columns: List[Text]
  rows: List[TableRow]


def _create_rank_dict(numbers):
  return {f: rank for rank, f in enumerate(sorted(set(numbers)))}


def _to_table_proto(table):
  new_table = interaction_pb2.Table()
  for column in table.columns:
    new_table.columns.add().text = column
  for row in table.rows:
    new_row = new_table.rows.add()
    for cell in row.cells:
      new_row.cells.add().text = cell.value
  return new_table


def _from_table_proto(table):
  """Converts a proto table. Computes ranks for float columns."""
  table_columns = []
  table_rows = []

  for _ in table.rows:
    table_rows.append([])

  for i, column in enumerate(table.columns):
    table_columns.append(column.text)

    values = [row.cells[i].text for row in table.rows]

    try:
      floats = _to_floats(values)
      rank_dict = _create_rank_dict(floats)
      ranks = [rank_dict[f] for f in floats]
    except ValueError:
      ranks = [None] * len(values)

    for j, (value, rank) in enumerate(zip(values, ranks)):
      table_rows[j].append(CellValue(value, rank))

  table_rows = [TableRow(row) for row in table_rows]
  return Table(table_columns, table_rows)


class Counter:

  def count(self, message):
    pass


class SelectClause(abc.ABC):
  """An SQL-like select clause that maps a list of rows to a set of values."""

  @abc.abstractmethod
  def evaluate(self, table, rows):
    Ellipsis

  @abc.abstractmethod
  def verbalize(self):
    Ellipsis

  @abc.abstractmethod
  def increment_counters(self, name, counter):
    Ellipsis


class CountSelectClause(SelectClause):

  def evaluate(self, table, rows):
    return {_format_number(len(rows))}

  def verbalize(self):
    return 'the count'

  def increment_counters(self, name, counter):
    counter.count(f'{name}: Select: COUNT')


class Aggregation(enum.Enum):
  """Mostly numeric value aggregations."""
  NONE = 0
  LOWEST = 1
  GREATEST = 2
  FIRST = 3
  LAST = 4
  SUM = 5
  RANGE = 6
  AVERAGE = 7

  def evaluate(self, values):
    if self == Aggregation.NONE:
      return {v.value for v in values}
    else:
      if len(values) < 2:
        # We are strict and require at leat 2 values for aggregation.
        return None
      if self == Aggregation.FIRST:
        return {values[0].value}
      if self == Aggregation.LAST:
        return {values[-1].value}
      else:
        if not _have_rank(values):
          # Rank is none but numeric operation expected.
          return None
        if self == Aggregation.LOWEST:
          return {min(values, key=lambda value: value.rank).value}
        if self == Aggregation.GREATEST:
          return {max(values, key=lambda value: value.rank).value}

        float_values = _to_floats(value.value for value in values)

        if self == Aggregation.SUM:
          return {_format_number(sum(float_values))}
        if self == Aggregation.AVERAGE:
          return {_format_number(np.mean(float_values))}
        if self == Aggregation.RANGE:
          return {_format_number(max(float_values) - min(float_values))}
        else:
          raise ValueError(f'Unknown aggregation: {self}')

  def verbalize(self, column_name):
    if self == Aggregation.NONE:
      return column_name
    elif self == Aggregation.FIRST:
      return f'the first {column_name}'
    elif self == Aggregation.LAST:
      return f'the last {column_name}'
    elif self == Aggregation.LOWEST:
      return f'the lowest {column_name}'
    elif self == Aggregation.GREATEST:
      return f'the greatest {column_name}'
    elif self == Aggregation.SUM:
      return f'the total {column_name}'
    elif self == Aggregation.AVERAGE:
      return f'the average {column_name}'
    elif self == Aggregation.RANGE:
      return f'the range of {column_name}'
    else:
      raise ValueError(f'Unknown aggregation: {self}')


@dataclasses.dataclass(frozen=True)
class ValueAggregationClause(SelectClause):
  """An SQL like select clause à la 'select SUM(COLUMN_NAME)'."""
  aggregation: Aggregation
  column_name: Text

  def evaluate(self, table, rows):
    try:
      value_index = _column_index_by_name(table, self.column_name)
    except ValueError:
      return None
    values = [_value_from_index(row, value_index) for row in rows]
    return self.aggregation.evaluate(values)

  def verbalize(self):
    return self.aggregation.verbalize(self.column_name)

  def increment_counters(self, name, counter):
    counter.count(f'{name}: Select: {self.aggregation.name}')


class EvaluationResult(enum.Enum):
  TRUE = 0
  FALSE = 1
  ERROR = 2


class Comparator(enum.Enum):
  """A comparator that can be used in a condition."""
  EQUALS = 0
  GREATER = 2
  LESSER = 3

  def verbalize(self):
    if self == Comparator.EQUALS:
      return 'is'
    elif self == Comparator.LESSER:
      return 'is less than'
    elif self == Comparator.GREATER:
      return 'is greater than'
    else:
      raise ValueError(f'Unknown comparator: {self}')

  def compare(self, left, right):
    if self == Comparator.EQUALS:
      return _bool_to_result(right == left)
    try:
      left = set(_to_floats(left))
      right = set(_to_floats(right))
    except ValueError:
      return EvaluationResult.ERROR
    if not left or not right:
      return EvaluationResult.ERROR
    if self == Comparator.LESSER:
      return _bool_to_result(max(left) < min(right))
    if self == Comparator.GREATER:
      return _bool_to_result(min(left) > max(right))
    raise ValueError(f'Unknown comparator: {self}')

  def increment_counters(self, name, counter):
    counter.count(f'{name}: Comparator {self.name}')


@dataclasses.dataclass(frozen=True)
class WhereClause:
  """An sql like where clause such as 'COLUMN_NAME < 5'."""
  column_name: Text
  cell_value: Text
  comparator: Comparator = Comparator.EQUALS

  def verbalize(self):
    return f'{self.column_name} {self.comparator.verbalize()} {self.cell_value}'

  def filter(self, table, rows):
    row_select_index = _column_index_by_name(table, self.column_name)
    return [
        row for row in rows if self.matches(
            _value_from_index(row, row_select_index).value,
            self.cell_value,
        )
    ]

  def matches(self, value, other_value):
    result = self.comparator.compare(
        {value},
        {other_value},
    )
    if result == EvaluationResult.TRUE:
      return True
    if result == EvaluationResult.FALSE:
      return False
    raise ValueError(f'Error comparing values {result}')

  def increment_counters(self, name, counter):
    self.comparator.increment_counters(f'{name}: where', counter)


class Expression(abc.ABC):
  """An expression that evaluates to a set of values."""

  @abc.abstractmethod
  def evaluate(self, table):
    Ellipsis

  @abc.abstractmethod
  def verbalize(self):
    Ellipsis

  @abc.abstractmethod
  def increment_counters(self, name, counter):
    Ellipsis


def _format_number(number):
  number = float(number)
  if number.is_integer():
    return str(int(number))
  return f'{number:.2f}'


@dataclasses.dataclass(frozen=True)
class ComplexExpression(Expression):
  """A complex expression evaluated against a table.

    All aggregations except COUNT require a 'value'.
  """
  where_clauses: List[WhereClause]
  select_clause: SelectClause

  def evaluate(self, table):
    rows = table.rows
    # TODO Consider combining into a single statement.
    for clause in self.where_clauses:
      try:
        rows = clause.filter(table, rows)
      except ValueError:
        return None

    if not rows:
      return None

    return self.select_clause.evaluate(table, rows)

  def verbalize(self):
    qualifiers = []
    for row in self.where_clauses:
      qualifiers.append(row.verbalize())

    qualification = ''
    if qualifiers:
      qualification = f' when {" and ".join(qualifiers)}'

    message = self.select_clause.verbalize()
    return f'{message}{qualification}'

  def increment_counters(self, name, counter):
    self.select_clause.increment_counters(name, counter)
    for row in self.where_clauses:
      row.increment_counters(name, counter)


@dataclasses.dataclass(frozen=True)
class ConstantExpression(Expression):
  """A constant expression is a set of values that don't depend on a table."""
  values: Set[Text]

  def evaluate(self, table):
    return self.values

  def verbalize(self):
    if not self.values:
      return 'empty'
    return ', '.join(sorted(self.values))

  def increment_counters(self, name, counter):
    pass


@dataclasses.dataclass(frozen=True)
class Condition:
  """A condition represents logical equality or inequality."""
  left: Expression
  right: Expression
  comparator: Comparator

  def evaluate(self, table):
    left = self.left.evaluate(table)
    if left is None:
      return EvaluationResult.ERROR
    right = self.right.evaluate(table)
    if right is None:
      return EvaluationResult.ERROR
    return self.comparator.compare(left, right)

  def verbalize(self):
    left = self.left.verbalize()
    right = self.right.verbalize()
    inter_text = self.comparator.verbalize()
    return f'{left} {inter_text} {right}'

  def increment_counters(self, name, counter):
    for expression in [self.left, self.right]:
      expression.increment_counters(name, counter)
    self.comparator.increment_counters(name, counter)


@dataclasses.dataclass(frozen=True)
class Statement:
  """A statement is a conjunction of conditions."""
  conditions: List[Condition]

  def evaluate(self, table):
    final_result = EvaluationResult.TRUE
    for condition in self.conditions:
      result = condition.evaluate(table)
      if result == EvaluationResult.ERROR:
        return result
      if result == EvaluationResult.FALSE:
        final_result = EvaluationResult.FALSE
    return final_result

  def verbalize(self):
    return ' and '.join([c.verbalize() for c in self.conditions])

  def increment_counters(self, name, counter):
    for condition in self.conditions:
      condition.increment_counters(name, counter)


def _column_index_by_name(table, column_name):
  indexes = [
      index for index, column in enumerate(table.columns)
      if column == column_name
  ]
  if not indexes:
    raise ValueError('Unknown column name')
  if len(indexes) > 1:
    raise ValueError('Ambiguous column name')
  return indexes[0]


def _value_from_index(row, index):
  return row.cells[index]


def _have_rank(values):
  return any([value.rank is not None for value in values])


def _bool_to_result(b):
  if b:
    return EvaluationResult.TRUE
  return EvaluationResult.FALSE


def _to_floats(values):
  return [text_utils.convert_to_float(v) for v in values]


def _add_where_clause(
    expression_fn,
    rng,
    table,
    column_name,
    where_clauses,
):
  """Attempts to add a row selection and creates a new expression."""
  try:
    column_index = _column_index_by_name(table, column_name)
  except ValueError:
    return None
  # TODO Consider allowing other (numeric) values in expression.
  column_values = (row.cells[column_index].value for row in table.rows)
  pairs = list(itertools.product(column_values, Comparator))
  rng.shuffle(pairs)
  for column_value, comparator in pairs:
    new_expr = expression_fn(
        where_clauses + [WhereClause(column_name, column_value, comparator)])
    new_values = new_expr.evaluate(table)
    if new_values is None:
      continue
    else:
      return new_expr, new_values
  return None


def _synthesize_where_clause(
    config,
    expression_fn,
    rng,
    table,
):
  """Synthesizes a random row selection.

  Args:
    config: Controls sample probabilities.
    expression_fn: Creates an expression.
    rng: random number gnerator.
    table: The table

  Returns:
    The final expression and its values WRT the table.

  Raises:
    SynthesizationError: if no valid expression can be built.
  """
  expr = expression_fn([])
  values = expr.evaluate(table)
  if values is None:
    raise SynthesizationError('Cannot create expression')

  # Attempt to add a row selection.
  column_names = list(table.columns)
  rng.shuffle(column_names)
  for row_select_name in column_names:
    result = _add_where_clause(expression_fn, rng, table, row_select_name,
                               expr.where_clauses)
    if result is None:
      continue
    if rng.random_sample() > config.prob_stop_value_row_expansion:
      return result
    expr, values = result
  return expr, values


def _synthesize_expression(
    config,
    rng,
    table,
    value_name,
):
  """Creates a synthetic expression that evaluates to some values or fails."""

  def _create_expression(
      where_clauses,
      aggregation,
  ):
    return ComplexExpression(
        where_clauses=where_clauses,
        select_clause=ValueAggregationClause(aggregation, value_name),
    )

  aggregations = list(Aggregation)
  rng.shuffle(aggregations)

  for aggregation in aggregations:
    expression_fn = functools.partial(
        _create_expression,
        aggregation=aggregation,
    )
    try:
      return _synthesize_where_clause(
          config=config,
          expression_fn=expression_fn,
          rng=rng,
          table=table,
      )
    except SynthesizationError:
      continue

  raise SynthesizationError('Cannot create expression')


def _get_table_with_shuffled_column(
    rng,
    table,
    column_name,
):
  """Creates a new table by shuffling a single column."""

  column_index = _column_index_by_name(table, column_name)

  values = [row.cells[column_index] for row in table.rows]
  rng.shuffle(values)

  new_rows = [list(row.cells) for row in table.rows]
  for row_index, row in enumerate(new_rows):
    row[column_index] = values[row_index]

  return Table(
      columns=table.columns,
      rows=[TableRow(row) for row in new_rows],
  )


def _get_opposite_table(
    rng,
    table,
    condition,
    column_name,
    current_result,
    attempts,
):
  """Attempts to create a new table with the opposite result for a condition."""
  for _ in range(attempts):
    try:
      new_table = _get_table_with_shuffled_column(
          rng,
          table,
          column_name,
      )
    except ValueError:
      continue
    result = condition.evaluate(new_table)
    if result == EvaluationResult.ERROR:
      continue
    if current_result != result:
      return new_table
  return None


def _get_condition(
    config,
    rng,
    left,
    left_values,
    right,
    right_values,
    comparator,
):
  """Creates a condition and randomly converts expression to constants."""
  if config.prob_statement_replacement > rng.random_sample():
    left = ConstantExpression(left_values)
  elif config.prob_statement_replacement > rng.random_sample():
    right = ConstantExpression(right_values)
  if left == right:
    # Make sure we don't generate a completely tautological condition.
    left = ConstantExpression(left_values)
  condition = Condition(left, right, comparator)
  return condition


def _find_comparator_and_opposite_table(
    config,
    rng,
    table,
    column_name,
    left,
    left_values,
    right,
    right_values,
    result,
    add_opposite_table,
):
  """Attempts to find a valid comparator and the opposite table."""
  comparators = list(Comparator)
  rng.shuffle(comparators)

  for comparator in comparators:
    if comparator.compare(left_values, right_values) == result:
      # Replace complex statements with constant expressions.
      condition = _get_condition(config, rng, left, left_values, right,
                                 right_values, comparator)
      if add_opposite_table:
        opposite_table = _get_opposite_table(
            rng,
            table,
            condition,
            column_name,
            result,
            attempts=config.attempts,
        )
        if opposite_table is None:
          continue
        return condition, opposite_table
      else:
        return condition, None
  raise SynthesizationError("Couldn't find comparator and table")


def _synthesize_count_condition(
    config,
    rng,
    table,
    result,
    add_opposite_table,
):
  """Attempts to synthesize a count-based expression."""

  def _create_expression(where_clauses):
    return ComplexExpression(
        where_clauses=where_clauses,
        select_clause=CountSelectClause(),
    )

  left, left_values = _synthesize_where_clause(config, _create_expression, rng,
                                               table)
  right, right_values = _synthesize_where_clause(config, _create_expression,
                                                 rng, table)
  columns = list(table.columns)
  rng.shuffle(columns)
  for column in columns:
    try:
      return _find_comparator_and_opposite_table(
          config,
          rng,
          table,
          column,
          left,
          left_values,
          right,
          right_values,
          result,
          add_opposite_table,
      )
    except SynthesizationError:
      continue
  raise SynthesizationError('Could not create count condition.')


def _synthesize_condition(
    config,
    rng,
    table,
    result,
    add_opposite_table,
):
  """Attempts to synthesize a condition."""

  if rng.random_sample() < config.prob_count_aggregation:
    # Creates a count example.
    # Count examples are special because their values are in a different domain.
    return _synthesize_count_condition(
        config,
        rng,
        table,
        result,
        add_opposite_table,
    )

  # All aggregations except count maintain the domain so if we create two
  # expressions for the same column they should be comparable.
  columns = list(table.columns)
  rng.shuffle(columns)
  for column in columns:
    left, left_values = _synthesize_expression(config, rng, table, column)
    for _ in range(config.attempts):
      right, right_values = _synthesize_expression(config, rng, table, column)

      try:
        return _find_comparator_and_opposite_table(
            config,
            rng,
            table,
            column,
            left,
            left_values,
            right,
            right_values,
            result,
            add_opposite_table,
        )
      except SynthesizationError:
        continue
  raise SynthesizationError("Couldn't synthesize condition")


def _synthesize_statement(
    config,
    rng,
    table,
    result,
    add_opposite_table,
):
  """Attempts to synthesize a statement with a single condition."""
  for _ in range(config.attempts):
    try:
      condition, opposite_table = _synthesize_condition(
          config,
          rng,
          table,
          result,
          add_opposite_table,
      )
    except SynthesizationError:
      continue
    return Statement([condition]), opposite_table
  raise SynthesizationError('Couldn\'t synthesize condition')


def synthesize_from_interaction(
    config,
    rng,
    interaction,
    counter,
    add_opposite_table = True,
):
  """Synthesizes up to for new statements for a given interaction.

  Args:
    config: Specifies sampling hyper-parameters.
    rng: A random number generator.
    interaction: Table will be used to generate random statements.
    counter: For aggregating statistics.
    add_opposite_table: If true, add opposite table.

  Returns:
    A list of up to 4 statements.

  Attempts to synthesize one positive and one negative statement.
  For each statement attempts to generate a table with the opposite outcome.
  """
  table = _from_table_proto(interaction.table)

  def get_interaction(interaction, table, statement, result, name):
    new_interaction = interaction_pb2.Interaction()
    if interaction.id:
      new_interaction.id = interaction.id
    else:
      new_interaction.id = interaction.table.table_id
    new_interaction.table.CopyFrom(_to_table_proto(table))
    new_interaction.table.table_id = interaction.table.table_id
    new_question = new_interaction.questions.add()
    new_question.id = new_interaction.id + '_' + name
    new_question.original_text = statement.verbalize()
    new_question.text = text_utils.normalize_for_match(
        new_question.original_text)
    if result == EvaluationResult.TRUE:
      new_question.answer.class_index = 1
    elif result == EvaluationResult.FALSE:
      new_question.answer.class_index = 0
    else:
      raise ValueError('Unexpected: {result}')
    return new_interaction

  interactions = []

  for name, result in [
      ('pos', EvaluationResult.TRUE),
      ('neg', EvaluationResult.FALSE),
  ]:
    try:
      statement, opposite_table = _synthesize_statement(
          config,
          rng,
          table,
          result,
          add_opposite_table,
      )
    except SynthesizationError as e:
      counter.count(f'{name}: Synthesization failure')
      counter.count(f'{name}: Synthesization failure: {e}')
      continue
    counter.count(f'{name}: Synthesization success')

    statement.increment_counters(name, counter)

    interactions.append(
        get_interaction(interaction, table, statement, result, name))
    if add_opposite_table:
      if result == EvaluationResult.FALSE:
        opposite_result = EvaluationResult.TRUE
      else:
        opposite_result = EvaluationResult.FALSE
      interactions.append(
          get_interaction(interaction, opposite_table, statement,
                          opposite_result, name))

  return interactions
