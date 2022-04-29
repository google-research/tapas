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

import collections

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tapas.protos import interaction_pb2
from tapas.utils import synthesize_entablement

Aggregation = synthesize_entablement.Aggregation
CellValue = synthesize_entablement.CellValue
Comparator = synthesize_entablement.Comparator
ComplexExpression = synthesize_entablement.ComplexExpression
Condition = synthesize_entablement.Condition
ConstantExpression = synthesize_entablement.ConstantExpression
CountSelectClause = synthesize_entablement.CountSelectClause
EvaluationResult = synthesize_entablement.EvaluationResult
Statement = synthesize_entablement.Statement
Table = synthesize_entablement.Table
TableRow = synthesize_entablement.TableRow
ValueAggregationClause = synthesize_entablement.ValueAggregationClause
WhereClause = synthesize_entablement.WhereClause


class TestCounter(synthesize_entablement.Counter):

  def __init__(self):
    self._counts = collections.Counter()

  def count(self, message):
    self._counts[message] += 1

  def get_counts(self):
    return dict(self._counts)


class SynthesizeEntablementTest(parameterized.TestCase):

  def test_format(self):
    self.assertEqual(synthesize_entablement._format_number(1.01), '1.01')
    self.assertEqual(synthesize_entablement._format_number(1.001), '1.00')
    self.assertEqual(synthesize_entablement._format_number(1.511), '1.51')
    self.assertEqual(synthesize_entablement._format_number(1.00), '1')

  @parameterized.parameters(
      ('1', '1', Comparator.EQUALS, EvaluationResult.TRUE),
      ('1', '2', Comparator.EQUALS, EvaluationResult.FALSE),
      ('1', '2', Comparator.GREATER, EvaluationResult.FALSE),
      ('1', '2', Comparator.LESSER, EvaluationResult.TRUE),
      ('2', '1', Comparator.GREATER, EvaluationResult.TRUE),
      ('2', '1', Comparator.LESSER, EvaluationResult.FALSE),
      ('a', 'a', Comparator.EQUALS, EvaluationResult.TRUE),
      ('a', 'b', Comparator.EQUALS, EvaluationResult.FALSE),
      ('a', 'b', Comparator.GREATER, EvaluationResult.ERROR),
      ('a', 'b', Comparator.LESSER, EvaluationResult.ERROR),
  )
  def test_constant_singleton_expressions(self, left, right, cmp, result):
    statement = Statement([
        Condition(
            left=ConstantExpression({left}),
            right=ConstantExpression({right}),
            comparator=cmp,
        )
    ])
    empty_table = Table(columns=[], rows=[])
    self.assertEqual(statement.evaluate(empty_table), result)

  @parameterized.parameters(
      ({'1', '2'}, {'1', '2'}, Comparator.EQUALS, EvaluationResult.TRUE),
      ({'1', '2'}, {'2', '3'}, Comparator.GREATER, EvaluationResult.FALSE),
      ({'1', '2'}, {'2', '3'}, Comparator.LESSER, EvaluationResult.FALSE),
      ({'1', '2'}, {'3', '4'}, Comparator.GREATER, EvaluationResult.FALSE),
      ({'1', '2'}, {'3', '4'}, Comparator.LESSER, EvaluationResult.TRUE),
      ({'3', '4'}, {'1', '2'}, Comparator.GREATER, EvaluationResult.TRUE),
      ({'3', '4'}, {'1', '2'}, Comparator.LESSER, EvaluationResult.FALSE),
  )
  def test_constant_expressions(self, left, right, cmp, result):
    statement = Statement([
        Condition(
            left=ConstantExpression(left),
            right=ConstantExpression(right),
            comparator=cmp,
        )
    ])
    empty_table = Table(columns=[], rows=[])
    self.assertEqual(statement.evaluate(empty_table), result)

  @parameterized.parameters(
      (None, 'B', Aggregation.GREATEST, {'1'}),
      (None, 'B', Aggregation.LAST, {'1'}),
      (None, 'B', Aggregation.LOWEST, {'0'}),
      (None, 'C', Aggregation.FIRST, {'5'}),
      (None, 'C', Aggregation.GREATEST, {'5'}),
      (None, 'C', Aggregation.LAST, {'3'}),
      (None, 'C', Aggregation.AVERAGE, {'4'}),
      (None, 'C', Aggregation.LOWEST, {'3'}),
      (None, 'C', Aggregation.NONE, {'5', '4', '3'}),
      (None, 'C', Aggregation.RANGE, {'2'}),
      (None, 'C', Aggregation.SUM, {'12'}),
      (WhereClause('A', 'a'), 'C', Aggregation.NONE, {'5'}),
      (WhereClause('A', 'b'), 'B', Aggregation.NONE, {'1'}),
      (WhereClause('A', 'b'), 'C', Aggregation.NONE, {'4'}),
      (
          WhereClause('C', '3', Comparator.GREATER),
          'C',
          Aggregation.NONE,
          {'4', '5'},
      ),
      (
          WhereClause('C', '5', Comparator.LESSER),
          'C',
          Aggregation.NONE,
          {'3', '4'},
      ),
  )
  def test_complex_expressions(
      self,
      where_clause,
      select_value_name,
      aggregation,
      right_value,
  ):
    table = Table(
        columns=['A', 'B', 'C'],
        rows=[
            TableRow(
                [CellValue('a', None),
                 CellValue('0', 0),
                 CellValue('5', 2)]),
            TableRow(
                [CellValue('b', None),
                 CellValue('1', 1),
                 CellValue('4', 1)]),
            TableRow(
                [CellValue('c', None),
                 CellValue('1', 1),
                 CellValue('3', 0)]),
        ])
    statement = Statement([
        Condition(
            left=ComplexExpression(
                where_clauses=[] if where_clause is None else [where_clause],
                select_clause=ValueAggregationClause(aggregation,
                                                     select_value_name)),
            right=ConstantExpression(right_value),
            comparator=Comparator.EQUALS,
        )
    ])
    self.assertEqual(statement.evaluate(table), EvaluationResult.TRUE)

  @parameterized.parameters(
      ('1', '1', Comparator.EQUALS, '1 is 1'),
      ('1', '2', Comparator.EQUALS, '1 is 2'),
      ('1', '2', Comparator.GREATER, '1 is greater than 2'),
      ('1', '2', Comparator.LESSER, '1 is less than 2'),
      ('2', '1', Comparator.GREATER, '2 is greater than 1'),
      ('2', '1', Comparator.LESSER, '2 is less than 1'),
      ('a', 'a', Comparator.EQUALS, 'a is a'),
      ('a', 'b', Comparator.EQUALS, 'a is b'),
      ('a', 'b', Comparator.GREATER, 'a is greater than b'),
      ('a', 'b', Comparator.LESSER, 'a is less than b'),
  )
  def test_verbalize_constant_singleton_expressions(self, left, right, cmp,
                                                    result):
    statement = Statement([
        Condition(
            left=ConstantExpression({left}),
            right=ConstantExpression({right}),
            comparator=cmp,
        )
    ])
    self.assertEqual(statement.verbalize(), result)

  @parameterized.parameters(
      ({'1', '2'}, {'1', '2'}, Comparator.EQUALS, '1, 2 is 1, 2'),
      ({'1', '2'}, {'2', '3'}, Comparator.GREATER, '1, 2 is greater than 2, 3'),
      ({'1', '2'}, {'2', '3'}, Comparator.LESSER, '1, 2 is less than 2, 3'),
      ({'1', '2'}, {'3', '4'}, Comparator.GREATER, '1, 2 is greater than 3, 4'),
      ({'1', '2'}, {'3', '4'}, Comparator.LESSER, '1, 2 is less than 3, 4'),
      ({'3', '4'}, {'1', '2'}, Comparator.GREATER, '3, 4 is greater than 1, 2'),
      ({'3', '4'}, {'1', '2'}, Comparator.LESSER, '3, 4 is less than 1, 2'),
  )
  def test_verbalize_constant_expressions(self, left, right, cmp, result):
    statement = Statement([
        Condition(
            left=ConstantExpression(left),
            right=ConstantExpression(right),
            comparator=cmp,
        )
    ])
    self.assertEqual(statement.verbalize(), result)

  @parameterized.parameters(
      (None, 'B', Aggregation.AVERAGE, {'0'}, 'the average B is 0'),
      (None, 'B', Aggregation.RANGE, {'0'}, 'the range of B is 0'),
      (None, 'B', Aggregation.GREATEST, {'1'}, 'the greatest B is 1'),
      (None, 'B', Aggregation.LAST, {'1'}, 'the last B is 1'),
      (None, 'B', Aggregation.LOWEST, {'0'}, 'the lowest B is 0'),
      (None, 'B', Aggregation.SUM, {'0'}, 'the total B is 0'),
      (None, 'C', Aggregation.FIRST, {'5'}, 'the first C is 5'),
      (None, 'C', Aggregation.GREATEST, {'5'}, 'the greatest C is 5'),
      (None, 'C', Aggregation.LAST, {'3'}, 'the last C is 3'),
      (None, 'C', Aggregation.LOWEST, {'3'}, 'the lowest C is 3'),
      (None, 'C', Aggregation.NONE, {'5', '4', '3'}, 'C is 3, 4, 5'),
      (WhereClause('Name', 'Mike'), 'Points', Aggregation.NONE, {'5'},
       'Points when Name is Mike is 5'),
      (WhereClause('Name', 'Tina'), 'Grade', Aggregation.NONE, {'1'},
       'Grade when Name is Tina is 1'),
      (WhereClause('Name', 'Tina'), 'Height', Aggregation.NONE, {'4'},
       'Height when Name is Tina is 4'),
      (WhereClause('Points', '3',
                   Comparator.GREATER), 'Points', Aggregation.NONE, {'4', '5'},
       'Points when Points is greater than 3 is 4, 5'),
      (WhereClause('Points', '5',
                   Comparator.LESSER), 'Points', Aggregation.NONE, {'3', '4'},
       'Points when Points is less than 5 is 3, 4'),
  )
  def test_verbalize_complex_expressions(
      self,
      where_clause,
      select_value_name,
      aggregation,
      right_value,
      result,
  ):
    statement = Statement([
        Condition(
            left=ComplexExpression(
                where_clauses=[] if where_clause is None else [where_clause],
                select_clause=ValueAggregationClause(
                    aggregation=aggregation, column_name=select_value_name),
            ),
            right=ConstantExpression(right_value),
            comparator=Comparator.EQUALS,
        )
    ])
    self.assertEqual(statement.verbalize(), result)

  def test_table_conversion(self):
    table = Table(
        columns=['A', 'B', 'C'],
        rows=[
            TableRow(
                [CellValue('a', None),
                 CellValue('1', 0),
                 CellValue('300', 2)]),
            TableRow(
                [CellValue('a', None),
                 CellValue('1', 0),
                 CellValue('200', 1)]),
            TableRow(
                [CellValue('a', None),
                 CellValue('3', 1),
                 CellValue('100', 0)]),
        ])
    proto_table = synthesize_entablement._to_table_proto(table)
    new_table = synthesize_entablement._from_table_proto(proto_table)
    logging.info('table: %s', table)
    logging.info('new_table: %s', new_table)
    self.assertEqual(table, new_table)

  @parameterized.parameters((1.0), (0.0))
  def test_interaction(self, prob_count_aggregation):
    config = synthesize_entablement.SynthesizationConfig(
        prob_count_aggregation=prob_count_aggregation, attempts=10)
    interaction = interaction_pb2.Interaction(
        id='i_id',
        table=interaction_pb2.Table(
            table_id='t_id',
            columns=[
                interaction_pb2.Cell(text='Name'),
                interaction_pb2.Cell(text='Height'),
                interaction_pb2.Cell(text='Age')
            ],
            rows=[
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text='Peter'),
                    interaction_pb2.Cell(text='100'),
                    interaction_pb2.Cell(text='15')
                ]),
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text='Bob'),
                    interaction_pb2.Cell(text='150'),
                    interaction_pb2.Cell(text='15')
                ]),
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text='Tina'),
                    interaction_pb2.Cell(text='200'),
                    interaction_pb2.Cell(text='17')
                ]),
            ]),
        questions=[])

    pos_statements = set()
    neg_statements = set()

    counter = TestCounter()

    for i in range(100):
      rng = np.random.RandomState(i)
      interactions = synthesize_entablement.synthesize_from_interaction(
          config, rng, interaction, counter)
      for new_interaction in interactions:
        question = new_interaction.questions[0]
        if question.answer.class_index == 1:
          pos_statements.add(question.text)
        else:
          assert question.answer.class_index == 0
          neg_statements.add(question.text)
    self.assertEqual(neg_statements, pos_statements)
    logging.info('pos_statements: %s', pos_statements)

    counts = counter.get_counts()
    logging.info('counts: %s', counts)

    is_count_test = prob_count_aggregation == 1.0

    if is_count_test:
      self.assertGreater(len(pos_statements), 10)
      expected_statements = {
          '1 is less than the count when age is 15 and height is greater than 100',
          '1 is less than the count when height is less than 200 and age is 15',
          '1 is the count when height is greater than 100 and age is less than 17',
          '2 is the count when age is less than 17 and height is less than 200',
      }
    else:
      self.assertGreater(len(pos_statements), 100)
      expected_statements = {
          '0 is the range of age when height is greater than 100',
          '100 is less than the last height when height is less than 200',
          '125 is greater than height when name is peter',
          '15 is age when height is less than 150',
          '15 is the last age when height is less than 200',
          '150 is the last height when age is 15',
          '175 is the average height when age is less than 17',
          '200 is greater than the greatest height when age is less than 17',
          '250 is less than the total height when age is 15',
          '30 is less than the total age when height is greater than 100',
          'bob is name when age is greater than 15',
          'bob is the first name when age is 15 and height is less than 200',
          'peter is name when age is 15 and height is less than 150',
          'the average height when age is 15 is less than 175',
          'the first height when height is greater than 100 is 150',
          'the first height when height is less than 200 is 150',
          'the first name when age is 15 is name when name is peter',
          'the greatest height when age is 15 is less than 200',
          'the last age when height is greater than 100 is greater than 15',
          'the last name when age is 15 is bob',
          'the last name when age is less than 17 is peter',
          'the last name when height is greater than 100 is bob',
          'the last name when height is less than 200 is bob',
          'the lowest height when age is 15 is 150',
          'the range of age when height is greater than 100 is greater than 0',
          'the range of height when age is 15 is 100',
          'tina is name when age is greater than 15 and height is 200',
          'tina is the first name when age is 15',
          'tina is the last name when age is 15',
          'tina is the last name when height is greater than 100',
      }

    for statement in expected_statements:
      self.assertIn(statement, pos_statements)

    for name in ['pos', 'neg']:
      if is_count_test:
        self.assertGreater(counts[f'{name}: Synthesization success'], 10)
        self.assertGreater(counts[f'{name}: Select: COUNT'], 10)
      else:
        self.assertEqual(counts[f'{name}: Synthesization success'], 100)
        for aggregation in Aggregation:
          self.assertGreater(counts[f'{name}: Select: {aggregation.name}'], 0)
      for comparator in Comparator:
        min_count = 1 if prob_count_aggregation == 1.0 else 10
        self.assertGreater(
            counts[f'{name}: Comparator {comparator.name}'], min_count)
        self.assertGreater(
            counts[f'{name}: where: Comparator {comparator.name}'], min_count)

  def test_interaction_duplicate_column_name(self):
    """Test we don't crash when seeing ambiguous column names."""
    config = synthesize_entablement.SynthesizationConfig(attempts=10)
    interaction = interaction_pb2.Interaction(
        id='i_id',
        table=interaction_pb2.Table(
            table_id='t_id',
            columns=[
                interaction_pb2.Cell(text='Name'),
                interaction_pb2.Cell(text='Name'),
                interaction_pb2.Cell(text='Height')
            ],
            rows=[
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text='Peter'),
                    interaction_pb2.Cell(text='Peter'),
                    interaction_pb2.Cell(text='100')
                ]),
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text='Bob'),
                    interaction_pb2.Cell(text='Bob'),
                    interaction_pb2.Cell(text='150')
                ]),
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text='Tina'),
                    interaction_pb2.Cell(text='Tina'),
                    interaction_pb2.Cell(text='200')
                ]),
            ]),
        questions=[])
    for i in range(20):
      rng = np.random.RandomState(i)
      synthesize_entablement.synthesize_from_interaction(
          config, rng, interaction, synthesize_entablement.Counter())


if __name__ == '__main__':
  absltest.main()
