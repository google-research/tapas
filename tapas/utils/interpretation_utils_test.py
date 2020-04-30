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
# coding=utf8

import random

from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.utils import interpretation_utils

_Candidate = interpretation_utils.Candidate
_AggFun = interaction_pb2.Answer.AggregationFunction


def _to_interaction(table, float_value):
  interaction = interaction_pb2.Interaction()
  for row in table:
    new_row = interaction.table.rows.add()
    for value in row:
      new_cell = new_row.cells.add()
      if value is not None:
        new_cell.numeric_value.float_value = value
  new_question = interaction.questions.add()
  new_question.answer.float_value = float_value
  return interaction


class InterpretationUtilsTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters((0,), (1,), (2,), (3,), (4.), (5,))
  def test_float_conversion(self, seed):
    interaction = _to_interaction(
        [[1.0, 3.0, 6.0], [2.0, 0.0, None], [1.0, None, 0.0]], 3.0)
    rng = random.Random(seed)
    actual = interpretation_utils.find_candidates(rng, interaction.table,
                                                  interaction.questions[0])
    expected = [
        _Candidate(_AggFun.COUNT, 0, (0, 1, 2)),
        _Candidate(_AggFun.COUNT, 1, (0, 1, 2)),
        _Candidate(_AggFun.COUNT, 2, (0, 1, 2)),
        _Candidate(_AggFun.SUM, 0, (0, 1)),
        _Candidate(_AggFun.SUM, 0, (1, 2)),
        _Candidate(_AggFun.SUM, 1, (0,)),
        _Candidate(_AggFun.SUM, 1, (0, 1)),
        _Candidate(_AggFun.AVERAGE, 1, (0,)),
        _Candidate(_AggFun.AVERAGE, 2, (0, 2)),
    ]
    self.assertEqual(expected, actual)

  @parameterized.parameters((0,), (1,), (2,), (3,), (4.), (5,))
  def test_random_sampling(self, seed):
    interaction = _to_interaction(
        [[1.0, 3.0, 6.0], [2.0, 0.0, None], [1.0, None, 0.0]], 3.0)
    rng = random.Random(seed)

    interpretation_utils._MAX_NUM_CANDIDATES = 1

    actual = interpretation_utils.find_candidates(rng, interaction.table,
                                                  interaction.questions[0])
    self.assertLen(actual, 3)

  @parameterized.parameters((0,), (1,), (2,), (3,), (4.), (5,))
  def test_random_exploration(self, seed):
    interaction = _to_interaction(
        [[1.0, 3.0, 6.0], [2.0, 0.0, None], [1.0, None, 0.0]], 3.0)
    rng = random.Random(seed)

    interpretation_utils._MAX_INDICES_TO_EXPLORE = 1

    actual = interpretation_utils.find_candidates(rng, interaction.table,
                                                  interaction.questions[0])
    expected = [
        _Candidate(_AggFun.COUNT, 0, (0, 1, 2)),
        _Candidate(_AggFun.COUNT, 1, (0, 1, 2)),
        _Candidate(_AggFun.COUNT, 2, (0, 1, 2)),
        _Candidate(_AggFun.SUM, 1, (0,)),
        _Candidate(_AggFun.AVERAGE, 1, (0,)),
    ]
    self.assertEqual(expected, actual)

  @parameterized.parameters((0,), (1,), (2,), (3,), (4.), (5,))
  def test_selection_answer(self, seed):
    interaction = _to_interaction(
        [[1.0, 3.0, 6.0], [2.0, 0.0, None], [1.0, None, 0.0]], 100.0)
    coords = interaction.questions[0].answer.answer_coordinates.add()
    coords.row_index = 1
    coords.column_index = 2
    rng = random.Random(seed)

    actual = interpretation_utils.find_candidates(rng, interaction.table,
                                                  interaction.questions[0])
    expected = [_Candidate(_AggFun.NONE, 2, (1,))]
    self.assertEqual(expected, actual)


if __name__ == "__main__":
  absltest.main()
