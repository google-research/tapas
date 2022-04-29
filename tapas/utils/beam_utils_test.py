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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.utils import beam_utils


class BeamUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (5, 10, [(0, 5)]),
      (5, 2, [(0, 2), (2, 3), (3, 5)]),
      (10, 2, [(0, 2), (2, 3), (3, 5), (5, 7), (7, 8), (8, 10)]),
  )
  def test_get_sharded_ranges(
      self,
      end,
      max_length,
      expected,
  ):
    self.assertEqual(
        beam_utils._get_sharded_ranges(0, end, max_length), expected)

  @parameterized.parameters(
      (5, 10, 50),
      (5, 10, 5),
      (5, 2, 5),
      (3, 7, 13),
  )
  def test_get_row_sharded_interactions(
      self,
      num_columns,
      num_rows,
      max_num_cells,
  ):
    interaction = interaction_pb2.Interaction()
    for i in range(num_columns):
      interaction.table.columns.add().text = f'{i}'
    for j in range(num_rows):
      row = interaction.table.rows.add()
      for i in range(num_columns):
        row.cells.add().text = f'{j}_{i}'
    interactions = list(
        beam_utils.get_row_sharded_interactions(interaction, max_num_cells))
    restorted_interaction = interaction_pb2.Interaction()
    restorted_interaction.CopyFrom(interaction)
    del restorted_interaction.table.rows[:]
    for shard in interactions:
      self.assertEqual(shard.table.columns, interaction.table.columns)
      self.assertLessEqual(len(shard.table.rows) * num_columns, max_num_cells)
      for row in shard.table.rows:
        restorted_interaction.table.rows.add().CopyFrom(row)
    logging.info(restorted_interaction)
    self.assertEqual(interaction, restorted_interaction)


if __name__ == '__main__':
  absltest.main()
