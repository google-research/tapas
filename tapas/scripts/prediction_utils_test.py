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

import tempfile
from absl.testing import absltest
from tapas.protos import interaction_pb2
from tapas.scripts import calc_metrics_utils
from tapas.scripts import prediction_utils
import tensorflow.compat.v1 as tf


Cell = interaction_pb2.Cell
Cells = interaction_pb2.Cells
Table = interaction_pb2.Table


class PredictionUtilsTest(absltest.TestCase):

  def test_iterate_interactions(self):
    filepath = tempfile.mktemp()
    interactions = [
        interaction_pb2.Interaction(id='dev_723'),
        interaction_pb2.Interaction(id='dev_456'),
        interaction_pb2.Interaction(id='dev_123'),
    ]
    with tf.io.TFRecordWriter(filepath) as writer:
      for interaction in interactions:
        writer.write(interaction.SerializeToString())
    actual_interactions = list(prediction_utils.iterate_interactions(filepath))
    self.assertEqual(interactions, actual_interactions)

  def table_to_panda_frame(self):
    frame = prediction_utils.table_to_panda_frame(
        Table(
            columns=[Cell(text='a'), Cell(text='a')],
            rows=[Cells(cells=[Cell(text='0'), Cell(text='1')])]))
    self.assertEqual(['0'],
                     calc_metrics_utils._collect_cells_from_table({(0, 0)},
                                                                  frame))


if __name__ == '__main__':
  absltest.main()
