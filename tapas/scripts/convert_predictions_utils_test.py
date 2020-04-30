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
import pandas as pd
from tapas.protos import interaction_pb2
from tapas.scripts import convert_predictions_utils
import tensorflow.compat.v1 as tf

Cell = interaction_pb2.Cell
Cells = interaction_pb2.Cells
Table = interaction_pb2.Table


class ConvertPredictionsUtilsTest(absltest.TestCase):

  def test_convert_single_no_pred_aggr(self):
    interactions_path = tempfile.mktemp()
    with tf.python_io.TFRecordWriter(interactions_path) as writer:
      writer.write(
          interaction_pb2.Interaction(
              id='dev-1-2_3',
              table=Table(
                  columns=[Cell(text='A')],
                  rows=[Cells(cells=[Cell(text='answer')])],
              )).SerializeToString())
      writer.write(
          interaction_pb2.Interaction(id='dev-2-1_3').SerializeToString())

    predictions_path = tempfile.mktemp()
    predictions_df = pd.DataFrame(
        columns=['id', 'annotator', 'position', 'answer_coordinates'],
        data=[['dev-1', '2', '3', '["(0,0)"]']])
    predictions_df.to_csv(predictions_path, sep='\t', index=False)

    output_path = tempfile.mktemp()
    convert_predictions_utils._convert_single(
        interactions_path,
        predictions_path,
        output_path,
        convert_predictions_utils.DatasetFormat.WIKITABLEQUESTIONS,
    )

    with open(output_path, 'rt') as file_handle:
      self.assertEqual(file_handle.read(), 'dev-1\tanswer\ndev-2\n')


if __name__ == '__main__':
  absltest.main()
