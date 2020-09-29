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

import csv
import os
import tempfile

from absl import logging
from absl.testing import absltest
from tapas.utils import wtq_utils
import tensorflow.compat.v1 as tf


def _create_inputs(input_dir, tables, examples):
  for table in tables:
    filename = os.path.join(input_dir, table['table_id'])
    tf.io.gfile.makedirs(os.path.dirname(filename))
    with tf.io.gfile.GFile(filename, 'w') as table_file:
      writer = csv.writer(table_file)
      writer.writerow(table['columns'])
      writer.writerows(table['rows'])

  for number in range(0, 6):
    for name in wtq_utils._get_train_test(number, wtq_utils.Version.V_10):
      filepath = os.path.join(input_dir, 'data', name)
      tf.io.gfile.makedirs(os.path.dirname(filepath))
      with tf.io.gfile.GFile(filepath, 'w') as example_file:
        writer = csv.DictWriter(
            example_file,
            fieldnames=[
                'id',
                'utterance',
                'context',
                'targetValue',
            ],
            delimiter='\t',
        )
        writer.writeheader()
        for example in examples:
          writer.writerow(example)


class WtqUtilsTest(absltest.TestCase):

  def test_simple_test(self):
    with tempfile.TemporaryDirectory() as input_dir:
      with tempfile.TemporaryDirectory() as output_dir:
        _create_inputs(
            input_dir,
            tables=[{
                'table_id': 'csv/203-csv/515.csv',
                'columns': ['Text', 'Number'],
                'rows': [['A', 1], ['B', 2], ['тапас', 3]],
            }],
            examples=[
                {
                    'id': 'nt-2',
                    'utterance': 'What is text for 2?',
                    'context': 'csv/203-csv/515.csv',
                    'targetValue': 'B',
                },
            ])

        wtq_utils.convert(input_dir=input_dir, output_dir=output_dir)

        table_dir = os.path.join(output_dir, wtq_utils._TABLE_DIR_NAME)
        self.assertCountEqual(
            tf.io.gfile.listdir(output_dir), [
                'random-split-1-dev.tsv',
                'random-split-1-train.tsv',
                'random-split-2-dev.tsv',
                'random-split-2-train.tsv',
                'random-split-3-dev.tsv',
                'random-split-3-train.tsv',
                'random-split-4-dev.tsv',
                'random-split-4-train.tsv',
                'random-split-5-dev.tsv',
                'random-split-5-train.tsv',
                'table_csv',
                'test.tsv',
                'train.tsv',
            ])
        self.assertEqual(tf.io.gfile.listdir(table_dir), ['203-515.csv'])

        table_path = os.path.join(table_dir, '203-515.csv')
        with tf.io.gfile.GFile(table_path) as table_file:
          actual = [dict(row) for row in csv.DictReader(table_file)]
          self.assertEqual([{
              'Text': 'a',
              'Number': '1',
          }, {
              'Text': 'b',
              'Number': '2',
          }, {
              'Text': 'тапас',
              'Number': '3'
          }], actual)

        filename = os.path.join(output_dir, 'test.tsv')
        with tf.io.gfile.GFile(filename) as dev_file:
          actual = list(csv.DictReader(dev_file, delimiter='\t'))
          logging.info(actual)
          self.assertEqual(
              {
                  'id': 'nt-2',
                  'annotator': '0',
                  'position': '0',
                  'question': 'What is text for 2?',
                  'table_file': 'table_csv/203-515.csv',
                  'answer_coordinates': "['(-1, -1)']",
                  'aggregation': 'NONE',
                  'answer_text': "['B']",
                  'float_answer': '',
              }, dict(actual[0]))


if __name__ == '__main__':
  absltest.main()
