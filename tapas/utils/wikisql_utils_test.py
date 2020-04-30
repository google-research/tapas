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
import json
import os
import tempfile

from absl import logging
from absl.testing import absltest
from tapas.utils import wikisql_utils
import tensorflow.compat.v1 as tf


def _create_inputs(input_dir, tables, examples):
  for name in wikisql_utils._DATASETS:
    filename = os.path.join(input_dir, f'{name}.tables.jsonl')
    with tf.io.gfile.GFile(filename, 'w') as table_file:
      for table in tables:
        table_file.write(json.dumps(table) + '\n')

  for name in wikisql_utils._DATASETS:
    filename = os.path.join(input_dir, f'{name}.jsonl')
    with tf.io.gfile.GFile(filename, 'w') as example_file:
      for example in examples:
        example_file.write(json.dumps(example) + '\n')


class WikisqlUtilsTest(absltest.TestCase):

  def test_simple_test(self):
    with tempfile.TemporaryDirectory() as input_dir:
      with tempfile.TemporaryDirectory() as output_dir:
        _create_inputs(
            input_dir,
            tables=[{
                'id': '1-0000001-1',
                'header': ['Text', 'Number'],
                'types': ['text', 'real'],
                'rows': [['A', 1], ['B', 2], ['C', 3]],
            }],
            examples=[
                {
                    'question': 'What is text for 2?',
                    'table_id': '1-0000001-1',
                    'sql': {
                        'agg': 0,  # No aggregation
                        'sel': 0,  # Text column
                        'conds': [[1, 0, 2]]  # Column 1 = 2
                    },
                },
                {
                    'question': 'What is sum when number is greater than 1?',
                    'table_id': '1-0000001-1',
                    'sql': {
                        'agg': 4,  # SUM
                        'sel': 1,  # Number column
                        'conds': [[1, 1, 1]]  # Column 1 > 1
                    }
                }
            ])

        wikisql_utils.convert(input_dir=input_dir, output_dir=output_dir)

        table_path = os.path.join(
            output_dir,
            wikisql_utils._TABLE_DIR_NAME,
            '1-0000001-1.csv',
        )
        with tf.io.gfile.GFile(table_path) as table_file:
          actual = [dict(row) for row in csv.DictReader(table_file)]
          self.assertEqual([{
              'Text': 'A',
              'Number': '1',
          }, {
              'Text': 'B',
              'Number': '2',
          }, {
              'Text': 'C',
              'Number': '3'
          }], actual)

        filename = os.path.join(output_dir, 'dev.tsv')
        with tf.io.gfile.GFile(filename) as dev_file:
          actual = list(csv.DictReader(dev_file, delimiter='\t'))
          logging.info(actual)
          self.assertEqual(
              {
                  'id': 'dev-0',
                  'annotator': '0',
                  'position': '0',
                  'question': 'What is text for 2?',
                  'table_file': 'table_csv/1-0000001-1.csv',
                  'answer_coordinates': "['(1, 0)']",
                  'aggregation': '',
                  'answer_text': "['B']",
                  'float_answer': '',
              }, dict(actual[0]))
          self.assertEqual(
              {
                  'id': 'dev-1',
                  'annotator': '0',
                  'position': '0',
                  'question': 'What is sum when number is greater than 1?',
                  'table_file': 'table_csv/1-0000001-1.csv',
                  'answer_coordinates': "['(1, 1)', '(2, 1)']",
                  'aggregation': 'SUM',
                  'answer_text': "['5.0']",
                  'float_answer': '5.0',
              }, dict(actual[1]))


if __name__ == '__main__':
  absltest.main()
