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
import pickle
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from tapas.scripts import calc_metrics_utils
from tapas.scripts import prediction_utils
import tensorflow.compat.v1 as tf


def _read_data_examples(data_path):
  """Reads examples from a dataset csv file."""
  data_examples = {}
  with tf.io.gfile.GFile(data_path, mode='r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
      ex_id = '{}-{}_{}'.format(row['id'], row['annotator'], row['position'])
      question = row['question'].strip()
      table_id = row['table_file']
      gold_cell_coo = prediction_utils.parse_coordinates(
          row['answer_coordinates'])
      gold_agg_function = int(row['aggregation'])
      float_answer_raw = row['float_answer']
      float_answer = float(float_answer_raw) if float_answer_raw else None
      ex = calc_metrics_utils.Example(
          ex_id,
          question,
          table_id,
          None,
          gold_cell_coo,
          gold_agg_function,
          float_answer,
          has_gold_answer=True)
      data_examples[ex_id] = ex
  return data_examples


def _read_and_store_tables(tables_path, output_tables_file, tables_to_store):
  """Stores multiple tables as a dict of dataframes, for fast access.

  Args:
    tables_path: The path where all tables csv files are stored.
    output_tables_file: The file where all the data will be stored as a pickle
      dict.
    tables_to_store: A list of table ids to store in the output file.
  """
  all_tables = {}
  for f in tables_to_store:
    with tf.io.gfile.GFile(os.path.join(tables_path, f), mode='r') as f_table:
      table = pd.read_csv(f_table)
    all_tables[f] = table
  with tf.io.gfile.GFile(output_tables_file, 'wb') as f_out:
    pickle.dump(all_tables, f_out)


def _calc_denotation_accuracy(tables_file, examples, denotation_errors_path,
                              predictions_file_name):
  with tf.io.gfile.GFile(tables_file, 'rb') as f:
    tables = pickle.load(f)
  for example in examples.values():
    example.table = tables[example.table_id]
  return calc_metrics_utils.calc_denotation_accuracy(examples,
                                                     denotation_errors_path,
                                                     predictions_file_name)


def _write_predictions(data,
                       headers=('id', 'annotator', 'position',
                                'answer_coordinates', 'gold_aggr',
                                'pred_aggr')):
  predictions_df = pd.DataFrame(columns=headers, data=data)
  predictions_path = tempfile.mktemp(prefix='predictions_')
  with tf.io.gfile.GFile(predictions_path, 'w') as f:
    predictions_df.to_csv(f, sep='\t', index=False)
  return predictions_path


def _write_synthetic_predictions():
  predictions = [['dev-0', '0', '0', '["(2, 1)", "(2, 2)"]', '0', '0'],
                 ['dev-1', '0', '0', '["(1, 2)", "(1, 4)"]', '1', '0'],
                 ['dev-2', '0', '0', '["(6, 1)"]', '3', '0'],
                 ['dev-3', '0', '0', '["(0, 4)", "(1, 4)"]', '2', '2'],
                 ['dev-4', '0', '0', '["(0, 4)", "(1, 4)"]', '1', '1']]
  return _write_predictions(predictions)


def _write_table(test_tmpdir, headers, data):
  table_df = pd.DataFrame(columns=headers, data=data)
  table_path = tempfile.mktemp(prefix='table_', dir=test_tmpdir)
  table_name = os.path.basename(table_path)
  with tf.io.gfile.GFile(table_path, 'w') as f:
    table_df.to_csv(f, index=False)
  return table_name


def _write_synthetic_table(test_tmpdir):
  headers = [
      'Home team', 'Home team score', 'Away team', 'Away team score', 'Venue',
      'Crowd', 'Date'
  ]
  data = [[
      'St Kilda', '8.15', 'South Melbourne', '7.13', 'Junction Oval',
      '"22,000"', '14 May 1927'
  ],
          [
              'Footscray', '10.18', 'Hawthorn', '7.10', 'Western Oval',
              '"11,000"', '14 May 1927'
          ],
          [
              'Collingwood', '6.13', 'Richmond', '7.13', 'Victoria Park',
              '"20,000"', '14 May 1927'
          ],
          [
              'Carlton', '7.5', 'Essendon', '4.12', 'Princes Park', '"27,000"',
              '14 May 1927'
          ],
          [
              'North Melbourne', '8.5', 'Geelong', '7.13', 'Arden Street Oval',
              '"20,000"', '14 May 1927'
          ],
          [
              'Melbourne', '13.15', 'Fitzroy', '8.11', 'MCG', '"14,931"',
              '14 May 1927'
          ], ['total_count', '6', '6', '6', '6', '6', '-']]
  return _write_table(test_tmpdir, headers, data)


def _write_tables_dict(headers=None, data=None):
  test_tmpdir = tempfile.mkdtemp()
  if headers is None and data is None:
    table_name = _write_synthetic_table(test_tmpdir)
  else:
    table_name = _write_table(test_tmpdir, headers, data)
  output_tables_file = tempfile.mktemp(
      prefix='output_tables_file_', dir=test_tmpdir)
  _read_and_store_tables(test_tmpdir, output_tables_file, [table_name])
  return test_tmpdir, output_tables_file, table_name


def _write_dataset(dataset):
  headers = [
      'id', 'annotator', 'position', 'question', 'table_file',
      'answer_coordinates', 'answer_text', 'aggregation', 'float_answer'
  ]
  dataset_df = pd.DataFrame(columns=headers, data=dataset)
  data_path = tempfile.mktemp(prefix='examples_')
  with tf.io.gfile.GFile(data_path, 'w') as f:
    dataset_df.to_csv(f, sep='\t', index=False)
  return data_path


def _write_synthetic_dataset(table_name):
  dataset = ([
      [
          'dev-0', '0', '0', '-', table_name, '["(2, 1)", "(2, 2)"]', '["-"]',
          calc_metrics_utils._Answer.NONE, ''
      ],
      [
          'dev-1', '0', '0', '-', table_name, '["(1, 2)", "(1, 3)", "(1,4)"]',
          '["-"]', calc_metrics_utils._Answer.SUM, ''
      ],
      [
          'dev-2', '0', '0', '-', table_name,
          '["(0, 0)","(0, 1)","(0, 2)","(0, 3)","(0, 4)","(0, 5)"]', '["-"]',
          calc_metrics_utils._Answer.COUNT, ''
      ],
      [
          'dev-3', '0', '0', '-', table_name, '["(0, 4)","(1, 4)"]', '["-"]',
          calc_metrics_utils._Answer.AVERAGE, ''
      ],
      [
          'dev-4', '0', '0', '-', table_name, '["(0, 4)","(1, 4)"]', '["-"]',
          calc_metrics_utils._Answer.SUM, ''
      ],
  ])
  return _write_dataset(dataset)


class CalcMetricsTest(parameterized.TestCase):

  def test_create_tables_dict(self):
    _, output_tables_file, table_name = _write_tables_dict()
    with tf.io.gfile.GFile(output_tables_file, 'rb') as f:
      tables = pickle.load(f)
      assert table_name in tables

  def test_read_data_examples(self):
    data_path = _write_synthetic_dataset('table_1')
    examples = _read_data_examples(data_path)
    self.assertEqual(examples['dev-2-0_0'].gold_cell_coo,
                     set([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]))

  def test_read_predictions(self):
    data_path = _write_synthetic_dataset('table_1')
    examples = _read_data_examples(data_path)
    predictions_path = _write_synthetic_predictions()
    calc_metrics_utils.read_predictions(predictions_path, examples)
    self.assertEqual(examples['dev-2-0_0'].gold_agg_function,
                     calc_metrics_utils._Answer.COUNT)
    self.assertEqual(examples['dev-2-0_0'].pred_agg_function,
                     calc_metrics_utils._Answer.NONE)

  def test_calc_structure_metrics(self):
    data_path = _write_synthetic_dataset('table_1')
    examples = _read_data_examples(data_path)
    predictions_path = _write_synthetic_predictions()
    calc_metrics_utils.read_predictions(predictions_path, examples)
    test_tmpdir = tempfile.mkdtemp()
    results = calc_metrics_utils.calc_structure_metrics(
        examples, denotation_errors_path=test_tmpdir)
    self.assertEqual(results.aggregation_acc, 0.6)
    self.assertEqual(results.cell_acc, 0.6)
    self.assertEqual(results.joint_acc, 0.6)

    denotation_errors = pd.read_csv(
        os.path.join(test_tmpdir, 'structured_examples.tsv'), sep='\t')
    self.assertEqual(denotation_errors.iloc[0, 1], 'dev-0-0_0')
    self.assertEqual(denotation_errors.iloc[0, 2],
                     calc_metrics_utils._Answer.NONE)
    self.assertEqual(denotation_errors.iloc[0, 3],
                     calc_metrics_utils._Answer.NONE)

  def test_denotation_accuracy(self):
    test_tmpdir, output_tables_file, table_name = _write_tables_dict()
    data_path = _write_synthetic_dataset(table_name)
    examples = _read_data_examples(data_path)
    predictions_path = _write_synthetic_predictions()
    calc_metrics_utils.read_predictions(predictions_path, examples)
    predictions_file_name = 'predictions'
    denotation_accuracy = _calc_denotation_accuracy(
        output_tables_file,
        examples,
        denotation_errors_path=test_tmpdir,
        predictions_file_name=predictions_file_name,
    )
    self.assertEqual(denotation_accuracy, 0.8)

    denotation_errors = pd.read_csv(
        os.path.join(test_tmpdir,
                     'denotation_examples_{}'.format(predictions_file_name)),
        sep='\t')
    self.assertEqual(denotation_errors.iloc[0, 1], 'dev-0-0_0')
    self.assertEqual(denotation_errors.iloc[0, 2], '-')
    self.assertEqual(denotation_errors.iloc[0, 5], "['6.13', 'Richmond']")
    self.assertEqual(denotation_errors.iloc[0, 7], '[(2, 1), (2, 2)]')

  def test_calc_denotation_accuracy_handles_nans(self):
    test_tmpdir, output_tables_file, table_name = _write_tables_dict()
    data_path = _write_dataset([[
        'dev-0', '0', '0', '-', table_name, '[]', '[]',
        calc_metrics_utils._Answer.SUM, 'NAN'
    ]])
    examples = _read_data_examples(data_path)
    predictions_path = _write_predictions(
        data=[['dev-0', '0', '0', '[]', '0', '1']])
    calc_metrics_utils.read_predictions(predictions_path, examples)
    denotation_accuracy = _calc_denotation_accuracy(
        output_tables_file,
        examples,
        denotation_errors_path=test_tmpdir,
        predictions_file_name='predictions',
    )
    self.assertEqual(1.0, denotation_accuracy)

  def test_calc_denotation_accuracy_handles_low_precision_floats(self):
    test_tmpdir, output_tables_file, table_name = _write_tables_dict(
        headers=['FLOAT'], data=[['992.39']])
    data_path = _write_dataset([[
        'dev-0',
        '0',
        '0',
        '-',
        table_name,
        '[]',
        '[]',
        calc_metrics_utils._Answer.NONE,
        '992.3900146484375',
    ]])
    examples = _read_data_examples(data_path)
    predictions_path = _write_predictions(
        data=[['dev-0', '0', '0', '["(0, 0)"]', '0', '0']])
    calc_metrics_utils.read_predictions(predictions_path, examples)
    denotation_accuracy = _calc_denotation_accuracy(
        output_tables_file,
        examples,
        denotation_errors_path=test_tmpdir,
        predictions_file_name='predictions',
    )
    self.assertEqual(1.0, denotation_accuracy)

  def test_calc_denotation_accuracy_without_gold_answer(self):
    table = pd.DataFrame([['a', 'b'], ['0', '1']], columns=['A', 'B'])
    denotation_accuracy = calc_metrics_utils.calc_denotation_accuracy(
        examples={
            '0':
                calc_metrics_utils.Example(
                    example_id='0',
                    question='q',
                    table_id='tab_0',
                    table=table,
                    gold_cell_coo=set(),
                    gold_agg_function=calc_metrics_utils._Answer.NONE,
                    float_answer=None,
                    has_gold_answer=False,
                ),
            '1':
                calc_metrics_utils.Example(
                    example_id='1',
                    question='q',
                    table_id='tab_0',
                    table=table,
                    gold_cell_coo={(0, 0)},
                    gold_agg_function=calc_metrics_utils._Answer.NONE,
                    float_answer=None,
                    has_gold_answer=True,
                    pred_cell_coo={(0, 0)},
                )
        },
        denotation_errors_path=None,
        predictions_file_name=None)
    self.assertEqual(0.5, denotation_accuracy)

  def test_read_predictions_without_pred_aggr(self):
    predictions_path = _write_predictions(
        data=[['dev-0', '0', '0', '["(0,0)"]']],
        headers=('id', 'annotator', 'position', 'answer_coordinates'))
    examples = {
        'dev-0-0_0':
            calc_metrics_utils.Example(
                example_id='dev-0-0_0',
                question='q',
                table_id='tab_0',
                table=pd.DataFrame(),
                gold_cell_coo={},
                gold_agg_function=calc_metrics_utils._Answer.NONE,
                float_answer=None,
                has_gold_answer=True,
            )
    }
    calc_metrics_utils.read_predictions(predictions_path, examples)
    self.assertLen(examples, 1)
    self.assertEqual(
        next(iter(examples.values())).pred_agg_function,
        calc_metrics_utils._Answer.NONE)


if __name__ == '__main__':
  absltest.main()
