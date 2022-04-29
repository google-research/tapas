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

import math
import os
import tempfile

from absl import flags
from absl import logging
from absl.testing import absltest
from tapas.protos import interaction_pb2
from tapas.protos import negative_retrieval_examples_pb2
from tapas.retrieval import tf_example_utils
from tapas.utils import number_annotation_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS
TEST_PATH = 'tapas/retrieval/testdata'

_RESERVED_SYMBOLS = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EMPTY]')

_NegativeRetrievalExample = negative_retrieval_examples_pb2.NegativeRetrievalExample


def _get_int_feature(example, key):
  return list(example.features.feature[key].int64_list.value)


def _get_byte_feature(example, key):
  return list(example.features.feature[key].bytes_list.value)


def _clean_nans(array):
  return ['nan' if math.isnan(x) else x for x in array]


def _get_float_feature(example, key):
  return _clean_nans(example.features.feature[key].float_list.value)


def _create_vocab(vocab_file, tokens):
  with open(vocab_file, 'tw') as output_file:
    for word in _RESERVED_SYMBOLS:
      output_file.write(f'{word}\n')
    for token in tokens:
      output_file.write(f'{token}\n')


class TfExampleUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = TEST_PATH

  def test_get_empty_example(self):
    max_seq_length = 15

    input_path = os.path.join(self.test_data_dir, 'retrieval_interaction.pbtxt')
    with open(input_path) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())
    number_annotation_utils.add_numeric_values(interaction)

    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, [
          'by', 'created', 'do', 'dragon', 'go', 'hannibal', 'harris', 'in',
          'lecter', 'movies', 'novels', 'order', 'original', 'red', 'the',
          'thomas', 'what', 'work'
      ])
      converter = tf_example_utils.ToRetrievalTensorflowExample(
          config=tf_example_utils.RetrievalConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
          ))
      example = converter.convert(interaction, index=0, negative_example=None)
      logging.info(example)
      # Check the question.
      self.assertEqual(
          _get_int_feature(example, 'question_input_ids'),
          [2, 22, 17, 8, 20, 11, 14, 15, 10, 13, 3, 0, 0, 0, 0])
      self.assertEqual(
          _get_int_feature(example, 'question_input_mask'),
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
      # Check document title + table.
      self.assertEqual(
          _get_int_feature(example, 'input_ids'),
          [2, 11, 14, 3, 7, 6, 18, 23, 16, 21, 12, 19, 9, 19, 9])
      self.assertEqual(
          _get_int_feature(example, 'input_mask'),
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      self.assertEqual(
          _get_int_feature(example, 'segment_ids'),
          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

  def test_convert_with_negative_tables(self):
    max_seq_length = 12
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, range(10))
      converter = tf_example_utils.ToRetrievalTensorflowExample(
          config=tf_example_utils.RetrievalConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
          ))
      interaction = interaction_pb2.Interaction(
          table=interaction_pb2.Table(
              columns=[
                  interaction_pb2.Cell(text='A'),
                  interaction_pb2.Cell(text='B'),
                  interaction_pb2.Cell(text='C'),
              ],
              rows=[
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='0 6'),
                      interaction_pb2.Cell(text='4 7'),
                      interaction_pb2.Cell(text='5 6'),
                  ]),
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='1 7'),
                      interaction_pb2.Cell(text='3 6'),
                      interaction_pb2.Cell(text='5 5'),
                  ]),
              ],
              table_id='table_0',
          ),
          questions=[interaction_pb2.Question(
              id='id',
              original_text='2',
          )],
      )
      number_annotation_utils.add_numeric_values(interaction)
      n_table = interaction_pb2.Table(
          columns=[
              interaction_pb2.Cell(text='A'),
              interaction_pb2.Cell(text='B'),
          ],
          rows=[
              interaction_pb2.Cells(cells=[
                  interaction_pb2.Cell(text='0 6'),
                  interaction_pb2.Cell(text='4 7'),
              ]),
              interaction_pb2.Cells(cells=[
                  interaction_pb2.Cell(text='1 7'),
                  interaction_pb2.Cell(text='3 6'),
              ]),
          ],
          table_id='table_1',
      )
      number_annotation_utils.add_numeric_table_values(n_table)
      n_example = _NegativeRetrievalExample()
      n_example.table.CopyFrom(n_table)
      n_example.score = -82.0
      n_example.rank = 1
      example = converter.convert(interaction, 0, n_example)
      logging.info(example)
      self.assertEqual(
          _get_int_feature(example, 'input_ids'), [
              2, 5, 3, 1, 1, 1, 6, 10, 11, 7, 9, 11, 2, 5, 3, 1, 1, 6, 10, 7, 9,
              0, 0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'row_ids'), [
              0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'column_ids'), [
              0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 1, 2, 1, 2, 1, 2, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'segment_ids'), [
              0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'input_mask'), [
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'inv_column_ranks'), [
              0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'column_ranks'), [
              0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'numeric_relations'), [
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0
          ])
      self.assertEqual(
          _get_int_feature(example, 'table_id_hash'), [911224864, 1294380046])
      self.assertEqual(
          _get_float_feature(example, 'numeric_values'), [
              'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 0.0, 4.0, 5.0, 1.0, 3.0,
              5.0, 'nan', 'nan', 'nan', 'nan', 'nan', 0.0, 4.0, 1.0, 3.0, 'nan',
              'nan', 'nan'
          ])
      self.assertEqual(
          _get_float_feature(example, 'numeric_values_scale'), [
              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
          ])
      self.assertEqual(
          [i.decode('utf-8') for i in _get_byte_feature(example, 'table_id')],
          ['table_0', 'table_1'])

if __name__ == '__main__':
  absltest.main()
