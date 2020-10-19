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

import math
import os
import tempfile

from absl import logging
from absl.testing import absltest
from tapas.protos import interaction_pb2
from tapas.protos import table_selection_pb2
from tapas.utils import number_annotation_utils
from tapas.utils import text_utils
from tapas.utils import tf_example_utils

_RESERVED_SYMBOLS = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EMPTY]')
_NAN = float('nan')


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

  def test_get_empty_example(self):
    max_seq_length = 3

    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, [])
      converter = tf_example_utils.ToClassifierTensorflowExample(
          config=tf_example_utils.ClassifierConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
              add_aggregation_candidates=False,
          ))
      example = converter.get_empty_example()
      logging.info(example)
      question_id = _get_byte_feature(example, 'question_id')[0].decode('utf-8')
      self.assertEqual(question_id, text_utils.get_padded_question_id())

  def test_convert(self):
    max_seq_length = 12
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, range(10))
      converter = tf_example_utils.ToClassifierTensorflowExample(
          config=tf_example_utils.ClassifierConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
              add_aggregation_candidates=False,
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
                      interaction_pb2.Cell(text='0'),
                      interaction_pb2.Cell(text='4'),
                      interaction_pb2.Cell(text='5'),
                  ]),
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='1'),
                      interaction_pb2.Cell(text='3'),
                      interaction_pb2.Cell(text='5'),
                  ]),
              ],
          ),
          questions=[interaction_pb2.Question(id='id', original_text='2')],
      )
      number_annotation_utils.add_numeric_values(interaction)
      example = converter.convert(interaction, 0)
      logging.info(example)
      self.assertEqual(
          _get_int_feature(example, 'input_ids'),
          [2, 8, 3, 1, 1, 1, 6, 10, 11, 7, 9, 11])
      self.assertEqual(
          _get_int_feature(example, 'row_ids'),
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
      self.assertEqual(
          _get_int_feature(example, 'column_ids'),
          [0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3])
      self.assertEqual(
          _get_int_feature(example, 'column_ranks'),
          [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1])
      self.assertEqual(
          _get_int_feature(example, 'numeric_relations'),
          [0, 0, 0, 0, 0, 0, 4, 2, 2, 4, 2, 2])

  def test_convert_with_trimmed_cell(self):
    max_seq_length = 16
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, range(10))
      converter = tf_example_utils.ToClassifierTensorflowExample(
          config=tf_example_utils.ClassifierConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
              add_aggregation_candidates=False,
              cell_trim_length=2,
              drop_rows_to_fit=True))
      interaction = interaction_pb2.Interaction(
          table=interaction_pb2.Table(
              columns=[
                  interaction_pb2.Cell(text='A'),
                  interaction_pb2.Cell(text='A A'),
                  interaction_pb2.Cell(text='A A A A'),
              ],
              rows=[
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='A A A'),
                      interaction_pb2.Cell(text='A A A'),
                      interaction_pb2.Cell(text='A A A'),
                  ]),
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='A A A'),
                      interaction_pb2.Cell(text='A A A'),
                      interaction_pb2.Cell(text='A A A'),
                  ]),
              ],
          ),
          questions=[interaction_pb2.Question(id='id', original_text='A')],
      )
      number_annotation_utils.add_numeric_values(interaction)
      example = converter.convert(interaction, 0)
      logging.info(example)
      # We expect the second row to be dropped all cells should be trimmed to
      # >= 2 tokens.
      self.assertEqual(
          _get_int_feature(example, 'column_ids'),
          [0, 0, 0, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 0, 0])

  def test_convert_with_document_title_and_answer_text(self):
    max_seq_length = 15
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, ['a', 'b', 'c', 'd', 'e'])
      converter = tf_example_utils.ToClassifierTensorflowExample(
          config=tf_example_utils.ClassifierConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
              add_aggregation_candidates=False,
              use_document_title=True,
              update_answer_coordinates=True,
          ))
      interaction = interaction_pb2.Interaction(
          table=interaction_pb2.Table(
              document_title='E E',
              columns=[
                  interaction_pb2.Cell(text='A'),
                  interaction_pb2.Cell(text='A B C'),
              ],
              rows=[
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='A B'),
                      interaction_pb2.Cell(text='A B C'),
                  ]),
              ],
          ),
          questions=[
              interaction_pb2.Question(
                  id='id',
                  original_text='D',
                  answer=interaction_pb2.Answer(answer_texts=['B C']),
              )
          ],
      )
      example = converter.convert(interaction, 0)
      logging.info(example)
      self.assertEqual(
          _get_int_feature(example, 'input_ids'),
          [2, 5, 3, 10, 10, 3, 6, 6, 7, 8, 6, 7, 6, 7, 8])
      self.assertEqual(
          _get_int_feature(example, 'label_ids'),
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

  def test_convert_with_invalid_unicode(self):
    max_seq_length = 12
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, range(10))
      converter = tf_example_utils.ToClassifierTensorflowExample(
          config=tf_example_utils.ClassifierConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
              add_aggregation_candidates=False,
          ))
      interaction = interaction_pb2.Interaction(
          table=interaction_pb2.Table(
              columns=[
                  interaction_pb2.Cell(text='A\x80\xA6I'),
                  interaction_pb2.Cell(text='B\x80\xA6I'),
                  interaction_pb2.Cell(text='C\x80\xA6I'),
              ],
              rows=[
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='0'),
                      interaction_pb2.Cell(text='4'),
                      interaction_pb2.Cell(text='5'),
                  ]),
                  interaction_pb2.Cells(cells=[
                      interaction_pb2.Cell(text='1'),
                      interaction_pb2.Cell(text='3'),
                      interaction_pb2.Cell(text='5'),
                  ]),
              ],
          ),
          questions=[interaction_pb2.Question(id='id', original_text='2')],
      )
      number_annotation_utils.add_numeric_values(interaction)
      example = converter.convert(interaction, 0)
      logging.info(example)
      self.assertEqual(
          _get_int_feature(example, 'input_ids'),
          [2, 8, 3, 1, 1, 1, 6, 10, 11, 7, 9, 11])
      self.assertEqual(
          _get_int_feature(example, 'row_ids'),
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
      self.assertEqual(
          _get_int_feature(example, 'column_ids'),
          [0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3])
      self.assertEqual(
          _get_int_feature(example, 'column_ranks'),
          [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1])
      self.assertEqual(
          _get_int_feature(example, 'numeric_relations'),
          [0, 0, 0, 0, 0, 0, 4, 2, 2, 4, 2, 2])

  def test_convert_with_token_selection(self):
    max_seq_length = 12
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, 'vocab.txt')
      _create_vocab(vocab_file, range(10))
      converter = tf_example_utils.ToClassifierTensorflowExample(
          config=tf_example_utils.ClassifierConversionConfig(
              vocab_file=vocab_file,
              max_seq_length=max_seq_length,
              max_column_id=max_seq_length,
              max_row_id=max_seq_length,
              strip_column_names=False,
              add_aggregation_candidates=False,
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
          ),
          questions=[interaction_pb2.Question(id='id', original_text='2')],
      )
      table_coordinates = []
      for r, c, t in [(0, 0, 0), (1, 0, 0), (1, 2, 0), (2, 0, 0), (2, 2, 0),
                      (2, 2, 1)]:
        table_coordinates.append(
            table_selection_pb2.TableSelection.TokenCoordinates(
                row_index=r, column_index=c, token_index=t))
      interaction.questions[0].Extensions[
          table_selection_pb2.TableSelection.table_selection_ext].CopyFrom(
              table_selection_pb2.TableSelection(
                  selected_tokens=table_coordinates))

      number_annotation_utils.add_numeric_values(interaction)
      example = converter.convert(interaction, 0)
      logging.info(example)
      self.assertEqual(
          _get_int_feature(example, 'input_ids'),
          [2, 8, 3, 1, 6, 11, 7, 11, 11, 0, 0, 0])
      self.assertEqual(
          _get_int_feature(example, 'row_ids'),
          [0, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 0])
      self.assertEqual(
          _get_int_feature(example, 'column_ids'),
          [0, 0, 0, 1, 1, 3, 1, 3, 3, 0, 0, 0])
      self.assertEqual(
          _get_int_feature(example, 'column_ranks'),
          [0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0])
      self.assertEqual(
          _get_int_feature(example, 'numeric_relations'),
          [0, 0, 0, 0, 4, 2, 4, 2, 2, 0, 0, 0])

if __name__ == '__main__':
  absltest.main()
