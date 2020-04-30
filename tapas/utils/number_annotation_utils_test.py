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

from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from google.protobuf import text_format


def _number(float_value):
  nv = interaction_pb2.NumericValue()
  nv.float_value = float_value
  return nv


def _date(year=None, month=None, day=None):
  nv = interaction_pb2.NumericValue()
  if year:
    nv.date.year = year
  if month:
    nv.date.month = month
  if day:
    nv.date.day = day
  return nv


class AddNumericTableValuesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', [], [], 0.7),
      # If we remove one of the numbers from the 'Number' column we expect
      # no values to be returned for that column.
      ('below_threshold', [(0, 1, '')], [(1, {})], 0.7),
      # Same as above but we also lower the threshold so we expect results.
      ('above_threshold', [(0, 1, '')], [(1, {
          1: _number(2),
          2: _number(3)
      })], 0.5),
      # We change two number values so that they also pass as dates but still
      # expect numbers as a result since that is the most common type.
      ('ambiguous_values', [(1, 1, '2000'), (2, 1, '2001')], [(1, {
          0: _number(1),
          1: _number(2000),
          2: _number(2001)
      })], 0.7),
      # Same as above but we also remove the first value, now the output should
      # be dates since there are two dates and two numbers and we prefer dates.
      ('ambiguous_date_values', [(0, 1, ''), (1, 1, '2000'),
                                 (2, 1, '2001')], [(1, {
                                     1: _date(year=2000),
                                     2: _date(year=2001)
                                 })], 0.5))
  def test_table_values(self, row_updates, expected_updates,
                        min_consolidation_fraction):

    expected_table = text_format.Parse(
        """
          columns {
            text: 'Name'
          }
          columns {
            text: 'Number'
          }
          columns {
            text: 'Date'
          }
          rows {
            cells {
              text: 'A'
            }
            cells {
              text: '1'
            }
            cells {
              text: 'August 2014'
            }
          }
          rows {
            cells {
              text: 'B'
            }
            cells {
              text: '2'
            }
            cells {
              text: 'July 7'
            }
          }
          rows {
            cells {
              text: 'C'
            }
            cells {
              text: '3'
            }
            cells {
              text: 'March 17, 2015'
            }
          }
    """, interaction_pb2.Table())

    for row_index, col_index, text in row_updates:
      expected_table.rows[row_index].cells[col_index].text = text

    actual_table = interaction_pb2.Table()
    actual_table.CopyFrom(expected_table)
    number_annotation_utils.add_numeric_table_values(
        actual_table, min_consolidation_fraction=min_consolidation_fraction)

    expected_table.rows[0].cells[1].numeric_value.CopyFrom(_number(1))
    expected_table.rows[1].cells[1].numeric_value.CopyFrom(_number(2))
    expected_table.rows[2].cells[1].numeric_value.CopyFrom(_number(3))
    expected_table.rows[0].cells[2].numeric_value.CopyFrom(
        _date(year=2014, month=8))
    expected_table.rows[1].cells[2].numeric_value.CopyFrom(
        _date(month=7, day=7))
    expected_table.rows[2].cells[2].numeric_value.CopyFrom(
        _date(year=2015, month=3, day=17))
    for col_index, new_dict in expected_updates:
      for row_index in range(len(expected_table.rows)):
        expected_table.rows[row_index].cells[col_index].ClearField(
            'numeric_value')
        if row_index in new_dict:
          expected_table.rows[row_index].cells[
              col_index].numeric_value.CopyFrom(new_dict[row_index])

    self.assertEqual(expected_table, actual_table)


class AddNumericValuesTest(parameterized.TestCase):

  def test_add_numeric_values_to_questions(self):
    actual_interaction = text_format.Parse(
        """
          questions {
            original_text: 'What are all the buildings in canada?'
          }
          questions {
            original_text: 'Which building has more than 17 floors?'
          }
          questions {
            original_text:
              'Are there one or two buildings build on March 17, 2015?'
          }""", interaction_pb2.Interaction())
    number_annotation_utils.add_numeric_values_to_questions(actual_interaction)

    expected_interaction = text_format.Parse(
        """
          questions {
            original_text: 'What are all the buildings in canada?'
            text: 'what are all the buildings in canada?'
            annotations {
            }
          }
          questions {
            original_text: 'Which building has more than 17 floors?'
            text: 'which building has more than 17 floors?'
            annotations {
             spans {
               begin_index: 29
               end_index: 31
               values {
                 float_value: 17.0
               }
             }
            }
          }
          questions {
            original_text:
              'Are there one or two buildings build on March 17, 2015?'
            text: 'are there one or two buildings build on march 17, 2015?'
            annotations {
             spans {
               begin_index: 10
               end_index: 13
               values {
                 float_value: 1.0
               }
             }
             spans {
               begin_index: 17
               end_index: 20
               values {
                 float_value: 2.0
               }
             }
             spans {
               begin_index: 40
               end_index: 54
               values {
                 date {
                   year: 2015
                   month: 3
                   day: 17
                 }
               }
             }
            }
          }""", interaction_pb2.Interaction())

    self.assertEqual(expected_interaction, actual_interaction)


if __name__ == '__main__':
  absltest.main()
