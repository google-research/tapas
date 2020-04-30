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
from tapas.utils import number_utils
from google.protobuf import text_format


def _get_spans(text_proto):
  return list(
      text_format.Parse(text_proto, interaction_pb2.NumericValueSpans()).spans)


class NumberUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(('float', '1000.0'), ('ordinal', '1000th'),
                                  ('comma', '1,000'), ('simple', '1000'))
  def test_parse_simple_numbers(self, text):
    span = interaction_pb2.NumericValueSpan(
        begin_index=0,
        end_index=len(text),
        values=[interaction_pb2.NumericValue(float_value=1000)])
    self.assertEqual([span], number_utils.parse_text(text))

  @parameterized.named_parameters(('negative_notrailing_zero', '-.3'),
                                  ('positive_notrailing_zero', '.3'),
                                  ('negative', '-0.63'), ('positive', '0.63'))
  def test_parse_simple_floats(self, text):
    span = interaction_pb2.NumericValueSpan(
        begin_index=0,
        end_index=len(text),
        values=[interaction_pb2.NumericValue(float_value=float(text))])
    self.assertEqual([span], number_utils.parse_text(text))

  def test_dont_parse_weird_examples(self):
    for text in ('1....', '1,,,,,,', '10000,..', '-.1,,,,00,20'):
      spans = number_utils.parse_text(text)
      self.assertNotEmpty(spans)
      for span in spans:
        # Make sure that we don't parse the entire text as a number.
        self.assertNotEqual((span.begin_index, span.end_index), (0, len(text)),
                            text)

  def test_parse_range(self):
    text = '2005-2010'
    expected_spans = """
      spans {
        begin_index: 0
        end_index: 4
        values {
          float_value: 2005.
        }
        values {
          date {
            year: 2005
          }
        }
      }
      spans {
        begin_index: 5
        end_index: 9
        values {
          float_value: 2010.
        }
        values {
          date {
            year: 2010
          }
        }
      }
      """
    self.assertEqual(_get_spans(expected_spans), number_utils.parse_text(text))

  @parameterized.named_parameters(('cardinal', 'twelve'),
                                  ('ordinal', 'twelfth'))
  def test_parse_numerals(self, text):
    span = interaction_pb2.NumericValueSpan(
        begin_index=0,
        end_index=len(text),
        values=[interaction_pb2.NumericValue(float_value=12)])
    self.assertEqual([span], number_utils.parse_text(text))

  @parameterized.named_parameters(('month_day', 'August 12', 8, 12),
                                  ('weekday_month_day', 'Monday, Sep 7', 9, 7),
                                  ('day_month', '5 Nov', 11, 5))
  def test_parse_dates(self, text, month, day):
    span = interaction_pb2.NumericValueSpan(
        begin_index=0,
        end_index=len(text),
        values=[
            interaction_pb2.NumericValue(
                date=interaction_pb2.Date(month=month, day=day))
        ])
    self.assertEqual([span], number_utils.parse_text(text))

  @parameterized.named_parameters(
      ('dots', '19.10.1912', 19, 10, 1912),
      ('month_name', 'November 14, 2005', 14, 11, 2005),
      ('dashes', '11-14-2005', 14, 11, 2005),
      ('spaces', 'january 28,   1933', 28, 1, 1933))
  def test_parse_complete_dates(self, text, day, month, year):
    span = interaction_pb2.NumericValueSpan(
        begin_index=0,
        end_index=len(text),
        values=[
            interaction_pb2.NumericValue(
                date=interaction_pb2.Date(year=year, month=month, day=day))
        ])
    self.assertEqual([span], number_utils.parse_text(text))

  def test_parse_text(self):
    expected_spans = """
      spans {
        begin_index: 0
        end_index: 9
        values {
          float_value: 1000000.0
        }
      }
      spans {
        begin_index: 12
        end_index: 18
        values {
          float_value: 10000.0
        }
      }
      spans {
        begin_index: 19
        end_index: 32
        values {
          date {
            year: 1846
            month: 11
          }
        }
      }
     spans {
        begin_index: 36
        end_index: 49
        values {
          date {
            year: 1847
            month: 2
          }
        }
    }
    spans {
      begin_index: 53
      end_index: 57
      values {
        float_value: 1908.0
      }
      values {
        date {
          year: 1908
        }
      }
    }"""
    self.assertEqual(
        _get_spans(expected_spans),
        number_utils.parse_text(
            '1,000,000, $10,000 November 1846 to February 1847 in 1908'))

  def test_ignore_nans(self):
    self.assertEmpty(number_utils.parse_text('inf'))
    self.assertEmpty(number_utils.parse_text('nan'))
    self.assertEmpty(number_utils.parse_text('Nan Hayworth'))

  def test_is_ordinal(self):
    self.assertTrue(number_utils.is_ordinal('first'))
    self.assertTrue(number_utils.is_ordinal('1st'))
    self.assertTrue(number_utils.is_ordinal('second'))
    self.assertTrue(number_utils.is_ordinal('2nd'))
    self.assertTrue(number_utils.is_ordinal('third'))
    self.assertTrue(number_utils.is_ordinal('3rd'))
    self.assertTrue(number_utils.is_ordinal('42th'))
    self.assertFalse(number_utils.is_ordinal('one'))
    self.assertFalse(number_utils.is_ordinal('1'))
    self.assertFalse(number_utils.is_ordinal('two'))
    self.assertFalse(number_utils.is_ordinal('2'))
    self.assertFalse(number_utils.is_ordinal('three'))
    self.assertFalse(number_utils.is_ordinal('3'))
    self.assertFalse(number_utils.is_ordinal('hearth'))


if __name__ == '__main__':
  absltest.main()
