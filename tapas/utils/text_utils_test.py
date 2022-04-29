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
# coding=utf8

import string
from typing import Optional, Text
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tapas.utils import constants
from tapas.utils import text_utils


def _get_all_spans_as_text(text, max_ngram_length):
  spans = text_utils.get_all_spans(text, max_ngram_length)
  return [text[span[0]:span[1]] for span in spans]


class TextUtilsTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      ("int", 1, 1.0),
      ("float", 1.0, 1.0),
      ("str_int", "1", 1.0),
      ("str_float", "1.0", 1.0),
      ("example1", "1,000.7", 1000.7),
      ("example2", "1,000", 1000.0),
      ("example3", "5,5556", 5.5556),
      ("example4", "0.0.0.1", 1.0),
      ("example5", "0,0,0,1", 1.0),
  )
  def test_float_conversion(self, value, expected):
    self.assertEqual(expected, text_utils.convert_to_float(value))

  def test_float_conversion_fails(self):
    with self.assertRaises(ValueError):
      text_utils.convert_to_float("hello")

  @parameterized.named_parameters(
      ("simple", " a b c", ["a", "b", "c"]),
      ("irregular_space", " a   b c", ["a", "b", "c"]),
      ("punctuation", " a,   b . c", ["a", ",", "b", ".", "c"]),
  )
  def test_split_strings(self, value, expected):
    self.assertEqual(expected, text_utils.tokenize_string(value))

  @parameterized.named_parameters(
      ("simple", "abc", "abc"),
      ("diacritics", u"perché", u"perche"),
      ("quotes", u"‘quotes’", u"'quotes'"),
      ("dashes", u"“d‒ash”", u"d-ash"),
      ("citations", u"citation‡", u"citation"),
      ("parenthesis", u"abc (123)", u"abc"),
      ("whitespace", u"abc   123", u"abc 123"),
      ("lowercase", u"aBcD", u"abcd"),
  )
  def test_wtq_normalization(self, value, expected):
    self.assertEqual(expected, text_utils.wtq_normalize(value))

  @parameterized.named_parameters(
      ("simple", [1.0, "1"], ["1.0", "1.0"]),
      ("remove_nans", [float("NAN")], []),
      ("sort", ["b", "a"], ["a", "b"]),
      ("lowercase", ["A", "a"], ["a", "a"]),
      ("normalize_floats", [1.0000, "1.0"], ["1.0", "1.0"]),
  )
  def test_normalize_answers(self, value, expected):
    self.assertEqual(expected, text_utils.normalize_answers(value))

  def test_get_all_spans_split_punctuation(self):
    self.assertEqual([
        "a",
        "bc",
        "d",
    ], _get_all_spans_as_text("a,,bc d !", max_ngram_length=1))
    self.assertEqual([
        "a",
        "a,a",
        "a",
        "a  bc",
        "bc",
        "bc d",
        "d",
    ], _get_all_spans_as_text("!a,a  bc d", max_ngram_length=2))

  def test_normalize_for_match_lowercases(self):
    self.assertEqual("lowercase", text_utils.normalize_for_match("LOWERCASE"))

  @parameterized.named_parameters(("na", "N/A"), ("question_mark", "?"),
                                  ("punctuation", " , .;"), ("empty", ""))
  def test_format_and_tokenize_text_recognizes_empty_text(self, text):
    self.assertEqual([constants.EMPTY_TEXT],
                     text_utils.format_and_tokenize_text(text))

  def test_format_and_tokenize_text_strips_punctuation(self):
    self.assertEqual(["a", "b", "7"],
                     text_utils.format_and_tokenize_text(" a, b_;7!!"))

  @parameterized.parameters(
      ("", 5),
      ("ABCfhg", 6),
      ("123456789", 10),
      ("".join(string.punctuation), 32),
      ("特殊字母Специальные письма", 22),
  )
  def test_str_to_ints(self, text, length):
    ints = text_utils.str_to_ints(text, length)
    self.assertLen(ints, length)
    logging.info("ints: %s %s ", ints, text)
    self.assertEqual(text_utils.ints_to_str(ints), text)

  @parameterized.parameters(
      ("", [0]),
      ("ABCfhg", [66, 67, 68, 103, 105, 104]),
      ("123456789", [50, 51, 52, 53, 54, 55, 56, 57, 58, 0]),
      ("".join(string.punctuation), [
          34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 59, 60,
          61, 62, 63, 64, 65, 92, 93, 94, 95, 96, 97, 124, 125, 126, 127
      ]),
      ("特殊字母Специальные письма", [
          29306, 27531, 23384, 27598, 1058, 1088, 1078, 1095, 1081, 1073, 1084,
          1101, 1086, 1100, 1078, 33, 1088, 1081, 1090, 1101, 1085, 1073
      ]),
  )
  def test_ints_to_str(self, text, ints):
    self.assertEqual(text_utils.ints_to_str(ints), text)

  @parameterized.parameters(
      ("ABC-//__ABC-__"),
      ("",),
      ("ABCfhg",),
      ("123456789",),
      ("".join(string.punctuation),),
      ("特殊字母Специальные письма",),
  )
  def test_get_question_id(self, example_id):
    sequence_id = text_utils.get_sequence_id(example_id, "0")
    question_id = text_utils.get_question_id(sequence_id, 0)
    new_example_id, annotator, position = text_utils.parse_question_id(
        question_id)
    self.assertEqual(new_example_id, example_id)
    self.assertEqual(annotator, "0")
    self.assertEqual(position, 0)

  def test_find_all_substrings(self):
    needle = "copy"
    haystack = "copy some copy all copy."
    indexes = text_utils.find_all_substrings(needle, haystack)
    self.assertEqual(indexes, [(0, 4), (10, 14), (19, 23)])
    for start, end in indexes:
      self.assertEqual(needle, haystack[start:end])

  def test_filter_invalid_unicode(self):
    invalid_unicode = bytes("gold medaMedal\x80\xA6I", "utf-8")
    valid_text, is_invalid = text_utils.filter_invalid_unicode(invalid_unicode)
    self.assertTrue(is_invalid)
    self.assertEqual(valid_text, "")

    text = "gold medaMedal"
    valid_text, is_invalid = text_utils.filter_invalid_unicode(text)
    self.assertFalse(is_invalid)
    self.assertEqual(valid_text, text)

  @parameterized.named_parameters(
      ("interaction_id_with_slash", "1234/0-1", "1234"),
      ("interaction_id_without_slash", "123df-1", "123df"),
      ("question_id_with_slash_", "1234-abc/12-0_0", "1234-abc"),
      ("question_id_without_slash", "123df-1_0", "123df"),
      ("example_id", "1da3f", "1da3f"),
  )
  def test_get_example_id(self, input_text, example_id):
    self.assertEqual(text_utils.get_example_id(input_text), example_id)

  @parameterized.named_parameters(
      ("no_ith_table_1", "1234", None, 0, "1234-0"),
      ("no_ith_table_2", "4d3f", None, 3, "4d3f-3"),
      ("ith_table_1", "balloon", 0, 0, "balloon/0-0"),
      ("ith_table_2", "l33tc0d3", 12, 1, "l33tc0d3/12-1"),
  )
  def test_interaction_id_creation(self, example_id,
                                   ith_table, suffix,
                                   interaction_id):
    self.assertEqual(
        text_utils.create_interaction_id(example_id, ith_table, suffix),
        interaction_id)

  @parameterized.named_parameters(
      ("numeric_1", "1234", "1234-0"),
      ("alphabetic", "4d3f", "4d3f-0"),
      ("alphanumeric", "d13e8", "d13e8-0"),
  )
  def test_default_creation(self, example_id, interaction_id):
    self.assertEqual(
        text_utils.create_interaction_id(example_id), interaction_id)


if __name__ == "__main__":
  absltest.main()
