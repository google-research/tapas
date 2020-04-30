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
# coding=utf8

import string

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
      ("特殊字母Специальные письма", 64),
  )
  def test_str_to_ints(self, text, length):
    ints = text_utils.str_to_ints(text, length)
    self.assertLen(ints, length)
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


if __name__ == "__main__":
  absltest.main()
