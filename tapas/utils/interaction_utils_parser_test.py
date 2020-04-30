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
from tapas.utils import interaction_utils_parser
from google.protobuf import text_format


_Mode = interaction_utils_parser.SupervisionMode


def _set_float32_safe_interaction(interaction):
  new_interaction = interaction_pb2.Interaction()
  new_interaction.ParseFromString(interaction.SerializeToString())
  interaction.CopyFrom(new_interaction)


def _set_float32_safe_answer(answer):
  new_answer = interaction_pb2.Answer()
  new_answer.ParseFromString(answer.SerializeToString())
  answer.CopyFrom(new_answer)


class InteractionUtilsParserTest(parameterized.TestCase):

  def test_unambiguous_matching(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "ab" } }
        rows { cells { text: "b" } }
        rows { cells { text: "bc" } }
      }
      questions {
        answer {
          answer_texts: "a"
          answer_texts: "b"
        }
      }""", interaction_pb2.Interaction())

    question = interaction_utils_parser.parse_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)

    expected_answer = text_format.Parse(
        """
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_coordinates {
            row_index: 2
            column_index: 0
          }
          answer_texts: "a"
          answer_texts: "b"
    """, interaction_pb2.Answer())

    self.assertEqual(expected_answer, question.answer)

  def test_ambiguous_matching(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "a" } }
      }
      questions {
        answer {
          answer_texts: "a"
          answer_texts: "a"
        }
      }""", interaction_pb2.Interaction())

    with self.assertRaises(ValueError):
      interaction_utils_parser.parse_question(interaction.table,
                                              interaction.questions[0],
                                              _Mode.REMOVE_ALL)

  def test_float_value(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "a" } }
      }
      questions {
        answer {
          answer_texts: "1.0"
        }
      }""", interaction_pb2.Interaction())

    question = interaction_utils_parser.parse_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)

    expected_answer = text_format.Parse(
        """
      answer_texts: "1.0"
      float_value: 1.0
    """, interaction_pb2.Answer())

    self.assertEqual(expected_answer, question.answer)

  @parameterized.named_parameters(
      ("no_filter", _Mode.NONE, """
        answer_coordinates {
          row_index: 0
          column_index: 0
        }
        answer_coordinates {
          row_index: 1
          column_index: 0
        }
        answer_texts: "2"
        aggregation_function: COUNT"""),
      ("remove_all", _Mode.REMOVE_ALL, """
        answer_texts: "2"
        float_value: 2.0"""),
  )
  def test_strategies(self, mode, expected_answer):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "b" } }
      }
      questions {
        answer {
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_coordinates {
            row_index: 1
            column_index: 0
          }
          answer_texts: "2"
          aggregation_function: COUNT
        }
      }""", interaction_pb2.Interaction())

    question = interaction_utils_parser.parse_question(interaction.table,
                                                       interaction.questions[0],
                                                       mode)
    self.assertEqual(
        text_format.Parse(expected_answer, interaction_pb2.Answer()),
        question.answer)

  def test_set_answer_text_when_multiple_answers(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "2008" } }
      }
      questions {
        answer {
          answer_texts: "1"
          answer_texts: "2"
          float_value: 2008.0
        }
      }""", interaction_pb2.Interaction())

    question = interaction_utils_parser.parse_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)

    expected_answer = text_format.Parse(
        """
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_texts: "2008"
          float_value: 2008.0
    """, interaction_pb2.Answer())

    self.assertEqual(expected_answer, question.answer)

  def test_set_answer_text_strange_float_format_when_multiple_answers(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "2008" } }
      }
      questions {
        answer {
          answer_texts: "1"
          answer_texts: "2"
          float_value: 2008.001
        }
      }""", interaction_pb2.Interaction())
    _set_float32_safe_interaction(interaction)
    question = interaction_utils_parser.parse_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)
    _set_float32_safe_interaction(interaction)
    expected_answer = text_format.Parse(
        """
          answer_texts: "2008.0009765625"
          float_value: 2008.001
    """, interaction_pb2.Answer())
    _set_float32_safe_answer(expected_answer)
    self.assertEqual(expected_answer, question.answer)

  def test_set_use_answer_text_when_single_float_answer(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "2008.00000000000" } }
      }
      questions {
        answer {
          answer_texts: "2008.00000000000"
          float_value: 2008.0
        }
      }""", interaction_pb2.Interaction())

    question = interaction_utils_parser.parse_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)
    _set_float32_safe_interaction(interaction)
    expected_answer = text_format.Parse(
        """
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_texts: "2008.00000000000"
          float_value: 2008.0
    """, interaction_pb2.Answer())
    _set_float32_safe_answer(expected_answer)
    self.assertEqual(expected_answer, question.answer)


if __name__ == "__main__":
  absltest.main()
