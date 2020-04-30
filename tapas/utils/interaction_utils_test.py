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

import os

from absl import flags
from absl.testing import absltest
from tapas.protos import interaction_pb2
from tapas.utils import interaction_utils
from google.protobuf import text_format


FLAGS = flags.FLAGS
TEST_PATH = 'tapas/utils/testdata/'


class InteractionUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = TEST_PATH

  def test_simple_questions(self):
    with open(os.path.join(self.test_data_dir, 'questions.tsv'),
              'r') as file_handle:
      interactions = interaction_utils.read_from_tsv_file(file_handle)
    self.assertLen(interactions, 2)
    self.assertEqual(
        text_format.Parse(
            """
      id: "nt-14053-1"
      table {
        table_id: "table_csv/203_386.csv"
      }
      questions {
        id: "nt-14053-1_0"
        original_text: "who were the captains?"
        answer {
          answer_coordinates {
            row_index: 0
            column_index: 3
          }
          answer_coordinates {
            row_index: 1
            column_index: 3
          }
          answer_texts: "Heinrich Brodda"
          answer_texts: "Oskar Staudinger"
        }
      }
      questions {
        id: "nt-14053-1_1"
        original_text: "which ones lost their u-boat on may 5?"
        answer {
          answer_coordinates {
            row_index: 1
            column_index: 3
          }
          answer_coordinates {
            row_index: 2
            column_index: 3
          }
          answer_texts: "Oskar Staudinger"
          answer_texts: "Herbert Neckel"
        }
      }
      questions {
        id: "nt-14053-1_2"
        original_text: "of those, which one is not oskar staudinger?"
        answer {
          answer_coordinates {
            row_index: 2
            column_index: 3
          }
          answer_texts: "Herbert Neckel"
        }
      }
    """, interaction_pb2.Interaction()), interactions[0])
    self.assertEqual(
        text_format.Parse(
            """
      id: "nt-5431-0"
      table {
        table_id: "table_csv/204_703.csv"
      }
      questions {
        id: "nt-5431-0_0"
        original_text: "what are all the countries?"
        answer {
          answer_coordinates {
            row_index: 0
            column_index: 1
          }
          answer_coordinates {
            row_index: 1
            column_index: 1
          }
          answer_texts: "Canada (CAN)"
          answer_texts: "Russia (RUS)"
        }
      }
    """, interaction_pb2.Interaction()), interactions[1])

  def test_questions_with_aggregation(self):
    """Tests that the most important function names can be parsed."""
    with open(
        os.path.join(self.test_data_dir, 'questions_aggregation.tsv'),
        'r') as file_handle:
      interactions = interaction_utils.read_from_tsv_file(file_handle)
    self.assertLen(interactions, 2)
    self.assertEqual(
        text_format.Parse(
            """
      id: "nt-14053-1"
      table {
        table_id: "table_csv/203_386.csv"
      }
      questions {
        id: "nt-14053-1_0"
        original_text: "who were the captains?"
        answer {
          answer_coordinates {
            row_index: 0
            column_index: 3
          }
          answer_coordinates {
            row_index: 1
            column_index: 3
          }
          answer_texts: "Heinrich Brodda"
          answer_texts: "Oskar Staudinger"
        }
      }
      questions {
        id: "nt-14053-1_1"
        original_text: "which ones lost their u-boat on may 5?"
        answer {
          answer_coordinates {
            row_index: 1
            column_index: 3
          }
          answer_texts: "Oskar Staudinger"
          aggregation_function: NONE
        }
      }
      questions {
        id: "nt-14053-1_2"
        original_text: "of those, which one is not oskar staudinger?"
        answer {
          answer_coordinates {
            row_index: 2
            column_index: 3
          }
          answer_texts: "Herbert Neckel"
          aggregation_function: NONE
        }
      }
    """, interaction_pb2.Interaction()), interactions[0])
    self.assertEqual(
        text_format.Parse(
            """
      id: "nt-4436-0"
      table {
        table_id: "table_csv/203_88.csv"
      }
      questions {
        id: "nt-4436-0_0"
        original_text: "which language has more males then females?"
        answer {
          answer_coordinates {
            row_index: 2
            column_index: 0
          }
          aggregation_function: SUM
          answer_texts: "Russian"
        }
      }
      questions {
        id: "nt-4436-0_1"
        original_text: "which of those have less than 500 males?"
        answer {
          answer_coordinates {
            row_index: 5
            column_index: 0
          }
          aggregation_function: COUNT
          answer_texts: "Romanian"
        }
      }
      questions {
        id: "nt-4436-0_2"
        original_text: "the ones have less than 20 females?"
        answer {
          answer_coordinates {
            row_index: 5
            column_index: 0
          }
          answer_coordinates {
            row_index: 7
            column_index: 0
          }
          answer_texts: "Romanian"
          answer_texts: "Estonian"
          aggregation_function: AVERAGE
        }
      }
    """, interaction_pb2.Interaction()), interactions[1])

  def test_questions_with_float_answer(self):
    """Tests that the most important function names can be parsed."""
    with open(
        os.path.join(self.test_data_dir, 'questions_float_answer.tsv'),
        'r') as file_handle:
      interactions = interaction_utils.read_from_tsv_file(file_handle)
    float_answers = []
    for interaction in interactions:
      float_answers.append([
          question.answer.float_value
          if question.answer.HasField('float_value') else None
          for question in interaction.questions
      ])
    self.assertSequenceEqual(float_answers,
                             [[None, None, None], [150.0, 7.0, 7.5]])


if __name__ == '__main__':
  absltest.main()
