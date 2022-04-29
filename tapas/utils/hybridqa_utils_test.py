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
"""Tests for tapas.utils.google.hybridqa_utils."""

import collections
import json
import tempfile

from absl.testing import absltest
from tapas.protos import interaction_pb2
from tapas.utils import hybridqa_utils

from google.protobuf import text_format


_CellSelectionMetrics = hybridqa_utils.CellSelectionMetrics
_AnswerType = hybridqa_utils.AnswerType


class HybridqaUtilsTest(absltest.TestCase):

  def test_parse_cell(self):
    cell = interaction_pb2.Cell()
    hybridqa_utils._parse_cell(
        cell,
        text='Hello World',
        links=['/wiki/World'],
        descriptions={'/wiki/World': '...'})

    self.assertEqual(
        cell,
        text_format.Parse(
            """
    text: "Hello World"
    [language.tapas.AnnotatedText.annotated_cell_ext] {
      annotations {
        identifier: "/wiki/World"
      }
    }
    """, interaction_pb2.Cell()))

  def test_parse_interaction(self):
    expected_interaction = text_format.Parse(
        """
    id: "meaning-0"
    table {
      rows {
        cells {
          text: "It is 42"
        }
      }
    }
    questions {
      id: "meaning-0_0"
      original_text: "Meaning of life"
      answer {
        answer_coordinates {
          row_index: 0
          column_index: 0
        }
        answer_texts: "42"
      }
      alternative_answers {
        answer_coordinates {
          row_index: 0
          column_index: 0
        }
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
      descriptions {
        key: "/wiki/World"
        value: "The World is the Earth and all life on it, ..."
      }
    }
    """, interaction_pb2.Interaction())
    counters = collections.Counter()
    interaction = hybridqa_utils._parse_interaction(
        expected_interaction.table,
        descriptions={
            '/wiki/World': 'The World is the Earth and all life on it, ...'
        },
        example={
            'question_id': 'meaning',
            'question': 'Meaning of life',
            'answer-text': '42',
            'answer-node': [[
                'World',
                [5, 6],
                '/wiki/Earth',
                'passage',
            ]]
        },
        counters=counters,
    )
    self.assertDictEqual(counters, {
        'Disjoint answers': 1,
        'Answer type is 1_in_table': 1
    })
    self.assertEqual(interaction, expected_interaction)

  def test_get_best_cells(self):
    result = hybridqa_utils.get_best_cells([(1, 1, 0.8), (1, 2, 0.7),
                                            (3, 2, 0.4), (1, 2, 1.0)])
    self.assertEqual(result, [(2, 1), (1, 1), (2, 3)])

  def test_eval_cell_selection(self):
    tempfile_name = tempfile.mkstemp()[1]
    answer_coordinates = str(['(5, 6)', '(1, 1)', '(1, 2)'])
    token_probabilities = json.dumps([(1, 1, 0.8), (2, 1, 0.7), (2, 3, 0.4),
                                      (6, 5, 0.9)])
    with open(tempfile_name, 'w') as f:
      f.write('question_id\tanswer_coordinates\ttoken_probabilities\n')
      f.write(f'meaning-1_0\t{answer_coordinates}\t{token_probabilities}\n')

    question_1 = interaction_pb2.Question(
        id='meaning-0_0', alternative_answers=[interaction_pb2.Answer()])
    question_2 = text_format.Parse(
        """
    id: "meaning-1_0"
    original_text: "Meaning of life"
    answer {
      answer_coordinates {
        row_index: 5
        column_index: 6
      }
      answer_coordinates {
        row_index: 2
        column_index: 3
      }
      answer_texts: "42"
    }
    alternative_answers {
    }
    """, interaction_pb2.Question())
    questions = {question_1.id: question_1, question_2.id: question_2}
    metrics = dict(hybridqa_utils.eval_cell_selection(questions, tempfile_name))

    self.assertEqual(
        metrics, {
            _AnswerType.ALL:
                _CellSelectionMetrics(
                    recall=0.5,
                    precision=1 / 3,
                    non_empty=0.5,
                    answer_len=1.5,
                    coverage=0.5,
                    recall_at_1=0.5,
                    recall_at_3=0.5,
                    recall_at_5=0.5),
            _AnswerType.MANY_IN_TEXT:
                _CellSelectionMetrics(
                    recall=1.0,
                    precision=1 / 3,
                    non_empty=1.0,
                    answer_len=3.0,
                    coverage=1.0,
                    recall_at_1=1.0,
                    recall_at_3=1.0,
                    recall_at_5=1.0),
            _AnswerType.NO_ANSWER:
                _CellSelectionMetrics(
                    recall=0.0,
                    precision=None,
                    non_empty=0.0,
                    answer_len=0.0,
                    coverage=0.0,
                    recall_at_1=0.0,
                    recall_at_3=0.0,
                    recall_at_5=0.0),
        })


if __name__ == '__main__':
  absltest.main()
