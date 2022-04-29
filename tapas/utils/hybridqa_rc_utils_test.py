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
"""Tests for tapas.utils.google.hybridqa_rc_utils."""

from typing import Text, Mapping, Tuple, Any, Type

from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.utils import hybridqa_rc_utils

from google.protobuf import text_format

NoisyCellStrategy = hybridqa_rc_utils.NoisyCellStrategy


class GetTableDimensionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('size_0_0', {
          'header': [],
          'data': []
      }, (0, 0)),
      ('size_2_2', {
          'header': ['col1', 'col2'],
          'data': [[12, 4], [9, 21]]
      }, (2, 2)),
      ('size_2_1', {
          'header': ['col'],
          'data': [[1], [4]]
      }, (2, 1)),
      ('size_1_2', {
          'header': ['h1', 'h2'],
          'data': [[1, 32]]
      }, (1, 2)),
  )
  def test_possible_dicts(self, table_dict,
                          size):
    self.assertEqual(hybridqa_rc_utils.get_table_dimensions(table_dict), size)

  @parameterized.named_parameters(
      ('size_1_2', """
        columns {text: "A"} columns {text: "B"}
        rows {cells {text: "12"} cells {text: "32"}}
        """, (1, 2)),
      ('size_2_1', """
        columns {text: "H"}
        rows {cells {text: "43"}}
        rows {cells {text: "33"}}
        """, (2, 1)),
      ('size_0_0', '', (0, 0)),
  )
  def test_possible_interactions(self, text_proto_table,
                                 size):
    table = text_format.Parse(text_proto_table, interaction_pb2.Table())
    self.assertEqual(hybridqa_rc_utils.get_table_dimensions(table), size)

  @parameterized.named_parameters(
      ('key_error_1', {
          'headers': [],
          'data': []
      }, ValueError),
      ('key_error_2', {
          'header': []
      }, ValueError),
      ('type_error_1', {
          'header': 12,
          'data': []
      }, ValueError),
      ('type_error_2', {
          'header': [],
          'data': 'Hey'
      }, ValueError),
      ('type_error_3', {
          'header': [],
          'data': {}
      }, ValueError),
      ('type_error_4', [[1, 2, 3], [4, 5, 6]], TypeError),
  )
  def test_bad_table_dicts(self, table_dict,
                           exception_type):
    with self.assertRaises(exception_type):
      hybridqa_rc_utils.get_table_dimensions(table_dict)


class ParseAnswerCellTest(absltest.TestCase):

  def test_parse_answer_cell(self):
    cell = interaction_pb2.Cell()
    hybridqa_rc_utils._parse_answer_cell(
        cell,
        text='Hello World',
        links=['/wiki/World'],
        descriptions={'/wiki/World': '...'},
        url_unquote=True)

    self.assertEqual(
        cell,
        text_format.Parse(
            """
    text: "Hello World : ..."
    [language.tapas.AnnotatedText.annotated_cell_ext] {
      annotations {
        identifier: "/wiki/World"
      }
    }
    """, interaction_pb2.Cell()))

  def test_parse_answer_cell_unicode_1(self):
    cell = interaction_pb2.Cell()
    hybridqa_rc_utils._parse_answer_cell(
        cell,
        text='Zurich, Switzerland',
        links=['/wiki/Z%C3%BCrich'],
        descriptions={'/wiki/Zürich': '...'},
        url_unquote=True)

    self.assertEqual(
        cell,
        text_format.Parse(
            """
    text: "Zurich, Switzerland : ..."
    [language.tapas.AnnotatedText.annotated_cell_ext] {
      annotations {
        identifier: "/wiki/Zürich"
      }
    }
    """, interaction_pb2.Cell()))

  def test_parse_answer_cell_unicode_without_unquote(self):
    cell = interaction_pb2.Cell()
    with self.assertRaises(ValueError):
      hybridqa_rc_utils._parse_answer_cell(
          cell,
          text='Zurich, Switzerland',
          links=['/wiki/Z%C3%BCrich'],
          descriptions={'/wiki/Zürich': '...'},
          url_unquote=False)


class ParseAnswerInteractionsTest(absltest.TestCase):

  def test_parse_answer_interactions_basic(self):
    table_json = {
        'uid':
            0,
        'url':
            'https://en.wikipedia.org/wiki/Earth',
        'title':
            'Earth',
        'header': [['Name', []], ['Number', []]],
        'data': [
            [['U.K', ['/wiki/UK']], ['1.2', []]],
            [['Globe', ['/wiki/World']], ['3.2', []]],
        ]
    }
    interactions = hybridqa_rc_utils._parse_answer_interactions(
        table_json,
        descriptions={
            '/wiki/World': 'The World is the Earth and all life on it, ...',
            '/wiki/UK': 'The United Kingdom',
        },
        example={
            'question_id': 'abcd',
            'question': 'Meaning of life',
            'answer-text': '42',
            'answer-node': [[
                'Earth',
                [1, 0],
                '/wiki/World',
                'passage',
            ]]
        },
        single_cell_examples=False)
    interactions = list(interactions)
    self.assertLen(interactions, 1)
    expected_interaction = text_format.Parse(
        """
    id: "abcd/0-0"
    table {
      columns {
          text: ""
      }
      rows {
        cells {
          text: "Globe : The World is the Earth and all life on it, ..."
          [language.tapas.AnnotatedText.annotated_cell_ext] {
            annotations {
              identifier: "/wiki/World"
            }
          }
        }
      }
      table_id: "0"
      document_title: "Earth"
      document_url: "https://en.wikipedia.org/wiki/Earth"
    }
    questions {
      id: "abcd/0-0_0"
      original_text: "Meaning of life"
      answer {
        answer_texts: "42"
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
      descriptions {
        key: "/wiki/World"
        value: "The World is the Earth and all life on it, ..."
      }
    }
    """, interaction_pb2.Interaction())
    self.assertEqual(interactions[0], expected_interaction)

  def test_parse_answer_interactions_multi_cells(self):
    table_json = {
        'uid':
            0,
        'url':
            'https://en.wikipedia.org/wiki/Earth',
        'title':
            'Earth',
        'header': [['Name', []], ['Number', []]],
        'data': [
            [['U.K', ['/wiki/UK']], ['1.2', []]],
            [['France', ['/wiki/France']], ['1.2', []]],
            [['Globe', ['/wiki/World']], ['3.2', []]],
        ]
    }
    interactions = hybridqa_rc_utils._parse_answer_interactions(
        table_json,
        descriptions={
            '/wiki/World':
                'The World is the Earth and all life on it, ...',
            '/wiki/UK':
                'The United Kingdom is a sovereign country located off the north­western coast of the European mainland.',
            '/wiki/France':
                'France is a country in the Europe continent.',
        },
        example={
            'question_id':
                '1234',
            'question':
                'What is the Number for European countries?',
            'answer-text':
                '1.2',
            'answer-node': [[
                '1.2',
                [0, 1],
                None,
                'table',
            ], [
                '1.2',
                [1, 1],
                None,
                'table',
            ]]
        },
        single_cell_examples=False)
    interactions = list(interactions)
    self.assertLen(interactions, 1)
    expected_interaction = text_format.Parse(
        """
    id: "1234/0-0"
    table {
      columns {
          text: ""
      }
      columns {
          text: ""
      }
      rows {
        cells {
          text: "1.2"
        }
        cells {
          text: "1.2"
        }
      }
      table_id: "0"
      document_title: "Earth"
      document_url: "https://en.wikipedia.org/wiki/Earth"
    }
    questions {
      id: "1234/0-0_0"
      original_text: "What is the Number for European countries?"
      answer {
        answer_texts: "1.2"
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
    }
    """, interaction_pb2.Interaction())
    self.assertEqual(interactions[0], expected_interaction)

  def test_parse_answer_interactions_single_cells(self):
    table_json = {
        'uid':
            0,
        'url':
            'https://en.wikipedia.org/wiki/Earth',
        'title':
            'Earth',
        'header': [['Name', []], ['Number', []]],
        'data': [
            [['U.K', ['/wiki/UK']], ['1.2', []]],
            [['France', ['/wiki/France']], ['1.2', []]],
            [['Globe', ['/wiki/World']], ['3.2', []]],
        ]
    }
    interactions = hybridqa_rc_utils._parse_answer_interactions(
        table_json,
        descriptions={
            '/wiki/World':
                'The World is the Earth and all life on it, ...',
            '/wiki/UK':
                'The United Kingdom is a sovereign country located off the north­western coast of the European mainland.',
            '/wiki/France':
                'France is a country in the Europe continent.',
        },
        example={
            'question_id':
                '1234',
            'question':
                'What is the Number for European countries?',
            'answer-text':
                '1.2',
            'answer-node': [[
                '1.2',
                [0, 1],
                None,
                'table',
            ], [
                '1.2',
                [1, 1],
                None,
                'table',
            ]]
        },
        single_cell_examples=True)
    interactions = list(interactions)
    self.assertLen(interactions, 2)
    expected_interaction = text_format.Parse(
        """
    id: "1234/0-0"
    table {
      columns {
          text: ""
      }
      rows {
        cells {
          text: "1.2"
        }
      }
      table_id: "0"
      document_title: "Earth"
      document_url: "https://en.wikipedia.org/wiki/Earth"
    }
    questions {
      id: "1234/0-0_0"
      original_text: "What is the Number for European countries?"
      answer {
        answer_texts: "1.2"
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
    }
    """, interaction_pb2.Interaction())
    self.assertEqual(interactions[0], expected_interaction)
    self.assertEqual(interactions[1].id, '1234/1-0')
    self.assertLen(interactions[1].table.columns, 1)
    self.assertLen(interactions[1].table.rows, 1)
    self.assertLen(interactions[1].table.rows[0].cells, 1)

  def test_parse_answer_interactions_with_all_coordinates(self):
    table_json = {
        'uid':
            0,
        'url':
            'https://en.wikipedia.org/wiki/Earth',
        'title':
            'Earth',
        'header': [['Name', []], ['Number', []]],
        'data': [
            [['U.K', ['/wiki/UK']], ['1.2', []]],
            [['France', ['/wiki/France']], ['1.2', []]],
            [['Globe', ['/wiki/World']], ['3.2', []]],
        ]
    }
    interactions = hybridqa_rc_utils._parse_answer_interactions(
        table_json,
        descriptions={
            '/wiki/World':
                'The World is the Earth and all life on it, ...',
            '/wiki/UK':
                'The United Kingdom is a sovereign country located off the north­western coast of the European mainland.',
            '/wiki/France':
                'France is a country in the Europe continent.',
        },
        example={
            'question_id': '1234',
            'question': 'What is the Number for European countries?',
            'answer-text': '1.2',
            'answer-node': [[
                '1.2',
                [0, 1],
                None,
                'table',
            ]]
        },
        single_cell_examples=True,
        use_original_coordinates=False)
    interactions = list(interactions)
    self.assertLen(interactions, 2)
    expected_interaction = text_format.Parse(
        """
    id: "1234/1-0"
    table {
      columns {
          text: ""
      }
      rows {
        cells {
          text: "1.2"
        }
      }
      table_id: "0"
      document_title: "Earth"
      document_url: "https://en.wikipedia.org/wiki/Earth"
    }
    questions {
      id: "1234/1-0_0"
      original_text: "What is the Number for European countries?"
      answer {
        answer_texts: "1.2"
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
    }
    """, interaction_pb2.Interaction())
    self.assertEqual(interactions[1], expected_interaction)
    self.assertEqual(interactions[0].id, '1234/0-0')
    self.assertLen(interactions[0].table.columns, 1)
    self.assertLen(interactions[0].table.rows, 1)
    self.assertLen(interactions[0].table.rows[0].cells, 1)


class CreateEvalInteractionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_interaction: interaction_pb2.Interaction = text_format.Parse(
        """
    id: "1234/1-0"
    table {
      columns {text: "Name"}
      columns {text: "Subject"}
      columns {text: "Score"}
      rows {
        cells {text: "Jessica"}
        cells {
          text: "Biology"
          [language.tapas.AnnotatedText.annotated_cell_ext] {
            annotations {
              identifier: "/wiki/Biology"
            }
          }
        }
        cells {text:"98.1"}
      }
      rows {
        cells {text: "Ralph"}
        cells {
          text: "Mathematics"
          [language.tapas.AnnotatedText.annotated_cell_ext] {
            annotations {
              identifier: "/wiki/Mathematics"
            }
          }
        }
        cells {text:"67.8"}
      }
      rows {
        cells {
          text: "Erwin"
          [language.tapas.AnnotatedText.annotated_cell_ext] {
            annotations {
              identifier: "/wiki/Erwin_Schrödinger"
            }
          }
        }
        cells {
          text: "Physics"
          [language.tapas.AnnotatedText.annotated_cell_ext] {
            annotations {
              identifier: "/wiki/Physics"
            }
          }
        }
        cells {text:"99.2"}
      }
      table_id: "0"
      document_title: "Earth"
      document_url: "https://en.wikipedia.org/wiki/Earth"
    }
    questions {
      id: "1234/1-0_0"
      original_text: "How much did the developer of the theory of relativity scored in Physics in his high school?"
      answer {
        answer_texts: "99.2"
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
      descriptions {
        key: "/wiki/Biology"
        value: "Biology is the natural science that studies life and living organisms."
      }
      descriptions {
        key: "/wiki/Mathematics"
        value: "The abstract science of number, quantity, and space."
      }
      descriptions {
        key: "/wiki/Physics"
        value: "The branch of science concerned with the nature and properties of matter and energy."
      }
      descriptions {
        key: "/wiki/Erwin_Schrödinger"
        value: "Nobel Prize-winning Austrian-Irish physicist who developed a number of fundamental results in quantum theory."
      }
    }
    """, interaction_pb2.Interaction())

  def test_full_interaction_multi_cell_table(self):
    coordinates = [(2, 0), (2, 2)]
    interactions = list(
        hybridqa_rc_utils._create_eval_answer_interactions(
            self.input_interaction, coordinates))
    self.assertLen(interactions, 1)
    self.assertEqual(
        interactions[0],
        text_format.Parse(
            """
    id: "1234/0-0"
    table {
      columns {
        text: ""
      }
      columns {
        text: ""
      }
      rows {
        cells {
          text: "Erwin : Nobel Prize-winning Austrian-Irish physicist who developed a number of fundamental results in quantum theory."
          [language.tapas.AnnotatedText.annotated_cell_ext] {
            annotations {
              identifier: "/wiki/Erwin_Schrödinger"
            }
          }
        }
        cells {
          text: "99.2"
        }
      }
      table_id: "0"
      document_title: "Earth"
      document_url: "https://en.wikipedia.org/wiki/Earth"
    }
    questions {
      id: "1234/0-0_0"
      original_text: "How much did the developer of the theory of relativity scored in Physics in his high school?"
      answer {
        answer_texts: "99.2"
      }
    }
    [language.tapas.AnnotationDescription.annotation_descriptions_ext] {
      descriptions {
        key: "/wiki/Erwin_Schrödinger"
        value: "Nobel Prize-winning Austrian-Irish physicist who developed a number of fundamental results in quantum theory."
      }
    }
    """, interaction_pb2.Interaction()))

  def test_multi_cell_table(self):
    coordinates = [(0, 0), (1, 1)]
    interactions = list(
        hybridqa_rc_utils._create_eval_answer_interactions(
            self.input_interaction, coordinates))
    self.assertLen(interactions, 1)
    self.assertEqual(
        interactions[0].table,
        text_format.Parse(
            """
    columns {
      text: ""
    }
    columns {
      text: ""
    }
    rows {
      cells {
        text: "Jessica"
      }
      cells {
        text: "Mathematics : The abstract science of number, quantity, and space."
        [language.tapas.AnnotatedText.annotated_cell_ext] {
          annotations {
            identifier: "/wiki/Mathematics"
          }
        }
      }
    }
    table_id: "0"
    document_title: "Earth"
    document_url: "https://en.wikipedia.org/wiki/Earth"
    """, interaction_pb2.Table()))

  def test_single_cell_tables_common_properties(self):
    coordinates = [(0, 1), (1, 1), (2, 0), (2, 2)]
    interactions = list(
        hybridqa_rc_utils._create_eval_answer_interactions(
            self.input_interaction, coordinates, single_cell_examples=True))

    self.assertLen(interactions, 4)

    # All tables have single row
    self.assertEqual({len(intr.table.rows) for intr in interactions}, {1})

    # The single row has single cell
    self.assertEqual({len(intr.table.rows[0].cells) for intr in interactions},
                     {1})

    # All tables have single column
    self.assertEqual({len(intr.table.columns) for intr in interactions}, {1})

    # All tables have empty column header
    self.assertEqual({intr.table.columns[0].text for intr in interactions},
                     {''})

    # All tables have same table_id
    self.assertEqual({intr.table.table_id for intr in interactions}, {'0'})

    # All tables have same document_title
    self.assertEqual({intr.table.document_title for intr in interactions},
                     {'Earth'})

    # All tables have same document_url
    self.assertEqual({intr.table.document_url for intr in interactions},
                     {'https://en.wikipedia.org/wiki/Earth'})

  def test_single_cell_tables_content(self):
    coordinates = [(0, 2), (2, 1), (1, 0)]
    interactions = list(
        hybridqa_rc_utils._create_eval_answer_interactions(
            self.input_interaction, coordinates, single_cell_examples=True))
    self.assertLen(interactions, 3)

    self.assertEqual(
        interactions[0].table.rows[0],
        text_format.Parse("""
    cells {
        text: "98.1"
      }
    """, interaction_pb2.Cells()))
    self.assertEqual(interactions[0].id, '1234/0-0')

    self.assertEqual(
        interactions[1].table.rows[0],
        text_format.Parse(
            """
    cells {
      text: "Physics : The branch of science concerned with the nature and properties of matter and energy."
      [language.tapas.AnnotatedText.annotated_cell_ext] {
        annotations {
          identifier: "/wiki/Physics"
        }
      }
    }
    """, interaction_pb2.Cells()))
    self.assertEqual(interactions[1].id, '1234/1-0')

    self.assertEqual(
        interactions[2].table.rows[0],
        text_format.Parse("""
    cells {
        text: "Ralph"
      }
    """, interaction_pb2.Cells()))
    self.assertEqual(interactions[2].id, '1234/2-0')


class NoisySampleSpaceSelectionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('full_rand_1', NoisyCellStrategy.FULL_RANDOM, 2, 3, [(1, 1),
                                                            (0, 2)], [(0, 0),
                                                                      (0, 1),
                                                                      (1, 0),
                                                                      (1, 2)]),
      ('full_rand_2', NoisyCellStrategy.FULL_RANDOM, 2, 2, [(1, 1),
                                                            (0, 0)], [(0, 1),
                                                                      (1, 0)]),
      ('full_rand_3', NoisyCellStrategy.FULL_RANDOM, 2, 2, [(1, 1), (0, 0),
                                                            (0, 1),
                                                            (1, 0)], []),
      ('selective_1', NoisyCellStrategy.SELECTIVE_RANDOM, 3, 3, [(1, 1), (0, 2)
                                                                ], [(0, 0),
                                                                    (0, 1),
                                                                    (1, 0),
                                                                    (1, 2),
                                                                    (2, 1),
                                                                    (2, 2)]),
      ('selective_2', NoisyCellStrategy.SELECTIVE_RANDOM, 2, 3, [(1, 1), (1, 0)
                                                                ], [(0, 0),
                                                                    (0, 1),
                                                                    (1, 2)]),
      ('selective_3', NoisyCellStrategy.SELECTIVE_RANDOM, 3, 3, [(2, 2)
                                                                ], [(0, 2),
                                                                    (1, 2),
                                                                    (2, 0),
                                                                    (2, 1)]),
      ('col_1', NoisyCellStrategy.COLUMN_CONFINED_RANDOM, 3, 3, [(0, 1), (1, 0)
                                                                ], [(1, 1),
                                                                    (2, 1),
                                                                    (0, 0),
                                                                    (2, 0)]),
      ('col_2', NoisyCellStrategy.COLUMN_CONFINED_RANDOM, 4, 4, [(0, 1), (1, 1)
                                                                ], [(2, 1),
                                                                    (3, 1)]),
      ('none_1', NoisyCellStrategy.NONE, 4, 4, [(0, 1), (1, 1)], []),
      ('none_2', NoisyCellStrategy.NONE, 2, 2, [(0, 0)], []),
  )
  def test_sample_space_generation(self, strategy, n_rows, n_cols,
                                   true_coordinates, expected_sample_space):
    generated_space = hybridqa_rc_utils.generate_sample_space(
        strategy, n_rows, n_cols, true_coordinates)
    self.assertCountEqual(expected_sample_space, [*generated_space])


if __name__ == '__main__':
  absltest.main()
