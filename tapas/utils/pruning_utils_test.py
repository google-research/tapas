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
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.protos import table_selection_pb2
from tapas.utils import pruning_utils
from tapas.utils import tf_example_utils
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

_Coordinates = tf_example_utils.TokenCoordinates


class PruningUtilsTest(parameterized.TestCase):

  def _assert_equals_selected_columns(self, expected_set, selected_tokens_set):
    selected_columns = set(
        [coord.column_index for coord in selected_tokens_set])
    self.assertEqual(expected_set, selected_columns)

  def _create_vocab(self, output_dir):
    vocab_file = os.path.join(output_dir, "vocab.txt")
    with tf.io.gfile.GFile(vocab_file, "w") as out:
      for word in "[PAD] [UNK] [CLS] [SEP] [MASK] [EMPTY]":
        out.write(f"{word}\n")
      for word in "0 1 2 3 4 5 6 a b c d e f g".split():
        out.write(f"{word}\n")
    return vocab_file

  def _get_interaction(self):
    return interaction_pb2.Interaction(
        table=interaction_pb2.Table(
            columns=[
                interaction_pb2.Cell(text="A:/, c"),
                interaction_pb2.Cell(text="B"),
                interaction_pb2.Cell(text="C"),
            ],
            rows=[
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text="0"),
                    interaction_pb2.Cell(text="4"),
                    interaction_pb2.Cell(text="6"),
                ]),
                interaction_pb2.Cells(cells=[
                    interaction_pb2.Cell(text="1"),
                    interaction_pb2.Cell(text="3"),
                    interaction_pb2.Cell(text="5"),
                ]),
            ],
        ),
        questions=[
            interaction_pb2.Question(
                id="id-1",
                original_text="A is 5",
                text="A is 5",
                answer=interaction_pb2.Answer(answer_coordinates=[
                    interaction_pb2.AnswerCoordinate(
                        row_index=2, column_index=2),
                    interaction_pb2.AnswerCoordinate(
                        row_index=0, column_index=2),
                ])),
            interaction_pb2.Question(
                id="id-2", original_text="B is A", text="A is 5 B is A")
        ],
    )

  def test_SelectAllTokensFn(self):
    r"""Tests SelectAllTokensFn behaviour.

    Ignores max number of tokens and select all columns.
    Test that all the tokens in the selected columns are selected.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      vocab_file = self._create_vocab(temp_dir)
      selector = pruning_utils.SelectAllTokensFn(
          vocab_file, use_previous_questions=False)
      interaction = self._get_interaction()
      selected_columns = selector.select_tokens(interaction,
                                                interaction.questions[0])
    self._assert_equals_selected_columns(
        set([0, 1, 2]), selected_columns.selected_tokens)

    expected_tokens = set([
        _Coordinates(0, 0, 0),
        _Coordinates(0, 0, 1),
        _Coordinates(0, 0, 2),
        _Coordinates(0, 0, 3),
        _Coordinates(0, 0, 4),
        _Coordinates(0, 1, 0),
        _Coordinates(0, 2, 0),
        _Coordinates(1, 0, 0),
        _Coordinates(1, 1, 0),
        _Coordinates(1, 2, 0),
        _Coordinates(2, 0, 0),
        _Coordinates(2, 1, 0),
        _Coordinates(2, 2, 0)
    ])
    self.assertEqual(expected_tokens, selected_columns.selected_tokens)

  def test_SelectHeuristicExactMatchCellsFn(self):
    r"""Tests HeuristicExactMatch behaviour for cell selection."""
    with tempfile.TemporaryDirectory() as temp_dir:
      vocab_file = self._create_vocab(temp_dir)
      selector = pruning_utils.HeuristicExactMatchTokenSelector(
          vocab_file=vocab_file,
          max_nb_tokens=12,
          selection_level=pruning_utils.SelectionType.CELL,
          use_previous_questions=False,
          use_previous_answer=False)
      interaction = self._get_interaction()
      selected_cells = selector.select_tokens(interaction,
                                              interaction.questions[0])
    selected_tokens = selected_cells.selected_tokens

    expected_tokens = set([
        _Coordinates(0, 0, 0),
        _Coordinates(0, 0, 1),
        _Coordinates(1, 0, 0),
        _Coordinates(0, 2, 0),
        _Coordinates(2, 0, 0),
        _Coordinates(1, 2, 0),
        _Coordinates(2, 2, 0)
    ])
    self.assertEqual(expected_tokens, selected_tokens)

    expected_debug = """
        columns {
          index: 0
          score:0.5333333333333333
          is_selected: true
        }
        columns {
          index: 1
          score:0.2
          is_selected: true
        }
        columns {
          index: 2
          score:0.5666666666666667
          is_selected: true
        }
      """
    self.assertEqual(
        selected_cells.debug,
        text_format.Parse(expected_debug,
                          table_selection_pb2.TableSelection.DebugInfo()),
    )

  @parameterized.parameters(
      (8, {2}),
      (11, {1, 2}),
      (15, {0, 2}),
      (18, {0, 1, 2}),
  )
  def test_MaxTokenSelector(
      self,
      max_nb_tokens,
      expected_columns,
  ):  # pylint: disable=g-doc-args
    r"""Tests HeuristicExactMatchTokenSelection.

    - Test scoring: The columns are ordered 2, 0, 1.
    - The cleaned tokens are well created: remove punctuation.
      If not the order changes and becomes 0, 1, 2.
    - Select number of columns that respects the maximum number of tokens.
      max_nb_tokens = 8. The columns are ordered 2, 0, 1.
      The respective number of tokens are 4, 3, 3.
      Then only one column is selected 2.
    - In the second case, we include column 1 rather than 0 even though 0 has
      a better score because 0 doesn"t fit in the budget
    - In the third case, we include 0 because we have more budget.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      vocab_file = self._create_vocab(temp_dir)
      selector = pruning_utils.HeuristicExactMatchTokenSelector(
          vocab_file=vocab_file,
          max_nb_tokens=max_nb_tokens,
          selection_level=pruning_utils.SelectionType.COLUMN,
          use_previous_questions=False,
          use_previous_answer=False)
      interaction = self._get_interaction()
      selected_columns = selector.select_tokens(interaction,
                                                interaction.questions[0])
    self._assert_equals_selected_columns(expected_columns,
                                         selected_columns.selected_tokens)


if __name__ == "__main__":
  absltest.main()
