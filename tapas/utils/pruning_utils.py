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
"""Defines the methods to us for columns selection."""

import collections
import enum
import queue
from typing import Dict, Tuple, Text, Set, Optional, List, Iterable

import apache_beam as beam
import dataclasses
from nltk.stem.porter import PorterStemmer
from tapas.protos import interaction_pb2
from tapas.protos import table_selection_pb2
from tapas.utils import text_utils
from tapas.utils import tf_example_utils

_NS = "main"
_Coordinates = tf_example_utils.TokenCoordinates


def get_column_texts(
    table,
    column_index,
):
  """Iterates through the text in the column including the header."""
  yield table.columns[column_index].text
  for row in table.rows:
    yield row.cells[column_index].text


def _get_column_cost(
    tokenizer,
    table,
    column_index,
):
  r"""Computes length of the serialized column."""
  num_tokens = 0
  for text in get_column_texts(table, column_index):
    num_tokens += len(tokenizer.tokenize(text))
  return num_tokens


def _get_question_cost(
    tokenizer,
    question,
):
  r"""Computes length of the serialized question (w/ special token offset)."""
  tokens = tokenizer.tokenize(
      text_utils.normalize_for_match(question.original_text))
  return tokenizer.question_encoding_cost(tokens)


def get_cleaned_seq_tokens(str_tokens):
  """Transform a string to a cleaned list of tokens.

  Args:
    str_tokens: the string to tokenize

  Returns:
    A list of tokens
  """
  stemmer = PorterStemmer()
  tokens = text_utils.tokenize_text(text_utils.format_text(str_tokens))
  return [stemmer.stem(token) for token in tokens]


def from_selected_columns_to_selected_tokens(
    interaction,
    selected_columns,
    tokenizer):
  r"""Gets all the tokens' coordinates of the selected columns."""

  selected_tokens = set()

  def _add_cell_tokens(column_index, row_index, cell_text, selected_tokens):
    cell_tokens = tokenizer.tokenize(cell_text)
    for token_index in range(len(cell_tokens)):
      selected_tokens.add(
          _Coordinates(
              column_index=column_index,
              row_index=row_index,
              token_index=token_index))

  for column_index in selected_columns:
    _add_cell_tokens(column_index, 0,
                     interaction.table.columns[column_index].text,
                     selected_tokens)
    for row_index, row in enumerate(interaction.table.rows):
      _add_cell_tokens(column_index, row_index + 1,
                       row.cells[column_index].text, selected_tokens)
  return selected_tokens


def _get_question_column_similarity(column_tokens,
                                    question_tokens,
                                    epsilon = 0.0):
  r"""Gives the scored using Jaccard coefficient.

  (for more details see https://en.wikipedia.org/wiki/Jaccard_index)
  Formula: score = |{token \in C | token \in Q}| / |{token\in C}|.
  Where C contains all the tokens extracted from the column i's strings
  and Q the question's string.

  Args:
    column_tokens: expected set of tokens from one column
    question_tokens: expected set of tokens from one question
    epsilon: expected value >= 0.0. In some cases the expected returned values >
      0.0 (non zero). The epsilon must be set to a value > 0.0.

  Returns:
    The Jaccard coefficient
  """

  nominator = column_tokens & question_tokens
  denominator = column_tokens | question_tokens
  if not denominator and not epsilon:
    return 0.0
  return (len(nominator) + epsilon) / (len(denominator) + epsilon)


class SelectionType(enum.Enum):
  COLUMN = 1
  CELL = 2


@dataclasses.dataclass
class TokenScoreDebugInfo:
  score: float = 0.0
  nb_tokens: int = 0


@dataclasses.dataclass(frozen=True)
class TokenSelection:
  selected_tokens: Set[_Coordinates]
  debug: Optional[table_selection_pb2.TableSelection.DebugInfo] = None


class TokenSelector:
  """Base class for column selection."""

  def __init__(self, vocab_file, use_previous_questions):
    self._use_previous_questions = use_previous_questions
    self._tokenizer = tf_example_utils.TapasTokenizer(vocab_file)

  def prepare(self):
    pass

  def select_tokens(
      self,
      interaction,
      question,
  ):
    raise NotImplementedError()

  def annotated_interaction(
      self,
      interaction,
  ):
    """Selects columns using select_columns_fn for each question in the current interaction."""
    new_interaction = interaction_pb2.Interaction()
    new_interaction.CopyFrom(interaction)
    questions = []

    for q in new_interaction.questions:
      text = q.text
      if self._use_previous_questions:
        questions.append(q.original_text)
        q.text = " ".join(questions)
      else:
        q.text = q.original_text
      token_selection = self.select_tokens(interaction, q)
      q.text = text
      selected_tokens = token_selection.selected_tokens
      table_ext = table_selection_pb2.TableSelection.table_selection_ext
      question_tokens = q.Extensions[table_ext].selected_tokens
      del question_tokens[:]
      for token in selected_tokens:
        new_token = question_tokens.add()
        new_token.column_index = token.column_index
        new_token.row_index = token.row_index
        new_token.token_index = token.token_index
      if token_selection.debug:
        q.Extensions[table_ext].debug.CopyFrom(token_selection.debug)
    return new_interaction


class MaxTokenSelector(TokenSelector):
  """Selects columns until a certain number of tokens is met."""

  def __init__(self, vocab_file, max_nb_tokens,
               selection_level, use_previous_answer,
               use_previous_questions):
    super().__init__(
        vocab_file=vocab_file, use_previous_questions=use_previous_questions)
    self._max_nb_tokens = max_nb_tokens
    self._use_previous_answer = use_previous_answer

    # Extracts columns or cells depending on the selection_level.
    if selection_level == SelectionType.CELL:
      self.select_tokens = self._select_cells
    elif selection_level == SelectionType.COLUMN:
      self.select_tokens = self._select_columns
    else:
      raise ValueError(
          f"The selection level is not implemented {selection_level}")

  def _get_column_tokens(self, interaction,
                         column_index):
    column_tokens = []
    for row in interaction.table.rows:
      column_tokens.extend(get_cleaned_seq_tokens(row.cells[column_index].text))
    column_tokens.extend(
        get_cleaned_seq_tokens(interaction.table.columns[column_index].text))
    return column_tokens

  def _get_columns_tokens(
      self, interaction):
    return {
        column_index: self._get_column_tokens(interaction, column_index)
        for column_index in range(len(interaction.table.columns))
    }

  def _get_columns_similarity_scores(
      self, interaction,
      question_tokens):
    columns_tokens = self._get_columns_tokens(interaction)
    columns_score = {
        column_index:
        _get_question_column_similarity(set(column_tokens), question_tokens)
        for column_index, column_tokens in columns_tokens.items()
    }
    return columns_score

  def _get_headers_tokens(
      self, interaction):
    return {
        column_index: get_cleaned_seq_tokens(column.text)
        for column_index, column in enumerate(interaction.table.columns)
    }

  def _get_headers_similarity_scores(
      self, interaction,
      question_tokens):
    headers_tokens = self._get_headers_tokens(interaction)
    return {
        column_index:
        _get_question_column_similarity(set(header_token), question_tokens)
        for column_index, header_token in headers_tokens.items()
    }

  def _get_cells(
      self, interaction
  ):
    r"""Extracts cells tokens.

    Args:
      interaction: Contains the table cells.

    Returns:
      Dictionary where the keys are the row indexe and column index of the cell.
      The value is the list of tokens o a cell.
    """
    cells = {}
    for column_index in range(len(interaction.table.columns)):
      cells[(column_index, 0)] = get_cleaned_seq_tokens(
          interaction.table.columns[column_index].text)

      for row_index, row in enumerate(interaction.table.rows):
        cells[(column_index, row_index + 1)] = get_cleaned_seq_tokens(
            row.cells[column_index].text)
    return cells

  def _get_row_tokens(self, interaction,
                      row_index):
    row_tokens = []
    for column_index in range(len(interaction.table.columns)):
      cell_tokens = get_cleaned_seq_tokens(
          interaction.table.rows[row_index].cells[column_index].text)
      row_tokens.extend(cell_tokens)
    return row_tokens

  def _get_header_tokens(
      self, interaction):
    row_tokens = []
    for column in interaction.table.columns:
      cell_tokens = get_cleaned_seq_tokens(column.text)
      row_tokens.extend(cell_tokens)
    return row_tokens

  def _get_row_similarity_scores(
      self, interaction,
      question_tokens):
    r"""Computes the rows scores.

    Args:
      interaction: Contains the table cells.
      question_tokens: List of the question tokens.

    Returns:
      Dictionary where the keys are the rows' indexes.
      Row index = 0 is the header.
    """

    header_tokens = self._get_header_tokens(interaction)
    row_scores = {
        0: _get_question_column_similarity(set(header_tokens), question_tokens)
    }
    for row_index in range(len(interaction.table.rows)):
      row_tokens = self._get_row_tokens(interaction, row_index)
      row_scores[row_index + 1] = _get_question_column_similarity(
          set(row_tokens), question_tokens)
    return row_scores

  def _update_priority_queue_from_previous_answer(
      self,
      selection_type,
      priority_queue,
      interaction,
      question,
  ):
    r"""gives high priority to a answer columns from the previous question.

    Args:
      selection_type: Specify the selection type: column or a cell.
      priority_queue: expected values: Tuple (-score, index of scored column) or
        Tuple (-score, index of scored column, index of row , index of token)
      interaction: contains the table cells.
      question: contains the original text of the question.

    Returns:
      A modified priority queue.
    """
    if not self._use_previous_answer:
      return priority_queue
    if len(interaction.questions) > 1:
      index = 0
      for index, q in enumerate(interaction.questions):
        if q.id == question.id:
          break
      if index > 0:
        answer_coords = interaction.questions[index -
                                              1].answer.answer_coordinates
        if selection_type == SelectionType.COLUMN:
          answer_coords = set([c.column_index for c in answer_coords])
        elif selection_type == SelectionType.CELL:
          answer_coords = [(c.row_index, c.column_index) for c in answer_coords]
        else:
          raise NotImplementedError()

        new_priority_queue = queue.PriorityQueue()
        # The priority queue prioritize the minimum scores:
        # max_priority_score = - max_score
        max_priority_score = -len(interaction.table.columns)

        while not priority_queue.empty():
          element = priority_queue.get()
          if selection_type == SelectionType.COLUMN:
            _, index_look_up = element
            new_element = (max_priority_score, index_look_up)
          elif selection_type == SelectionType.CELL:
            _, column_index, row_index, token_index = element
            index_look_up = (row_index, column_index)
            new_element = (max_priority_score, column_index, row_index,
                           token_index)
          else:
            raise NotImplementedError()

          if index_look_up in answer_coords:
            new_priority_queue.put(new_element)
          else:
            new_priority_queue.put(element)
        return new_priority_queue
    return priority_queue

  def _get_columns_from_priority_queue(
      self,
      columns_queue,
      interaction,
      question,
  ):
    r"""Selects tokenss with higher score up to max_nb_tokens.

    Args:
      columns_queue: expected values: Tuple (-score, index of scored column)
      interaction: contains the table cells.
      question: contains the original text of the question.

    Returns:
      The set of selected columns' indexes.
    """
    columns_queue = self._update_priority_queue_from_previous_answer(
        SelectionType.COLUMN, columns_queue, interaction, question)

    selected_columns = set()
    num_tokens = _get_question_cost(self._tokenizer, question)
    table = interaction.table
    num_columns_skipped = 0

    debug = table_selection_pb2.TableSelection.DebugInfo()

    while not columns_queue.empty():
      score, column_index = columns_queue.get()
      column_cost = _get_column_cost(self._tokenizer, table, column_index)
      column_debug = debug.columns.add()
      column_debug.index = column_index
      column_debug.score = -float(score)

      # Selects columns without exceeding maximum number of tokens.
      if num_tokens + column_cost > self._max_nb_tokens:
        num_columns_skipped += 1
        column_debug.is_selected = False
        continue

      column_debug.is_selected = True
      if num_columns_skipped > 0:
        beam.metrics.Metrics.counter(_NS, "Squeezing in another column").inc()
      num_tokens += column_cost
      selected_columns.add(column_index)

    if not selected_columns:
      beam.metrics.Metrics.counter(_NS, "No column selected").inc()

    if num_columns_skipped == 0:
      beam.metrics.Metrics.counter(_NS, "All columns selected").inc()
    else:
      beam.metrics.Metrics.counter(_NS, "Columns removed").\
          inc(num_columns_skipped)

    # Check if an answer column wasn't selected.
    for answer_coordinate in question.answer.answer_coordinates:
      answer_column_index = answer_coordinate.column_index
      if answer_column_index < 0:
        continue
      if answer_column_index not in selected_columns:
        beam.metrics.Metrics.counter(_NS, "Answer columns removed").inc()
        if not question.answer.HasField("float_value"):
          beam.metrics.Metrics.counter(_NS, "Unambiguous columns removed").inc()
        break
    selected_tokens = from_selected_columns_to_selected_tokens(
        interaction, selected_columns, self._tokenizer)
    return TokenSelection(selected_tokens, debug)

  def _get_tokens_from_priority_queue(
      self,
      tokens_queue,
      interaction,
      question,
  ):
    r"""Selects tokenss with higher score up to max_nb_tokens.

    Args:
      tokens_queue: expected values: Tuple (-score, column_index, row_index,
        token_index)
      interaction: contains the table cells.
      question: contains the original text of the question.

    Returns:
      The set of selected columns' indexes.
    """

    tokens_queue = self._update_priority_queue_from_previous_answer(
        SelectionType.CELL, tokens_queue, interaction, question)
    table = interaction.table
    selected_tokens = set()
    num_tokens = _get_question_cost(self._tokenizer, question)
    debug = table_selection_pb2.TableSelection.DebugInfo()
    debug_column = collections.defaultdict(TokenScoreDebugInfo)
    cell_score = collections.defaultdict(lambda: 0.0)

    if num_tokens > self._max_nb_tokens:
      beam.metrics.Metrics.counter(_NS, "No column selected").inc()
      return TokenSelection(selected_tokens)

    while not tokens_queue.empty() and num_tokens < self._max_nb_tokens:
      num_tokens += 1
      score, column_index, row_index, token_index = tokens_queue.get()
      selected_tokens.add(_Coordinates(column_index, row_index, token_index))
      debug_column[column_index].score -= float(score)
      debug_column[column_index].nb_tokens += 1
      previous_score = cell_score[(column_index, row_index)]
      cell_score[(column_index, row_index)] = max(-float(score), previous_score)

    # Add debug for column selection.
    num_columns_skipped = 0
    num_tokens_skipped = 0
    num_cells_skipped = 0
    for column_index in range(len(table.columns)):
      column_debug = debug.columns.add()
      column_debug.index = column_index
      if column_index in debug_column.keys():
        debug_info = debug_column[column_index]
        column_debug.score = debug_info.score / debug_info.nb_tokens
        column_debug.is_selected = True
        column_cost = _get_column_cost(self._tokenizer, table, column_index)
        num_tokens_skipped += column_cost - debug_info.nb_tokens
        num_rows_selected = len(
            set(coord.row_index
                for coord in selected_tokens
                if coord.column_index == column_index))
        num_cells_skipped += len(table.rows) + 1 - num_rows_selected
      else:
        column_debug.score = 0.0
        column_debug.is_selected = False
        num_columns_skipped += 1

    if num_tokens_skipped == 0:
      beam.metrics.Metrics.counter(_NS, "All tokens are selected").inc()
    else:
      beam.metrics.Metrics.counter(_NS,
                                   "Tokens removed").inc(num_tokens_skipped)
    if num_cells_skipped == 0:
      beam.metrics.Metrics.counter(
          _NS, "Selected at least one token of every cell").inc()
    else:
      beam.metrics.Metrics.counter(
          _NS, "Cells removed (no token selected)").inc(num_cells_skipped)
    if num_columns_skipped == 0:
      beam.metrics.Metrics.counter(
          _NS, "Selected at least one token of every column").inc()
    else:
      beam.metrics.Metrics.counter(
          _NS, "Columns removed (no token selected)").inc(num_columns_skipped)

    # Check if an answer column wasn't selected.
    scored_zero = 0
    for answer_coordinate in question.answer.answer_coordinates:
      answer_column_index = answer_coordinate.column_index
      answer_row_index = answer_coordinate.row_index
      if answer_column_index < 0:
        continue
      if (answer_column_index, answer_row_index) in cell_score.keys():
        found_score = cell_score[(answer_column_index, answer_row_index)]
        if found_score == 0:
          scored_zero += 1
      else:
        beam.metrics.Metrics.counter(_NS, "Answer cells removed").inc()
        if not question.answer.HasField("float_value"):
          beam.metrics.Metrics.counter(_NS, "Unambiguous cells removed").inc()
        break
    if scored_zero > 0:
      beam.metrics.Metrics.counter(_NS,
                                   "Answer cell is scored 0").inc(scored_zero)
    return TokenSelection(selected_tokens, debug)

  def _select_columns(
      self,
      interaction,
      question,
  ):
    raise NotImplementedError()

  def _select_cells(
      self,
      interaction,
      question,
  ):
    raise NotImplementedError()


class HeuristicExactMatchTokenSelector(MaxTokenSelector):
  r"""Extracts columns that contain tokens'strings match a subset of the question's string."""

  def _select_columns(
      self,
      interaction,
      question,
  ):
    r"""Extracts columns that contain tokens'strings match a subset of the question's string.

    Args:
      interaction: contains the cells.
      question: contains the original text of the question.

    Returns:
      The set of selected columns' indexes.
    """

    question_tokens = set(get_cleaned_seq_tokens(question.text))
    columns_queue = queue.PriorityQueue()
    for i in range(len(interaction.table.columns)):
      column_tokens = self._get_column_tokens(interaction, i)
      score = _get_question_column_similarity(
          set(column_tokens), question_tokens)
      columns_queue.put((-score, i))
    return self._get_columns_from_priority_queue(columns_queue, interaction,
                                                 question)

  def _select_cells(
      self,
      interaction,
      question,
  ):
    r"""Extracts cells that contain tokens'strings match a subset of the question's string.

    The final score used for the priority queue is cell_score + column_score +
    row_score.

    Args:
      interaction: contains the cells.
      question: contains the original text of the question.

    Returns:
      The set of selected tokens' indexes.
    """
    question_tokens = set(get_cleaned_seq_tokens(question.text))
    rows_scores = self._get_row_similarity_scores(interaction, question_tokens)
    columns_scores = self._get_columns_similarity_scores(
        interaction, question_tokens)
    cells = self._get_cells(interaction)
    tokens_queue = queue.PriorityQueue()
    for indexes, cell_tokens in cells.items():
      column_index, row_index = indexes
      row_score = rows_scores[row_index]
      column_score = columns_scores[column_index]
      cell_score = _get_question_column_similarity(
          set(cell_tokens), question_tokens)
      score = column_score + row_score + cell_score
      for token_index in range(len(cell_tokens)):
        tokens_queue.put((-score, column_index, row_index, token_index))
    return self._get_tokens_from_priority_queue(tokens_queue, interaction,
                                                question)


class SelectAllTokensFn(TokenSelector):
  """Extracts all the columns."""

  def select_tokens(self, interaction,
                    question):
    r"""Extracts all the columns.

    Args:
      interaction: gives the number of columns.
      question: not used.

    Returns:
      The set of all columns' indexes.
    """
    del question  # Unused
    selected_columns = set(range(len(interaction.table.columns)))
    selected_tokens = from_selected_columns_to_selected_tokens(
        interaction, selected_columns, self._tokenizer)
    return TokenSelection(selected_tokens)
