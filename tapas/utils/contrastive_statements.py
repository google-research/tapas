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
"""Utils for creating contrastive statements."""

import collections
import dataclasses
import enum
import math
import random
from typing import Callable, Dict, Iterable, Optional, Set, Text, Tuple

from absl import logging
import nltk
from tapas.protos import annotated_text_pb2
from tapas.protos import interaction_pb2
from tapas.utils import number_utils
from tapas.utils import sentence_tokenizer
from tapas.utils import text_utils


class ReferenceType(enum.Enum):
  ENTITY = 1
  CARDINAL = 2
  ORDINAL = 3
  DATE = 4


@dataclasses.dataclass
class Reference:
  """Stores all mentions of an identifier.

     In an abuse coordinates can contain table coordinates (column & row)
     or text coordinates (i.e. spans with begin and end byte index).
  """
  span_texts: Set[Text]
  coordinates: Set[Tuple[int, int]]


_ReferenceDict = Dict[Tuple[ReferenceType, Text], Reference]


@dataclasses.dataclass(frozen=True)
class InteractionReferences:
  table: _ReferenceDict
  question: Iterable[_ReferenceDict]


def _is_float(text):
  try:
    float(text)
    return True
  except ValueError:
    return False


def _add_identifier(identifier, reference_type, text,
                    references, row_index,
                    column_index):
  """Adds reference to 'references'."""
  if len(text) < 3 and not _is_float(text):
    return

  key = identifier, reference_type

  if key not in references:
    references[key] = Reference(set(), set())  # pytype: disable=container-type-mismatch
  references[key].span_texts.add(text)
  references[key].coordinates.add((row_index, column_index))


def _add_entity_from_cell(
    cell,
    references,
    row_index,
    column_index,
):
  """Add references for all annotations in the annotated text."""
  annotated_text = cell.Extensions[
      annotated_text_pb2.AnnotatedText.annotated_cell_ext]

  # We require that the annotations spans the whole text and is unique.
  if len(annotated_text.annotations) != 1:
    return

  annotation = annotated_text.annotations[0]

  span_len = annotation.end_byte_index - annotation.begin_byte_index

  if span_len != len(cell.text):
    return

  _add_identifier(
      annotation.identifier,
      ReferenceType.ENTITY,
      cell.text,
      references,
      row_index,
      column_index,
  )


def _to_identifier(
    text,
    numeric_value):
  """Converts numeric value to identifier / reference type."""
  if numeric_value.HasField('float_value'):
    reference_type = ReferenceType.CARDINAL
    if number_utils.is_ordinal(text.lower()):
      reference_type = ReferenceType.ORDINAL
    return str(numeric_value.float_value), reference_type
  elif numeric_value.HasField('date'):
    date = numeric_value.date
    return f'{date.year}-{date.month}-{date.day}', ReferenceType.DATE
  raise ValueError('Cannot convert numeric value')


def _is_numerically_one(numeric_value):
  return math.fabs(numeric_value.float_value - 1.0) < 1.0e-5


def _add_numeric_reference_from_cell(
    cell,
    references,
    row_index,
    column_index,
):
  """Adds number and date references."""
  text = text_utils.normalize_for_match(cell.text)
  spans = number_utils.parse_text(text)

  for span in spans:

    # Only keep spans that match the entire cell.
    if span.end_index - span.begin_index != len(text):
      continue

    for value in span.values:

      if _is_numerically_one(value):
        # One is special because of singuglar/plural and the pronoun.
        continue

      identifier, reference_type = _to_identifier(text, value)

      _add_identifier(
          identifier,
          reference_type,
          cell.text,
          references,
          row_index,
          column_index,
      )


def _get_question_references(
    question):
  """Converts numeric and entity annotations in question to references."""
  references = {}

  spans = number_utils.parse_text(
      text_utils.normalize_for_match(question.original_text))

  for span in spans:
    for value in span.values:
      if _is_numerically_one(value):
        # One is special because of singular/plural and the pronoun.
        continue

      text = question.original_text[span.begin_index:span.end_index]
      identifier, reference_type = _to_identifier(text, value)

      _add_identifier(
          identifier,
          reference_type,
          text,
          references,
          span.begin_index,
          span.end_index,
      )
  annotated_text = question.Extensions[
      annotated_text_pb2.AnnotatedText.annotated_question_ext]
  for annotation in annotated_text.annotations:
    begin_index = annotation.begin_byte_index
    end_index = annotation.end_byte_index
    _add_identifier(
        annotation.identifier,
        ReferenceType.ENTITY,
        question.original_text[begin_index:end_index],
        references,
        begin_index,
        end_index,
    )
  return references


@dataclasses.dataclass(frozen=True)
class ContrastiveCandidate:
  text: Text
  focus_texts: Set[Text]
  negative_replacement_texts: Set[Text]
  positive_replacement_texts: Set[Text]
  support_texts: Set[Text]


def _only_occurs_in_a_single_column(reference):
  columns = {column for row, column in reference.coordinates}
  return len(columns) == 1


def get_interaction_references(
    interaction,):
  """Finds references in the question and the table."""
  cell_references = {}
  for row_index, row in enumerate(interaction.table.rows):
    for column_index, cell in enumerate(row.cells):
      _add_entity_from_cell(cell, cell_references, row_index, column_index)
      _add_numeric_reference_from_cell(cell, cell_references, row_index,
                                       column_index)
  logging.vlog(1, 'cell_references: %s', cell_references)

  # We don't allow the same identifier to occur in multiple columns.
  # We do allow multiple rows.
  cell_references = {
      identifier: reference
      for identifier, reference in cell_references.items()
      if _only_occurs_in_a_single_column(reference)
  }

  # Restrict to the question identifiers that occur in the table.
  question_references = []
  for question in interaction.questions:
    question_references.append({
        i: ref
        for i, ref in _get_question_references(question).items()
        if i in cell_references
    })

  return InteractionReferences(cell_references, question_references)


def get_contrastive_candidates(
    interaction):
  """Creates replacement candidates."""
  table = interaction.table
  if len(table.columns) < 2:
    return
  if len(table.rows) < 2:
    return

  interaction_references = get_interaction_references(interaction)

  identifiers_per_row = collections.defaultdict(set)
  identifiers_per_column = collections.defaultdict(set)
  for identifier, reference in interaction_references.table.items():
    for row_index, column_index in reference.coordinates:
      identifiers_per_row[row_index].add(identifier)
      identifiers_per_column[column_index].add(identifier)
  logging.vlog(1, 'identifiers_per_row: %s', identifiers_per_row)
  logging.vlog(1, 'identifiers_per_column: %s', identifiers_per_column)

  for question, references in zip(interaction.questions,
                                  interaction_references.question):

    logging.vlog(1, 'references: %s', references)

    identifiers = set(references)

    logging.vlog(1, 'identifiers: %s', identifiers)

    for identifier, reference in references.items():
      logging.vlog(1, 'identifier: %s', identifier)

      cell_reference = interaction_references.table[identifier]
      for row_index, column_index in cell_reference.coordinates:

        # Get ids in same row.
        cell_identifiers_in_row = identifiers_per_row[row_index]
        # Intersect with the ones that occur in the question.

        logging.vlog(1, 'cell_identifiers_in_row: %s', cell_identifiers_in_row)

        cell_identifiers_in_row = cell_identifiers_in_row & identifiers
        # Remove the identifier itself.
        cell_identifiers_in_row = cell_identifiers_in_row - {identifier}

        logging.vlog(1, 'cell_identifiers_in_row: %s', cell_identifiers_in_row)

        # Get ids in the same column.
        cell_identifiers_in_column = identifiers_per_column[column_index]
        # Remove the ones that occur in the question.
        cell_identifiers_in_column = cell_identifiers_in_column - identifiers
        # Remove the ones that have a different reference_type.
        focus_reference_type = identifier[1]
        cell_identifiers_in_column = {(i, rt)
                                      for i, rt in cell_identifiers_in_column
                                      if rt == focus_reference_type}

        logging.vlog(1, 'cell_identifiers_in_column: %s',
                     cell_identifiers_in_column)

        if not cell_identifiers_in_column:
          # There is no id in the same column that doesn't occur in the
          # question.
          continue

        focus_texts = set(reference.span_texts)
        negative_replacement_texts = set()
        for i in cell_identifiers_in_column:
          negative_replacement_texts.update(
              interaction_references.table[i].span_texts)
        support_texts = set()
        for i in cell_identifiers_in_row:
          support_texts.update(references[i].span_texts)

        # Positive replacement text is how the focus entity is mentioned in
        # the table this can be different from how it is mentioned in the text.
        positive_replacements_texts = (
            interaction_references.table[identifier].span_texts)
        focus_texts.update(positive_replacements_texts)

        # Sanity checks that might occur if span texts are ambiguous.
        if focus_texts & negative_replacement_texts:
          continue

        if focus_texts & support_texts:
          continue

        if negative_replacement_texts & support_texts:
          continue

        yield ContrastiveCandidate(
            text=question.original_text,
            focus_texts=focus_texts,
            positive_replacement_texts=positive_replacements_texts,
            negative_replacement_texts=negative_replacement_texts,
            support_texts=support_texts,
        )


@dataclasses.dataclass(frozen=True)
class RewriteResult:
  has_support: bool
  text: Text
  num_context_tokens: int


class Rewriter:
  """Responsible for replacing references in text."""

  def __init__(self):
    self._tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    self._detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()

  def _contains_phrase(
      self,
      text,
      phrases,
  ):
    for phrase in phrases:
      phrase = self._normalize(phrase)
      if phrase in text:
        return True
    return False

  def _normalize(self, text):
    tokens = self._tokenizer.tokenize(text)
    normalized_text = ' ' + ' '.join(tokens) + ' '
    return normalized_text

  def _token_set(self, text):
    tokens = self._tokenizer.tokenize(text)
    return set(tokens)

  def _denormalize(self, text):
    return self._detokenizer.detokenize(text.split())

  def get_random_positive_statement(
      self,
      rng,
      candidate,
      text,
  ):
    return self.get_random_contrastive_statement(
        rng,
        candidate,
        text,
        is_negative_example=False,
    )

  def get_random_contrastive_statement(
      self,
      rng,
      candidate,
      text,
      is_negative_example = True,
  ):
    """Emits pairs of original and contrastive statement."""
    original_text = self._normalize(text)
    text = original_text

    if not self._contains_phrase(text, candidate.focus_texts):
      # Cannot locate focus text.
      return None

    replacement_texts = (
        candidate.negative_replacement_texts
        if is_negative_example else candidate.positive_replacement_texts)
    replacement_text = rng.choice(sorted(replacement_texts))
    replacement_text = self._normalize(replacement_text)

    for focus_text in candidate.focus_texts:
      focus_text = self._normalize(focus_text)
      text = text.replace(focus_text, replacement_text)

    # Compute set of tokens that have not been replaced.
    context_tokens = self._token_set(text) & self._token_set(original_text)

    has_support = self._contains_phrase(text, candidate.support_texts)
    return RewriteResult(
        has_support=has_support,
        text=self._denormalize(text),
        num_context_tokens=len(context_tokens),
    )


def get_contrastive_statements(
    rng,
    interaction,
    min_num_context_tokens = 4,
    count_fn = lambda m: None,
):
  """Creates pairs of original and contrastive statements."""
  candidates = get_contrastive_candidates(interaction)

  rewrite = Rewriter()

  for candidate in candidates:
    count_fn('Candidates')
    for paragraph in candidate.text.split('\n'):
      count_fn('Paragraphs')
      statements = sentence_tokenizer.tokenize(paragraph)
      for statement in statements:
        count_fn('Statements')
        statement = statement.strip()
        result = rewrite.get_random_contrastive_statement(
            rng,
            candidate,
            statement,
        )
        if result is not None:
          if result.num_context_tokens < min_num_context_tokens:
            count_fn('Statements with too few context tokens')
            continue
          count_fn('Statements rewritten')
          if result.has_support:
            count_fn('Statements rewritten with support')
          else:
            count_fn('Statements rewritten without support')

          positive_result = rewrite.get_random_positive_statement(
              rng,
              candidate,
              statement,
          )
          if positive_result is None:
            count_fn('ERROR: Cannot create positive result.')
            # Unexpected since we could create the negative result.
            continue
          yield result.has_support, positive_result.text, result.text
        else:
          count_fn('Statements that cannot be rewritten')
