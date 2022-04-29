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
"""Helper methods for the HybridQA dataset."""

import collections
import dataclasses
import enum
import json
import os
import string
import tempfile
from typing import Any, Iterable, List, Mapping, Text, Optional, Tuple, Set, Union, MutableMapping
import urllib.parse
import zipfile

from absl import logging
import numpy as np
from tapas.protos import annotated_text_pb2
from tapas.protos import interaction_pb2
from tapas.scripts import prediction_utils
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf

_annotated_cell = annotated_text_pb2.AnnotatedText.annotated_cell_ext
_annotation_descriptions = annotated_text_pb2.AnnotationDescription.annotation_descriptions_ext
_ZIP_NAME = 'WikiTables-WithLinks'
_TRANSLATION_TABLE = str.maketrans(string.punctuation,
                                   ' ' * len(string.punctuation))
_ARTICLES = ('a', 'an', 'the')  # List from HybridQA/evaluate_script.py
_RECALL_KS = (1, 3, 5)

CoordinateType = Tuple[int, int]


class Split(enum.Enum):
  TRAIN = 'train'
  DEV = 'dev'
  TEST = 'test'


class AnswerType(enum.Enum):
  ALL = 'all'
  ONE_IN_TABLE = '1_in_table'
  ONE_IN_TEXT = '1_in_text'
  MANY_IN_TABLE = '2+_in_table'
  MANY_IN_TEXT = '2+_in_text'
  TABLE_AND_TEXT = 'table_&_text'
  NO_ANSWER = 'no_answer'


def get_hybridqa_references(
    references_path):
  """Returns HybridQA references json as a python dict."""

  # Example file at
  # https://github.com/wenhuchen/HybridQA/blob/master/released_data/dev_reference.json.
  with tf.io.gfile.GFile(references_path, 'r') as fp:
    references = json.load(fp)
  for key, values in tuple(references.items()):
    references[key] = set(values)
  return references


def _get_answer_type(question):
  """Compute a class of answer for a granular evluation of questions."""
  table_answers = len(question.alternative_answers[0].answer_coordinates)
  text_answers = len(question.answer.answer_coordinates) - table_answers
  if table_answers and text_answers:
    return AnswerType.TABLE_AND_TEXT
  elif not table_answers and not text_answers:
    return AnswerType.NO_ANSWER
  elif table_answers == 1:
    return AnswerType.ONE_IN_TABLE
  elif text_answers == 1:
    return AnswerType.ONE_IN_TEXT
  elif table_answers:
    return AnswerType.MANY_IN_TABLE
  else:
    return AnswerType.MANY_IN_TEXT


def get_best_cells(
    probabilities,):
  """Groups tokens by cell and returns cell sorted by descending score."""
  cell_probabilities = collections.defaultdict(list)
  for column, row, probability in probabilities:
    cell_probabilities[(row, column)].append(probability)
  items = sorted(
      cell_probabilities.items(),
      reverse=True,
      key=lambda pair: np.array(pair[1]).mean())
  return [coordinate for coordinate, _ in items]


@dataclasses.dataclass(frozen=True)
class CellSelectionMetrics:
  """Helper class that holds the metrics when training a cell selection model.

  recall: The fraction of the questions where a corrrect cell was selected.
  precision: On the average question where cell was selected, the fraction of
    selected cells that is correct.
  non_empty: The fraction of the questions where any cell was selected.
  answer_len: The average number of cells selected.
  coverage: The fraction of the questions where a prediction exists in the file.
  recall_at_1: The fraction of the questions where a corrrect cell was selected
    if the best cell is picked.
  recall_at_3: The fraction of the questions where a corrrect cell was selected
    if the best 3 cells are picked.
  recall_at_5: The fraction of the questions where a corrrect cell was selected
    if the best 5 cells are picked.
  """
  recall: float
  precision: Optional[float]
  non_empty: float
  answer_len: float
  coverage: float
  recall_at_1: Optional[float] = None
  recall_at_3: Optional[float] = None
  recall_at_5: Optional[float] = None


def get_cell_selection_metrics(
    interactions_file,
    predictions_file,
):
  """Evaluates cell selection results in HybridQA experiment."""
  questions = {}
  for interaction in prediction_utils.iterate_interactions(interactions_file):
    for question in interaction.questions:
      # Do not evaluate hidden test set examples
      if question.HasField('answer'):
        questions[question.id] = question
  if not questions:
    return {}
  cell_selection_metrics = dict(
      eval_cell_selection(questions, predictions_file))[AnswerType.ALL]
  return dataclasses.asdict(cell_selection_metrics)


def eval_cell_selection(
    questions,
    predictions_file,
):
  """Evaluates cell selection results in HybridQA experiment.

  Args:
    questions: A map of Question protos by their respective ids.
    predictions_file: Path to a tsv file with predictions for a checkpoint.

  Yields:
    An AnswerType and its corresponding CellSelectionMetrics instance
  """
  total = collections.Counter()
  total_correct = collections.Counter()
  total_correct_at_k = {k: collections.Counter() for k in _RECALL_KS}
  total_seen = collections.Counter()
  total_non_empty = collections.Counter()
  total_coordinates = collections.Counter()
  sum_precision = collections.defaultdict(float)

  for question in questions.values():
    for answer_type in [AnswerType.ALL, _get_answer_type(question)]:
      total[answer_type] += 1

  for row in prediction_utils.iterate_predictions(predictions_file):
    question = questions.get(row['question_id'])
    if question is None:
      # The dataset lost some examples after an update.
      continue
    gold_coordinates = {(x.row_index, x.column_index)
                        for x in question.answer.answer_coordinates}
    coordinates = prediction_utils.parse_coordinates(row['answer_coordinates'])
    # We only care about finding one correct cell for the downstream model.
    correct_coordinates = len(coordinates & gold_coordinates)
    has_probabilities = 'token_probabilities' in row
    if has_probabilities:
      best_cells = get_best_cells(json.loads(row['token_probabilities']))
      correct_at_k = {
          k: bool(set(best_cells[:k]) & gold_coordinates) for k in _RECALL_KS
      }
    else:
      correct_at_k = {}
    for answer_type in [AnswerType.ALL, _get_answer_type(question)]:
      total_coordinates[answer_type] += len(coordinates)
      total_correct[answer_type] += bool(correct_coordinates)
      total_seen[answer_type] += 1
      for k, correct in correct_at_k.items():
        total_correct_at_k[k][answer_type] += correct
      if coordinates:
        sum_precision[answer_type] += correct_coordinates / len(coordinates)
        total_non_empty[answer_type] += 1

  for answer_type in AnswerType:
    if total[answer_type]:
      recall_at_k = {
          f'recall_at_{k}': (total_correct_at_k[k][answer_type] /
                             total[answer_type]) if has_probabilities else None
          for k in _RECALL_KS
      }
      yield answer_type, CellSelectionMetrics(
          recall=total_correct[answer_type] / total[answer_type],
          non_empty=total_non_empty[answer_type] / total[answer_type],
          coverage=total_seen[answer_type] / total[answer_type],
          answer_len=total_coordinates[answer_type] / total[answer_type],
          precision=((sum_precision[answer_type] / total_non_empty[answer_type])
                     if total_non_empty[answer_type] else None),
          **recall_at_k,
      )


def _parse_cell(
    cell,
    text,
    links,
    descriptions,
):
  """Parse text and links in a table cell."""
  for link in collections.OrderedDict.fromkeys(links):
    cell_annotation = cell.Extensions[_annotated_cell].annotations.add()
    cell_annotation.identifier = urllib.parse.unquote(link)
    if cell_annotation.identifier not in descriptions:
      raise ValueError(f'{link} not found in requests')
  cell.text = text


def parse_table(
    json_dict,
    descriptions,
):
  """Converts Table in JSON format to Table proto."""
  table = interaction_pb2.Table()
  table.table_id = str(json_dict['uid'])
  table.document_title = json_dict['title']
  table.document_url = json_dict['url']

  for text, links in json_dict['header']:
    _parse_cell(table.columns.add(), text, links, descriptions)

  for row_data in json_dict['data']:
    row = table.rows.add()
    for text, links in row_data:
      _parse_cell(row.cells.add(), text, links, descriptions)

  return table


def _normalize(phrase):
  """Lower text and remove punctuation, articles and extra whitespace."""
  tokens = phrase.translate(_TRANSLATION_TABLE).lower().split()
  return ' '.join(token for token in tokens if token not in _ARTICLES)


def find_answer_coordinates(
    answer_text, table, desc_map
):
  """Returns coordinates and matched identifiers for an answer text in a table."""
  answer = _normalize(answer_text)
  all_coordinates = set()
  table_only_coordinates = []
  matched_identifiers = collections.Counter()

  for row_index, row in enumerate(table.rows):
    for column_index, cell in enumerate(row.cells):
      if text_utils.find_all_substrings(answer, _normalize(cell.text)):
        all_coordinates.add((row_index, column_index))
        table_only_coordinates.append((row_index, column_index))
      else:
        for annotation in cell.Extensions[_annotated_cell].annotations:
          if text_utils.find_all_substrings(
              answer, _normalize(desc_map[annotation.identifier])):
            all_coordinates.add((row_index, column_index))
            matched_identifiers[annotation.identifier] += 1
            break

  return all_coordinates, table_only_coordinates, matched_identifiers


def find_dataset_coordinates(
    example_json_or_question
):
  """Yields tuples of coordinates from a dataset example.

  Args:
    example_json_or_question: Input example either a json dict of an example or
      a Question proto.
  """
  if isinstance(example_json_or_question, Mapping):
    example_json = example_json_or_question  # type: Mapping
    for _, coordinates, _, _ in example_json.get('answer-node', []):
      # The answer text may appear at this cell, or in paragraph linked here.
      yield tuple(coordinates)

  else:  # isinstance of interaction_pb2.Question
    question = example_json_or_question  # type: interaction_pb2.Question
    for coordinate in question.answer.answer_coordinates:
      yield coordinate.row_index, coordinate.column_index


def _parse_interaction(
    table,
    descriptions,
    example,
    counters,
):
  """Converts a single example to an interaction with a single question.

  Args:
    table: Table proto for this interaction.
    descriptions: The Wikipedia intro for each entity in the Table annotations.
    example: Question parsed from input JSON file.
    counters: Used for logging events as the interactions are parsed.

  Returns:
    Interaction proto.
  """

  interaction = interaction_pb2.Interaction()

  # We append -0 that corresponds to position annotator field
  interaction.id = example['question_id'] + '-0'
  interaction.table.CopyFrom(table)

  desc_map = interaction.Extensions[_annotation_descriptions].descriptions
  for key, value in descriptions.items():
    desc_map[key] = value

  question = interaction.questions.add()
  # We append _0 that corresponds to SQA position field
  question.id = f'{interaction.id}_0'
  question.original_text = example['question']

  # Reference answer for the question. The test set answers are hidden.
  if 'answer-text' in example:
    true_coordinates, table_only_coordinates, matched_identifiers = find_answer_coordinates(
        example['answer-text'], table, desc_map)

    question.answer.answer_texts.append(example['answer-text'])

    # We use this field to store just the table answers
    table_only_answer = question.alternative_answers.add()
    for row_index, column_index in table_only_coordinates:
      table_only_answer.answer_coordinates.add(
          row_index=row_index, column_index=column_index)

    for row_index, column_index in true_coordinates:
      question.answer.answer_coordinates.add(
          row_index=row_index, column_index=column_index)

    # This is used to compare the examples we find against the ones in the data.
    dataset_coordinates = frozenset(find_dataset_coordinates(example))

    if true_coordinates > dataset_coordinates:
      counters['Missing answers in dataset'] += 1
    elif true_coordinates < dataset_coordinates:
      counters['Missing answers in extraction'] += 1
    elif true_coordinates == dataset_coordinates:
      counters['Same answers'] += 1
    else:
      counters['Disjoint answers'] += 1

    counters[f'Answer type is {_get_answer_type(question).value}'] += 1

    if any(count > 1 for count in matched_identifiers.values()):
      counters['Answers in repeated identifier'] += 1

    if len(true_coordinates) > 1:
      counters['Multiple answers'] += 1

  return interaction


def extract_zip_archive(zip_path):
  tmpdir = tempfile.mkdtemp()
  with tf.io.gfile.GFile(zip_path, 'rb') as raw_zip:
    with zipfile.ZipFile(raw_zip) as zip_object:
      zip_object.extractall(tmpdir)
  return tmpdir


def read_json_directory(path):
  """Read all files in directory and load them as JSON."""
  result = {}
  for filename in tf.io.gfile.listdir(path):
    filepath = os.path.join(path, filename)
    with tf.io.gfile.GFile(filepath, 'r') as filehandle:
      json_map = json.load(filehandle)
      key = os.path.splitext(filename)[0]  # Removing the extension '.json'
      if key in result:
        raise ValueError(f'Duplicate id: {key}')
      result[key] = json_map
  return result


def _convert_split(
    split, main_path, table_dict,
    descriptions_dict
):
  """Convert a single JSON file to Interaction protos."""
  filename = f'{split.value}.json'
  if split in {Split.TRAIN, Split.DEV}:
    filename = filename.replace('.json', '.traced.json')
  file_path = os.path.join(main_path, filename)
  with tf.io.gfile.GFile(file_path, 'r') as input_file:
    examples = json.load(input_file)
  counters = collections.Counter()
  for example in examples:
    table_id = str(example['table_id'])
    interaction = _parse_interaction(
        table=table_dict[table_id],
        descriptions=descriptions_dict[table_id],
        example=example,
        counters=counters,
    )
    yield interaction
  logging.info('Counters for %s: %s', split.value, counters)


def convert(
    input_dir):
  """Reads examples and table in JSON format and converts to interactions."""
  tmpdir = extract_zip_archive(os.path.join(input_dir, f'{_ZIP_NAME}.zip'))

  # For each table, the Wikipedia intro of each entity linked to the table.
  descriptions_dir = os.path.join(tmpdir, _ZIP_NAME, 'request_tok')
  descriptions = read_json_directory(descriptions_dir)

  table_dir = os.path.join(tmpdir, _ZIP_NAME, 'tables_tok')
  tables = {
      key: parse_table(json_map, descriptions[key])
      for key, json_map in read_json_directory(table_dir).items()
  }

  main_path = os.path.join(input_dir, 'released_data')
  return {
      split.value: _convert_split(split, main_path, tables, descriptions)
      for split in Split
  }
