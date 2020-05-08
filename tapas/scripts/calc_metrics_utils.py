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
"""Denotation accuracy calculation for TAPAS predictions over WikiSQL."""

import math
import os
from typing import Any, Dict, List, Optional, Set, Text, Tuple

from absl import logging
import dataclasses
import pandas as pd
import sklearn.metrics
from tapas.protos import interaction_pb2
from tapas.scripts import prediction_utils
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf


_Answer = interaction_pb2.Answer


@dataclasses.dataclass
class Example:
  """Represents an example."""
  example_id: Text
  question: Text
  table_id: Text
  table: pd.DataFrame
  gold_cell_coo: Set[Tuple[int, int]]
  gold_agg_function: int
  float_answer: float
  has_gold_answer: bool
  pred_cell_coo: Set[Tuple[int, int]] = dataclasses.field(default_factory=set)
  pred_agg_function: int = _Answer.NONE


def read_data_examples_from_interactions(
    interactions_path):
  """Reads examples from an interactions file."""
  data_examples = {}
  for interaction in prediction_utils.iterate_interactions(interactions_path):
    for question in interaction.questions:
      data_examples[question.id] = example_from_question(interaction, question)
  return data_examples


def example_from_question(
    interaction,
    question,
):
  """Converts question to example."""
  ex_id = question.id
  question_text = question.original_text
  table = prediction_utils.table_to_panda_frame(interaction.table)
  table_id = interaction.table.table_id
  has_gold_answer = question.answer.is_valid

  gold_cell_coo = {
      (x.row_index, x.column_index) for x in question.answer.answer_coordinates
  }
  gold_agg_function = question.answer.aggregation_function
  float_value = question.answer.float_value if question.answer.HasField(
      'float_value') else None
  ex = Example(
      ex_id,
      question_text,
      table_id,
      table,
      gold_cell_coo,
      gold_agg_function,
      float_value,
      has_gold_answer,
  )
  return ex


def read_predictions(predictions_path, examples):
  """Reads predictions from a csv file."""
  for row in prediction_utils.iterate_predictions(predictions_path):
    pred_id = '{}-{}_{}'.format(row['id'], row['annotator'], row['position'])
    example = examples[pred_id]
    example.pred_cell_coo = prediction_utils.parse_coordinates(
        row['answer_coordinates'])
    example.pred_agg_function = int(row.get('pred_aggr', '0'))


def _calc_acc(correct):
  """Returns the fraction of true examples."""
  matches = sum(1 for ex in correct if ex)
  return matches / float(len(correct))


@dataclasses.dataclass(frozen=True)
class StructuredMetrics:
  aggregation_acc: float
  cell_acc: float
  joint_acc: float
  confusion_df: pd.DataFrame
  f1_scores_df: pd.DataFrame


def calc_structure_metrics(
    examples,
    denotation_errors_path = None):
  """Calculates metrics regarding the correctness of the predicted structure."""

  examples_to_write = []
  for ex_id in examples:
    pred_agg_function = examples[ex_id].pred_agg_function
    pred_cell_coo = examples[ex_id].pred_cell_coo

    gold_agg_function = examples[ex_id].gold_agg_function
    gold_cell_coo = examples[ex_id].gold_cell_coo

    assert pred_agg_function is not None
    assert pred_cell_coo is not None

    agg_function_correct = gold_agg_function == pred_agg_function
    coo_correct = gold_cell_coo == pred_cell_coo
    is_correct = agg_function_correct and coo_correct

    examples_to_write.append([
        ex_id,
        gold_agg_function,
        pred_agg_function,
        sorted(gold_cell_coo),
        sorted(pred_cell_coo),
        agg_function_correct,
        coo_correct,
        is_correct,
    ])

  frame = pd.DataFrame(
      examples_to_write,
      columns=[
          'id',
          'gold_agg',
          'pred_agg',
          'gold_cell_coo',
          'pred_cell_coo',
          'agg_function_correct',
          'coo_correct',
          'is_correct',
      ])

  aggregation_acc = frame['agg_function_correct'].mean()
  logging.info('aggregation_acc=%f', aggregation_acc)
  cell_acc = frame['coo_correct'].mean()
  logging.info('cell_acc=%f', cell_acc)
  joint_acc = frame['is_correct'].mean()
  logging.info('joint_acc=%f', joint_acc)

  agg_labels = list(_Answer.AggregationFunction.keys())

  gold_agg = frame['gold_agg']
  pred_agg = frame['pred_agg']

  confusion_mat = sklearn.metrics.confusion_matrix(gold_agg, pred_agg)
  confusion_df = pd.DataFrame(
      data=confusion_mat,
      columns=['pred_{}'.format(l) for l in agg_labels],
      index=['gold_{}'.format(l) for l in agg_labels])
  logging.info('*** Aggregation confusion matrix ***')
  logging.info('\n%s', confusion_df)

  f1_scores = [sklearn.metrics.f1_score(gold_agg, pred_agg, average=None)]
  f1_scores_df = pd.DataFrame(data=f1_scores, columns=agg_labels)
  logging.info('*** Aggregation F1 scores ***')
  logging.info('\n%s', f1_scores_df)

  if denotation_errors_path:
    with tf.io.gfile.GFile(
        os.path.join(denotation_errors_path, 'structured_examples.tsv'),
        'w') as f:
      frame.to_csv(f, sep='\t')

  return StructuredMetrics(
      aggregation_acc=aggregation_acc,
      cell_acc=cell_acc,
      joint_acc=joint_acc,
      confusion_df=confusion_df,
      f1_scores_df=f1_scores_df,
  )


def _collect_cells_from_table(cell_coos,
                              table):
  cell_values = []
  for cell in cell_coos:
    value = str(table.iat[cell[0], cell[1]])
    cell_values.append(value)
  return cell_values


def _safe_convert_to_float(value):
  float_value = text_utils.convert_to_float(value)
  if math.isnan(float_value):
    raise ValueError('Value is NaN %s' % value)
  return float_value


def _parse_value(value):
  """Parses a cell value to a number or lowercased string."""
  try:
    return _safe_convert_to_float(value)
  except ValueError:
    try:
      return value.lower()
    except ValueError:
      return value


def _to_float32s(elements):
  return tuple(text_utils.to_float32(v) for v in elements)


def execute(aggregation_type, cell_coos,
            table):
  """Executes predicted structure against a table to produce the denotation."""
  values = _collect_cells_from_table(cell_coos, table)
  values_parsed = [_parse_value(value) for value in values]
  values_parsed = tuple(values_parsed)
  if aggregation_type == _Answer.NONE:
    # In this case there is no aggregation
    return values_parsed, values
  else:  # Should perform aggregation.
    if not values and (aggregation_type == _Answer.AVERAGE or
                       aggregation_type == _Answer.SUM):
      # Summing or averaging an empty set results in an empty set.
      # NB: SQL returns null for sum over an empty set.
      return tuple(), values
    if aggregation_type == _Answer.COUNT:
      denotation = len(values)
    else:
      # In this case all values must be numbers (to be summed or averaged).
      try:
        values_num = [text_utils.convert_to_float(value) for value in values]
      except ValueError:
        return values_parsed, values
      if aggregation_type == _Answer.SUM:
        denotation = sum(values_num)
      elif aggregation_type == _Answer.AVERAGE:
        denotation = sum(values_num) / len(values_num)
      else:
        raise ValueError('Unknwon aggregation type: %s' % aggregation_type)
    return tuple([float(denotation)]), values


@dataclasses.dataclass
class DenotationResult:
  denotation: Optional[List[Any]]
  values: Optional[List[Text]]
  agg_function: Optional[int]
  cell_coordinates: Optional[Set[Tuple[int, int]]]


@dataclasses.dataclass
class DenotationStats:
  """Represent the denotation evaluation for a single example."""
  is_correct: bool
  pred_result: DenotationResult
  gold_result: Optional[DenotationResult]


def _highlight_cells(coordinates, table):
  """Returns a printable version of the table with highlighted cells."""
  result = table.copy()[list(table)].astype(str)
  for x, y in coordinates:
    result.iat[x, y] = '[[' + str(table.iat[x, y]) + ']]'
  return result


def _get_debug_row(result, table):
  if not result:
    return [None, None, None, None]
  return [
      result.denotation,
      result.values,
      _Answer.AggregationFunction.Name(result.agg_function),
      sorted(result.cell_coordinates) if result.cell_coordinates else None,
      _highlight_cells(result.cell_coordinates, table)
      if result.cell_coordinates else None,
  ]


def _get_gold_denotation_result(example):
  """Computes gold denotation of the example."""
  if not example.has_gold_answer:
    # No gold answer for this example.
    return None
  agg_function = example.gold_agg_function
  cell_coo = example.gold_cell_coo
  if example.float_answer is None:
    denotation, values = execute(agg_function, cell_coo, example.table)
  elif math.isnan(example.float_answer):
    denotation = []
    values = []
  else:
    denotation = [(example.float_answer)]
    values = []
  denotation = _to_float32s(denotation)
  denotation = text_utils.normalize_answers(denotation)
  return DenotationResult(
      denotation=denotation,
      values=values,
      agg_function=agg_function,
      cell_coordinates=cell_coo,
  )


def _get_pred_denotation_result(example):
  """Computes predicted denotation."""
  if example.pred_agg_function is None:
    raise ValueError('pred_agg_function is None')
  if example.pred_cell_coo is None:
    raise ValueError('pred_cell_coo is None')

  agg_function = example.pred_agg_function
  cell_coo = example.pred_cell_coo
  denotation, values = execute(agg_function, cell_coo, example.table)
  denotation = _to_float32s(denotation)
  denotation = text_utils.normalize_answers(denotation)
  return DenotationResult(
      denotation=denotation,
      values=values,
      agg_function=agg_function,
      cell_coordinates=cell_coo,
  )


def get_denotation_stats(example):
  """Computes denotation stats for single example."""
  pred_result = _get_pred_denotation_result(example)
  gold_result = _get_gold_denotation_result(example)
  is_correct = False
  if gold_result is not None:
    is_correct = pred_result.denotation == gold_result.denotation

  return DenotationStats(
      is_correct=is_correct,
      gold_result=gold_result,
      pred_result=pred_result,
  )


def calc_denotation_accuracy(examples,
                             denotation_errors_path,
                             predictions_file_name):
  """Calculates the denotation accuracy."""
  examples_to_write = []
  for example_id, example in sorted(examples.items()):
    denotation_stats = get_denotation_stats(example)
    examples_to_write.append(
        [example_id, example.question, denotation_stats.is_correct] +
        _get_debug_row(denotation_stats.gold_result, example.table) +
        _get_debug_row(denotation_stats.pred_result, example.table))

  frame = pd.DataFrame(
      examples_to_write,
      columns=[
          'example_id',
          'question',
          'is_correct',
          'gold denotation',
          'gold cell values',
          'gold cell coordinates',
          'gold aggregation',
          'gold table',
          'pred denotation',
          'pred cell values',
          'pred cell coordinates',
          'pred aggregation',
          'pred table',
      ])

  if denotation_errors_path is not None:
    examples_file = os.path.join(
        denotation_errors_path,
        'denotation_examples_{}'.format(predictions_file_name))
    with tf.io.gfile.GFile(examples_file, 'w') as f:
      frame.to_csv(f, sep='\t')

  denotation_acc = frame['is_correct'].mean()
  logging.info('denotation_accuracy=%f', denotation_acc)
  logging.info('total_test_examples=%d', len(examples))
  return denotation_acc
