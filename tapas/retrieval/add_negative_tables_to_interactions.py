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
"""Joins the negative tables with the interactions.

1.1/ Reads tables from tables file and returns (table_id, table).

1.2/ Reads json file to the negative tables. For each question return
     (negative_table_id,
      (question_id, negative_table_score, negative_table_rank)
     ).

2.1/ Joins the tables and json data (1.1 and 1.2) grouped by table_id.
     Returns (question_id,
              (negative_table, negative_table_score, negative_table_rank)
             ).

2.2/ Reads interactions and return (question_id, interaction).

3.1/ Joins the interactions with the negatif tables (2.1 and 2.2) grouped by
     question_id. Returns the new interaction with a list of negative tables.
     The negative table is different from the interaction's table.
     For each interaction we select at most number of negative table equals
     max_num_negatives.
     We select the highest scored ones (The most negative score: as the
     smilarity in the json files is more negative for a higher similarit.)
     that doesn't contain the answer.
     The new interactions contains a positive score: -1 * json_score to report
     positive scores.
"""

import json
import operator
import os
from typing import Any, Dict, Iterable, List, Optional, Text, Tuple

import apache_beam as beam
from tapas.protos import interaction_pb2
from tapas.protos import negative_retrieval_examples_pb2
from tapas.scripts import preprocess_nq_utils
from tapas.utils import pretrain_utils

_NS = "main"
_NegativeRetrievalExamples = negative_retrieval_examples_pb2.NegativeRetrievalExamples
_NegativeRetrievalExample = negative_retrieval_examples_pb2.NegativeRetrievalExample
_NE_Ext = _NegativeRetrievalExamples.negative_retrieval_examples_ext


def _merge_tables_ids_questions(
    key_join
):
  """Joins the tables with the jsonl question ids."""
  key, join = key_join
  if not join["tables"]:
    raise ValueError("Table Id not found")
  if len(join["tables"]) > 1:
    beam.metrics.Metrics.counter(_NS,
                                 "DulicatedTabelsIds_Tables_tfrecord").inc()
  if join["questions"]:
    for q_id, score, rank in join["questions"]:
      beam.metrics.Metrics.counter(_NS, "NumTableQuestionsFound_Merge").inc()
      yield q_id, (join["tables"][0], score, rank)


def _merge_tables_interactions(
    key_join,
    max_num_negatives,
):
  """Joins the interactions and multiple similar table id by question id.

  Args:
    key_join: Input to merge
    max_num_negatives: Max similar tables to add. None means no limit.

  Yields:
    Merged interactions.
  """
  _, join = key_join
  if len(join["interactions"]) > 1:
    beam.metrics.Metrics.counter(
        _NS, "DulicatedQuestionIds_Interactions_" +
        str(len(join["interactions"]))).inc()
  elif not join["interactions"]:
    beam.metrics.Metrics.counter(_NS,
                                 "QuestionIds_WithoutInteraction_Jsonl").inc()
  if join["interactions"]:
    interaction = join["interactions"][0]
    tables = join["tables"]
    sorted_tables = sorted(tables, key=lambda t: t[2])

    table_ids = set()
    true_negatives = []
    for table, score, rank in sorted_tables:
      if max_num_negatives is not None:
        if len(true_negatives) >= max_num_negatives:
          break
      if table.table_id in table_ids:
        continue
      table_ids.add(table.table_id)
      if table.table_id == interaction.table.table_id:
        continue
      if preprocess_nq_utils.table_contains_all_answers(
          table, interaction.questions[0]):
        continue
      true_negatives.append(_create_negative_example(table, score, rank))

    if not true_negatives:
      # Make sure we don't drop interactions.
      beam.metrics.Metrics.counter(_NS, "Interactions_WitFakeTable").inc()
      fake_table = interaction_pb2.Table()
      fake_table.table_id = "FAKE"
      fake_table.columns.add()
      fake_table.rows.add().cells.add()
      true_negatives.append(_create_negative_example(fake_table, 0.0, 0))

    if true_negatives:
      beam.metrics.Metrics.counter(_NS, "Interaction_With_True_negatives").inc()
      yield _create_interaction_with_negative_tables(interaction,
                                                     true_negatives)
    else:
      beam.metrics.Metrics.counter(_NS,
                                   "Interaction_Without_True_negatives").inc()


def _create_negative_example(
    table,
    score,
    rank,
):
  """Creates a negative example."""
  example = _NegativeRetrievalExample()
  example.type = _NegativeRetrievalExample.Type.BASELINE
  # The expected similarity score in the proto is more positive for a higher
  # similarity. Therefore we report -scores.
  example.score = -score
  example.rank = rank
  example.table.CopyFrom(table)
  return example


def _create_interaction_with_negative_tables(
    interaction,
    true_negatives,
):
  """Adds the negative table to the interaction."""
  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(interaction)
  q = new_interaction.questions[0]
  for true_negative in true_negatives:
    examples = q.Extensions[_NE_Ext].examples
    new_example = examples.add()
    new_example.CopyFrom(true_negative)
  return new_interaction


def _get_question_id_interaction(
    interaction,
):
  """Extracts one interaction per question."""
  if len(interaction.questions) != 1:
    raise ValueError("All interactions should have exactly one question.")
  beam.metrics.Metrics.counter(_NS, "NumInteractions_Interactions").inc()
  return interaction.questions[0].id.replace("-", "_"), interaction


def _get_similar_table_id_question(
    line):
  """Reads json data: extracts the question id and similar tables."""
  data = json.loads(line)
  question_id = data["query_id"]
  # Replace "-" by "_" to make the different files compatile.
  question_id = question_id.replace("-", "_")
  table_scores = data["table_scores"]
  for rank, t in enumerate(table_scores):
    beam.metrics.Metrics.counter(_NS, "NumQuestions_jsonl").inc()
    yield t["table_id"], (question_id, t["score"], rank)


def _get_table_id_table(
    table,):
  """Reads table_id, table."""
  beam.metrics.Metrics.counter(_NS, "NumTables_Tables").inc()
  return table.table_id, table


def add_negative_tables_to_interactions(
    max_num_negatives,
    input_interactions_files,
    input_json_files,
    input_tables_file,
    output_files,
):
  """Pipeline for adding the negative tables to the interactions."""
  if len(output_files) != len(input_interactions_files):
    raise ValueError("Size mismatch output and input interactons: %s %s" %
                     (output_files, input_interactions_files))

  if len(input_json_files) != len(input_interactions_files):
    raise ValueError("Size mismatch json and input interactons: %s %s" %
                     (input_json_files, input_interactions_files))

  def _pipeline(root):
    """Pipeline."""
    name = "tables"
    table_id_tables = (
        pretrain_utils.read_inputs(root, input_tables_file, name,
                                   proto_message=interaction_pb2.Table)
        | f"ReadTables_{name}" >> beam.Map(operator.itemgetter(1))
        | f"ReadTablesIdTables_{name}" >> beam.Map(_get_table_id_table,))

    for (input_file_interaction, input_file_json,
         output_file) in zip(input_interactions_files, input_json_files,
                             output_files):

      name = os.path.basename(input_file_interaction)

      similar_table_id_question = (
          root | f"Read_{name}" >> beam.io.textio.ReadFromText(
              input_file_json,
              coder=beam.coders.BytesCoder(),
          )
          | f"GetSimilarQuestionTables_{name}" >> beam.FlatMap(
              _get_similar_table_id_question,))

      q_id_interactions = (
          pretrain_utils.read_interactions(root, input_file_interaction, name)
          | f"ReadInteractions_{name}" >> beam.Map(operator.itemgetter(1))
          | f"ReadQuestionsInteractions_{name}" >> beam.Map(
              _get_question_id_interaction,))

      similar_q_id_tables = ({
          "tables": table_id_tables,
          "questions": similar_table_id_question
      }
                             | f"JoinByTableId_{name}" >> beam.CoGroupByKey()
                             | f"MergeTablesIds_{name}" >> beam.FlatMap(
                                 _merge_tables_ids_questions,))

      _ = ({
          "tables": similar_q_id_tables,
          "interactions": q_id_interactions
      }
           | f"JoinByQuestionId_{name}" >> beam.CoGroupByKey()
           | f"MergeTablesIteraction_{name}" >> beam.FlatMap(
               _merge_tables_interactions, max_num_negatives=max_num_negatives)
           | f"WriteNewInteractions_{name}" >> beam.io.WriteToTFRecord(
               output_file,
               shard_name_template="",
               coder=beam.coders.ProtoCoder(interaction_pb2.Interaction)))

  return _pipeline
