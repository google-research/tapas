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
"""Converts retrieval examples to QA examples.

Uses the baseline to create negative candidates.

"""

import collections
import json
import os
from typing import Iterable, List, Mapping, Optional, Set, Text, Tuple

from absl import app
from absl import flags
from tapas.protos import interaction_pb2
from tapas.protos import retriever_info_pb2
from tapas.retrieval import tfidf_baseline_utils
from tapas.scripts import prediction_utils
from tapas.scripts import preprocess_nq_utils
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf
import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None, "Interaction protos in tfrecord format.")

flags.DEFINE_string("table_file", None, "Table protos in tfrecord format.")

flags.DEFINE_string("output_dir", None,
                    "Dir where interactions will be written to.")

flags.DEFINE_string("index_files_pattern", None,
                    "JSONL file pattern containing the most similar tables.")

flags.DEFINE_integer("max_rank", 10, "Max rank to consider.")

flags.DEFINE_integer("title_multiplicator", 15, "See create_bm25_index.")

flags.DEFINE_bool("oracle_retrieval", False, "Always add correct table.")

flags.DEFINE_bool(
    "add_negatives",
    True,
    "Only add correctly retrieved tables.",
)


def _try_to_set_answer(
    table,
    answer_groups,
    new_question,
):
  """Try to find answer in table."""

  coordinates_to_answers = []
  other_answers = []
  for answers in answer_groups:
    # Try to find answer in table.
    answer_coordinates = []
    for answer in answers:
      coords = list(
          preprocess_nq_utils.find_answer(
              table,
              answer,
              search_in_header=False,
          ))
      if not coords:
        answer_coordinates = []
        break
      answer_coordinates.extend(coords)
    if not answer_coordinates:
      other_answers.append(answers)
    else:
      answer_coordinates.sort()
      for coordinates in answer_coordinates:
        coordinates_to_answers.append((coordinates, answers))
        break

  coordinates_to_answers.sort()
  answer_groups = [answers for _, answers in coordinates_to_answers]
  answer_groups.extend(other_answers)
  if answer_groups:
    new_question.answer.answer_texts.extend(answer_groups[0])
    for answers in answer_groups[1:]:
      new_question.alternative_answers.add().answer_texts.extend(answers)


def get_answer_texts(
    questions,):
  """Get all answer texts."""
  answers = set()

  def get_answers(question, answer):
    if not answer.answer_texts:
      raise ValueError(f"Question without answer: {question}")
    answer_texts = tuple(sorted(answer.answer_texts))
    answers.add(answer_texts)

  for question in questions:
    get_answers(question, question.answer)
    for answer in question.alternative_answers:
      get_answers(question, answer)

  return answers


def _get_neural_nearest_neighbors(
    file_pattern):
  """Read jsonl file with nearest neighbors according to neural model."""
  results = {}
  for path in tf.io.gfile.glob(file_pattern):
    with tf.io.gfile.GFile(path) as f:
      for json_line in f:
        line = json.loads(json_line)
        query_id = line["query_id"]
        assert query_id not in results
        results[query_id] = [
            (item["table_id"], item["score"]) for item in line["table_scores"]
        ]
  return results


def _get_table_rank_and_score(
    scored_hits,
    table_id,
):
  """Returns rank and score of 'table_id'."""
  for rank, (current_table_id, score) in enumerate(scored_hits):
    if current_table_id == table_id:
      return rank + 1, score
  return None


def _set_retriever_info(
    question,
    scored_hits,
    table_id,
):
  """Sets the basic retriever info."""
  result = _get_table_rank_and_score(scored_hits, table_id)
  if result is None:
    return
  rank, score = result
  ext = question.Extensions[retriever_info_pb2.RetrieverInfo.question_ext]
  ext.rank = rank
  ext.score = score


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print("Creating output dir ...")
  tf.io.gfile.makedirs(FLAGS.output_dir)

  interaction_files = []
  for filename in tf.io.gfile.listdir(FLAGS.input_dir):
    interaction_files.append(os.path.join(FLAGS.input_dir, filename))

  tables = {}
  if FLAGS.table_file:
    print("Reading tables ...")
    tables.update({
        table.table_id: table
        for table in tfidf_baseline_utils.iterate_tables(FLAGS.table_file)
    })

  print("Adding interactions tables ...")
  for interaction_file in interaction_files:
    interactions = prediction_utils.iterate_interactions(interaction_file)
    for interaction in interactions:
      tables[interaction.table.table_id] = interaction.table

  print("Creating index ...")

  if FLAGS.index_files_pattern:
    neighbors = _get_neural_nearest_neighbors(FLAGS.index_files_pattern)
    retrieve_fn = lambda question: neighbors.get(question.id, [])
  else:
    index = tfidf_baseline_utils.create_bm25_index(
        tables=tables.values(),
        title_multiplicator=FLAGS.title_multiplicator,
        num_tables=len(tables),
    )
    retrieve_fn = lambda question: index.retrieve(question.original_text)

  print("Processing interactions ...")
  for interaction_file in interaction_files:
    interactions = list(prediction_utils.iterate_interactions(interaction_file))

    examples = collections.defaultdict(list)
    for interaction in interactions:
      example_id, _ = preprocess_nq_utils.parse_interaction_id(interaction.id)
      examples[example_id].append(interaction)

    filename = os.path.basename(interaction_file)
    is_train = "train" in filename
    output = os.path.join(FLAGS.output_dir, filename)
    with tf.io.TFRecordWriter(output) as writer:
      num_correct = 0
      with tqdm.tqdm(
          examples.items(),
          total=len(examples),
          desc=filename,
          postfix=[{
              "prec": "0.00",
              "multiple_tables": 0,
              "multiple_answers": 0,
              "no_hits": 0,
          }],
      ) as pbar:
        for nr, example in enumerate(pbar):
          example_id, interaction_list = example

          questions = []
          for interaction in interaction_list:
            if len(interaction.questions) != 1:
              raise ValueError(f"Unexpected question in {interaction}")
            questions.append(interaction.questions[0])

          answers = get_answer_texts(questions)

          if len(set(q.original_text for q in questions)) != 1:
            raise ValueError(f"Different questions {questions}")
          question_text = questions[0].original_text
          scored_hits = retrieve_fn(questions[0])
          if not scored_hits:
            pbar.postfix[0]["no_hits"] += 1
          candidate_hits = scored_hits[:FLAGS.max_rank]

          correct_table_ids = {
              interaction.table.table_id for interaction in interaction_list
          }

          table_ids = {table_id for table_id, _ in candidate_hits}

          if correct_table_ids & table_ids:
            num_correct += 1
          prec = num_correct / (nr + 1)
          pbar.postfix[0]["prec"] = f"{prec:.2f}"
          if len(correct_table_ids) > 1:
            pbar.postfix[0]["multiple_tables"] += 1

          if is_train or FLAGS.oracle_retrieval:
            table_ids.update(correct_table_ids)

          for table_index, table_id in enumerate(sorted(table_ids)):
            table = tables[table_id]
            new_interaction = interaction_pb2.Interaction()
            new_interaction.table.CopyFrom(table)
            new_question = new_interaction.questions.add()
            new_question.original_text = question_text
            _try_to_set_answer(table, answers, new_question)
            _set_retriever_info(new_question, scored_hits, table_id)
            new_question.answer.is_valid = True
            if new_question.alternative_answers:
              pbar.postfix[0]["multiple_answers"] += 1
            if table_id in correct_table_ids:
              new_question.answer.class_index = 1
            else:
              new_question.answer.class_index = 0
              if not FLAGS.add_negatives:
                continue
            new_interaction.id = text_utils.get_sequence_id(
                example_id, str(table_index))
            new_question.id = text_utils.get_question_id(
                new_interaction.id, position=0)
            writer.write(new_interaction.SerializeToString())


if __name__ == "__main__":
  app.run(main)
