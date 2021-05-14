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
"""Uses the baseline to create nearest neighbor results."""

import collections
import json
import os

from absl import app
from absl import flags
from tapas.retrieval import tfidf_baseline_utils
from tapas.scripts import prediction_utils
from tapas.scripts import preprocess_nq_utils
import tensorflow.compat.v1 as tf
import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None, "Interaction protos in tfrecord format.")

flags.DEFINE_string("table_file", None, "Table protos in tfrecord format.")

flags.DEFINE_string("output_dir", None,
                    "Directory where interactions will be written to.")

flags.DEFINE_integer("max_rank", None, "Max rank to consider.")

flags.DEFINE_integer("title_multiplicator", 15, "See create_bm25_index.")


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
    for table in tqdm.tqdm(
        tfidf_baseline_utils.iterate_tables(FLAGS.table_file), total=375_000):
      tables[table.table_id] = table

  print("Adding interactions tables ...")
  for interaction_file in interaction_files:
    interactions = prediction_utils.iterate_interactions(interaction_file)
    for interaction in interactions:
      tables[interaction.table.table_id] = interaction.table

  print("Creating index ...")
  index = tfidf_baseline_utils.create_bm25_index(
      tables=tables.values(),
      title_multiplicator=FLAGS.title_multiplicator,
      num_tables=len(tables),
  )

  print("Processing interactions ...")
  for interaction_file in interaction_files:
    interactions = list(prediction_utils.iterate_interactions(interaction_file))

    examples = collections.defaultdict(list)
    for interaction in interactions:
      example_id, _ = preprocess_nq_utils.parse_interaction_id(interaction.id)
      examples[example_id].append(interaction)

    filename = os.path.basename(interaction_file)
    filename = os.path.splitext(filename)[0]
    output = os.path.join(FLAGS.output_dir, filename + "_results.jsonl")
    with tf.io.gfile.GFile(output, "w") as file_writer:
      num_correct = 0
      with tqdm.tqdm(
          examples.items(),
          total=len(examples),
          desc=filename,
      ) as pbar:
        for nr, example in enumerate(pbar):
          example_id, interaction_list = example

          questions = []
          for interaction in interaction_list:
            if len(interaction.questions) != 1:
              raise ValueError(f"Unexpected question in {interaction}")
            questions.append(interaction.questions[0])

          if len(set(q.original_text for q in questions)) != 1:
            raise ValueError(f"Different questions {questions}")
          question_text = questions[0].original_text
          scored_hits = index.retrieve(question_text)
          scored_hits = scored_hits[:FLAGS.max_rank]

          table_scores = []
          for scored_hit in scored_hits:
            table_scores.append({
                "table_id": scored_hit[0],
                "score": -scored_hit[1],
            })

          result = {
              "query_id": example_id + "_0_0",
              "table_scores": table_scores,
          }

          file_writer.write(json.dumps(result))
          file_writer.write("\n")


if __name__ == "__main__":
  app.run(main)
