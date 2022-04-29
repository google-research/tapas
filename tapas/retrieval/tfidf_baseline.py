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
"""A simple TF-IDF model for table retrieval."""

from typing import Iterable, List, Text

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from tapas.protos import interaction_pb2
from tapas.retrieval import tfidf_baseline_utils
from tapas.scripts import prediction_utils

FLAGS = flags.FLAGS

flags.DEFINE_list("interaction_files", None,
                  "Interaction protos in tfrecord format.")

flags.DEFINE_string("table_file", None, "Table protos in tfrecord format.")

flags.DEFINE_integer("max_table_rank", 50, "Max number of tables to retrieve.")

flags.DEFINE_integer("min_term_rank", 100,
                     "Min term frequency rank to consider.")

flags.DEFINE_boolean("drop_term_frequency", True,
                     "If True, ignore term frequency term.")


def _print(message):
  logging.info(message)
  print(message)


def evaluate(index, max_table_rank,
             thresholds,
             interactions,
             rows):
  """Evaluates index against interactions."""
  ranks = []
  for nr, interaction in enumerate(interactions):
    for question in interaction.questions:
      scored_hits = index.retrieve(question.original_text)
      reference_table_id = interaction.table.table_id
      for rank, (table_id, _) in enumerate(scored_hits[:max_table_rank]):
        if table_id == reference_table_id:
          ranks.append(rank)
          break
    if nr % (len(interactions) // 10) == 0:
      _print(f"Processed {nr:5d} / {len(interactions):5d}.")

  def precision_at_th(threshold):
    return sum(1 for rank in ranks if rank < threshold) / len(interactions)

  values = [f"{precision_at_th(threshold):.4}" for threshold in thresholds]
  rows.append(values)


def create_index(tables,
                 title_multiplicator, use_bm25):
  if use_bm25:
    return tfidf_baseline_utils.create_bm25_index(
        tables,
        title_multiplicator=title_multiplicator,
    )
  return tfidf_baseline_utils.create_inverted_index(
      tables=tables,
      min_rank=FLAGS.min_term_rank,
      drop_term_frequency=FLAGS.drop_term_frequency,
      title_multiplicator=title_multiplicator,
  )


def get_hparams():
  hparams = []
  for multiplier in [1, 2]:
    hparams.append({"multiplier": multiplier, "use_bm25": False})
  for multiplier in [10, 15]:
    hparams.append({"multiplier": multiplier, "use_bm25": True})
  return hparams


def main(_):

  max_table_rank = FLAGS.max_table_rank
  thresholds = [1, 5, 10, 15, max_table_rank]

  for interaction_file in FLAGS.interaction_files:
    _print(f"Test set: {interaction_file}")
    interactions = list(prediction_utils.iterate_interactions(interaction_file))

    for use_local_index in [True, False]:

      rows = []
      row_names = []

      for hparams in get_hparams():

        name = "local" if use_local_index else "global"
        name += "_bm25" if hparams["use_bm25"] else "_tfidf"
        name += f'_tm{hparams["multiplier"]}'

        _print(name)
        if use_local_index:
          index = create_index(
              tables=(i.table for i in interactions),
              title_multiplicator=hparams["multiplier"],
              use_bm25=hparams["use_bm25"],
          )
        else:
          index = create_index(
              tables=tfidf_baseline_utils.iterate_tables(FLAGS.table_file),
              title_multiplicator=hparams["multiplier"],
              use_bm25=hparams["use_bm25"],
          )
        _print("... index created.")
        evaluate(index, max_table_rank, thresholds, interactions, rows)
        row_names.append(name)

        df = pd.DataFrame(rows, columns=thresholds, index=row_names)
        _print(df.to_string())


if __name__ == "__main__":
  flags.mark_flag_as_required("interaction_files")
  flags.mark_flag_as_required("table_file")
  app.run(main)
