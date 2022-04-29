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
"""Evaluates precision@k scores for table retriever predictions."""
from absl import app
from absl import flags

from tapas.scripts import eval_table_retriever_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'prediction_files_local', None,
    'A list of files that contain model predictions as a TSV'
    'file with headers [table_id, query_rep, table_rep].')
flags.DEFINE_string(
    'prediction_files_global', None,
    'A list of files that contain model predictions for all'
    'of that tables in the corpous. Used as the index to'
    'retrieve tables from.')

flags.DEFINE_string(
    'retrieval_results_file_path', None,
    'A path to file where the best tables candidates and their scores, for each'
    'query will be written.')


def main(argv):

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.prediction_files_global:
    eval_table_retriever_utils.eval_precision_at_k(
        FLAGS.prediction_files_local,
        FLAGS.prediction_files_global,
        make_tables_unique=True,
        retrieval_results_file_path=FLAGS.retrieval_results_file_path)
  else:
    eval_table_retriever_utils.eval_precision_at_k(
        FLAGS.prediction_files_local,
        FLAGS.prediction_files_local,
        make_tables_unique=True,
        retrieval_results_file_path=FLAGS.retrieval_results_file_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('prediction_files_local')
  app.run(main)
