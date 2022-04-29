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
import json
import tempfile

from absl.testing import absltest
import numpy as np

from tapas.scripts import eval_table_retriever_utils


class EvalTableRetrieverTest(absltest.TestCase):

  def _generate_sythetic_data(self):
    queries = [
        eval_table_retriever_utils.QueryExample(
            query_id='1', table_ids=['a'], query=np.array([1, 0, 0, 0])),
        eval_table_retriever_utils.QueryExample(
            query_id='2', table_ids=['b'], query=np.array([0, 1, 0, 0])),
        eval_table_retriever_utils.QueryExample(
            query_id='3', table_ids=['c'], query=np.array([0, 0, 1, 0])),
        eval_table_retriever_utils.QueryExample(
            query_id='4', table_ids=['d'], query=np.array([0, 0, 0, 1])),
    ]
    tables = [
        eval_table_retriever_utils.TableExample(
            table_id='a', table=np.array([1, 0, 0, 0])),
        eval_table_retriever_utils.TableExample(
            table_id='b', table=np.array([0.5, 1, 0, 0])),
        eval_table_retriever_utils.TableExample(
            table_id='c', table=np.array([0, 0.5, 0.5, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='d', table=np.array([0, 0, 0, 0])),
        eval_table_retriever_utils.TableExample(
            table_id='e', table=np.array([0, 0, 1, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='f', table=np.array([0, 0, 0, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='g', table=np.array([0, 0, 0, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='h', table=np.array([0, 0, 0, 1])),
    ]
    return queries, tables

  def _load_results_from_file(self, path):
    results = []
    with open(path, 'r') as f:
      for row in f:
        results.append(json.loads(row))
    return results

  def test_retrieve(self):
    queries, tables = self._generate_sythetic_data()
    eval_table_retriever_utils._NUM_NEIGHBORS = 2
    index = eval_table_retriever_utils.build_table_index(tables)
    similarities, neighbors = eval_table_retriever_utils._retrieve(
        queries, index)
    expected_similarities = [1, 0.5, 1, 0.5, 1, 0.5, 1, 1]
    expected_neighbors = [0, 1, 1, 2, 4, 2, 4, 7]
    self.assertSequenceEqual(similarities.flatten().tolist(),
                             expected_similarities)
    self.assertSequenceEqual(neighbors.flatten().tolist(), expected_neighbors)

  def test_eval_process_predictions(self):
    """Tests whether precision@k calculations are done properly."""
    queries, tables = self._generate_sythetic_data()
    eval_table_retriever_utils._NUM_NEIGHBORS = 6
    index = eval_table_retriever_utils.build_table_index(tables)
    results_file_path = tempfile.mktemp()
    precision_at_k = eval_table_retriever_utils.process_predictions(
        queries, tables, index, retrieval_results_file_path=results_file_path)
    self.assertEqual(precision_at_k, {
        'precision_at_1': 0.5,
        'precision_at_5': 0.75
    })

    results = self._load_results_from_file(results_file_path)
    for result in results:
      if result['query_id'] == '1':
        for table in result['table_scores']:
          if table['table_id'] == 'a':
            self.assertEqual(table['score'], -1.0)

  def test_eval_duplicate_query_ids(self):
    """Tests whether precision@k calculations are done properly."""
    eval_table_retriever_utils._NUM_NEIGHBORS = 6

    queries = [
        eval_table_retriever_utils.QueryExample(
            query_id='1',
            table_ids=['b', 'c'],
            query=np.array([1, 0, 0, 0]),
        ),
        eval_table_retriever_utils.QueryExample(
            query_id='2',
            table_ids=['b'],
            query=np.array([0, 1, 0, 0]),
        ),
    ]
    tables = [
        eval_table_retriever_utils.TableExample(
            table_id='a', table=np.array([1, 0, 0, 0])),
        eval_table_retriever_utils.TableExample(
            table_id='b', table=np.array([0.5, 1, 0, 0])),
        eval_table_retriever_utils.TableExample(
            table_id='c', table=np.array([0, 0.5, 0.5, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='d', table=np.array([0, 0, 0, 0])),
        eval_table_retriever_utils.TableExample(
            table_id='e', table=np.array([0, 0, 1, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='f', table=np.array([0, 0, 0, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='g', table=np.array([0, 0, 0, 1])),
        eval_table_retriever_utils.TableExample(
            table_id='h', table=np.array([0, 0, 0, 1])),
    ]

    index = eval_table_retriever_utils.build_table_index(tables)
    results_file_path = tempfile.mktemp()
    precision_at_k = eval_table_retriever_utils.process_predictions(
        queries, tables, index, retrieval_results_file_path=results_file_path)
    self.assertEqual(precision_at_k, {
        'precision_at_1': 0.5,
        'precision_at_5': 1.0
    })


if __name__ == '__main__':
  absltest.main()
