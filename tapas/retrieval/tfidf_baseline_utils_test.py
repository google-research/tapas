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

from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.retrieval import tfidf_baseline_utils


class TfIdfBaselineUtilsTest(parameterized.TestCase):

  @parameterized.parameters((False, [("A", [("table_0", 1.0)]),
                                     ("B", [("table_1", 1.0)]),
                                     ("A C", [("table_0", 0.75),
                                              ("table_1", 0.25)])]),
                            (True, [("A", [("table_0", 0.5)]),
                                    ("B", [("table_1", 1.0)])]))
  def test_simple(self, drop_term_frequency, expected):
    index = tfidf_baseline_utils.create_inverted_index(
        [
            interaction_pb2.Table(table_id="table_0", document_title="a a c"),
            interaction_pb2.Table(table_id="table_1", document_title="b c")
        ],
        drop_term_frequency=drop_term_frequency)
    for query, results in expected:
      self.assertEqual(index.retrieve(query), results)

  def test_simple_bm25(self):
    expected = [("AA", [("table_0", 1.5475852968796064)]),
                ("BB", [("table_1", 1.2426585328757855)]),
                ("AA CC", [("table_0", 2.0749815245480145),
                           ("table_1", 0.668184203698534)])]
    index = tfidf_baseline_utils.create_bm25_index([
        interaction_pb2.Table(table_id="table_0", document_title="aa aa cc"),
        interaction_pb2.Table(table_id="table_1", document_title="bb cc"),
        interaction_pb2.Table(table_id="table_2", document_title="dd"),
        interaction_pb2.Table(table_id="table_3", document_title="ee"),
        interaction_pb2.Table(table_id="table_4", document_title="ff"),
        interaction_pb2.Table(table_id="table_5", document_title="gg"),
        interaction_pb2.Table(table_id="table_6", document_title="hh"),
    ])
    for query, results in expected:
      self.assertEqual(index.retrieve(query), results)

  def test_min_rank(self):
    index = tfidf_baseline_utils.create_inverted_index([
        interaction_pb2.Table(table_id="table_0", document_title="Table A"),
        interaction_pb2.Table(table_id="table_1", document_title="Table B")
    ],
                                                       min_rank=1)
    self.assertEqual(index.retrieve("A"), [("table_0", 1.0)])
    self.assertEqual(index.retrieve("B"), [("table_1", 1.0)])


if __name__ == "__main__":
  absltest.main()
