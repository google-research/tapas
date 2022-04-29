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
"""Tests for tableformer_utils."""

from absl.testing import parameterized
from tapas.utils import tableformer_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class TableAttentionUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_relative_relation_ids_query_table(self):
    # Table Structure
    # Q1 Q2 A0 B0 C0
    #       A1 B1 C1
    #       A2 B2 C2 PAD1 PAD2 PAD3 PAD4
    features = {}
    features['row_ids'] = tf.constant(
        [[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]])
    features['column_ids'] = tf.constant(
        [[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])
    features['segment_ids'] = tf.constant(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    features['input_mask'] = tf.constant(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    expected_ids = tf.constant(
        [[[12, 12, 11, 11, 11, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0],
          [12, 12, 11, 11, 11, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0],
          [9, 9, 8, 7, 7, 6, 0, 0, 6, 0, 0, 0, 0, 0, 0],
          [9, 9, 7, 8, 7, 0, 6, 0, 0, 6, 0, 0, 0, 0, 0],
          [9, 9, 7, 7, 8, 0, 0, 6, 0, 0, 6, 0, 0, 0, 0],
          [5, 5, 4, 0, 0, 3, 1, 1, 2, 0, 0, 0, 0, 0, 0],
          [5, 5, 0, 4, 0, 1, 3, 1, 0, 2, 0, 0, 0, 0, 0],
          [5, 5, 0, 0, 4, 1, 1, 3, 0, 0, 2, 0, 0, 0, 0],
          [5, 5, 4, 0, 0, 2, 0, 0, 3, 1, 1, 0, 0, 0, 0],
          [5, 5, 0, 4, 0, 0, 2, 0, 1, 3, 1, 0, 0, 0, 0],
          [5, 5, 0, 0, 4, 0, 0, 2, 1, 1, 3, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    with self.cached_session() as sess:
      result = sess.run(tableformer_utils.get_relative_relation_ids(features))
      self.assertAllEqual(sess.run(expected_ids), result)

  def test_get_relative_relation_ids_only_table(self):
    # Table Structure
    # A0 B0 C0
    # A1 B1 C1
    # A2 B2 C2 PAD1 PAD2 PAD3 PAD4
    features = {}
    features['row_ids'] = tf.constant([[0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]])
    features['column_ids'] = tf.constant(
        [[1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])
    features['segment_ids'] = tf.constant(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    features['input_mask'] = tf.constant(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    expected_ids = tf.constant([[[8, 7, 7, 6, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                                 [7, 8, 7, 0, 6, 0, 0, 6, 0, 0, 0, 0, 0],
                                 [7, 7, 8, 0, 0, 6, 0, 0, 6, 0, 0, 0, 0],
                                 [4, 0, 0, 3, 1, 1, 2, 0, 0, 0, 0, 0, 0],
                                 [0, 4, 0, 1, 3, 1, 0, 2, 0, 0, 0, 0, 0],
                                 [0, 0, 4, 1, 1, 3, 0, 0, 2, 0, 0, 0, 0],
                                 [4, 0, 0, 2, 0, 0, 3, 1, 1, 0, 0, 0, 0],
                                 [0, 4, 0, 0, 2, 0, 1, 3, 1, 0, 0, 0, 0],
                                 [0, 0, 4, 0, 0, 2, 1, 1, 3, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    with self.cached_session() as sess:
      result = sess.run(tableformer_utils.get_relative_relation_ids(features))
      self.assertAllEqual(sess.run(expected_ids), result)

  def test_get_relative_relation_ids_only_query(self):
    # Table Structure
    # Q1 Q2 PAD1 PAD2 PAD3 PAD4
    features = {}
    features['row_ids'] = tf.constant([[
        0,
        0,
        0,
        0,
        0,
    ]])
    features['column_ids'] = tf.constant([[0, 0, 0, 0, 0]])
    features['segment_ids'] = tf.constant([[0, 0, 0, 0, 0]])
    features['input_mask'] = tf.constant([[1, 1, 0, 0, 0]])
    expected_ids = tf.constant([[[12, 12, 0, 0, 0], [12, 12, 0, 0, 0],
                                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]]])
    with self.cached_session() as sess:
      result = sess.run(tableformer_utils.get_relative_relation_ids(features))
      self.assertAllEqual(sess.run(expected_ids), result)

  @parameterized.named_parameters(
      {
          'testcase_name': '_1',
          'disabled_attention_bias': 1,
      }, {
          'testcase_name': '_2',
          'disabled_attention_bias': 2,
      }, {
          'testcase_name': '_3',
          'disabled_attention_bias': 3,
      }, {
          'testcase_name': '_4',
          'disabled_attention_bias': 4,
      }, {
          'testcase_name': '_5',
          'disabled_attention_bias': 5,
      }, {
          'testcase_name': '_6',
          'disabled_attention_bias': 6,
      }, {
          'testcase_name': '_7',
          'disabled_attention_bias': 7,
      }, {
          'testcase_name': '_8',
          'disabled_attention_bias': 8,
      }, {
          'testcase_name': '_9',
          'disabled_attention_bias': 9,
      }, {
          'testcase_name': '_10',
          'disabled_attention_bias': 10,
      }, {
          'testcase_name': '_11',
          'disabled_attention_bias': 11,
      }, {
          'testcase_name': '_12',
          'disabled_attention_bias': 12,
      }, {
          'testcase_name': '_13',
          'disabled_attention_bias': 13,
      }, {
          'testcase_name': '_1000',
          'disabled_attention_bias': 1000,
      })
  def test_get_relative_relation_ids_disabled_ids(self,
                                                  disabled_attention_bias):
    # Table Structure
    # Q1 Q2 A0 B0 C0
    #       A1 B1 C1
    #       A2 B2 C2 PAD1 PAD2 PAD3 PAD4
    features = {}
    features['row_ids'] = tf.constant(
        [[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]])
    features['column_ids'] = tf.constant(
        [[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])
    features['segment_ids'] = tf.constant(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    features['input_mask'] = tf.constant(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

    expected_ids = [[[12, 12, 11, 11, 11, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0],
                     [12, 12, 11, 11, 11, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0],
                     [9, 9, 8, 7, 7, 6, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                     [9, 9, 7, 8, 7, 0, 6, 0, 0, 6, 0, 0, 0, 0, 0],
                     [9, 9, 7, 7, 8, 0, 0, 6, 0, 0, 6, 0, 0, 0, 0],
                     [5, 5, 4, 0, 0, 3, 1, 1, 2, 0, 0, 0, 0, 0, 0],
                     [5, 5, 0, 4, 0, 1, 3, 1, 0, 2, 0, 0, 0, 0, 0],
                     [5, 5, 0, 0, 4, 1, 1, 3, 0, 0, 2, 0, 0, 0, 0],
                     [5, 5, 4, 0, 0, 2, 0, 0, 3, 1, 1, 0, 0, 0, 0],
                     [5, 5, 0, 4, 0, 0, 2, 0, 1, 3, 1, 0, 0, 0, 0],
                     [5, 5, 0, 0, 4, 0, 0, 2, 1, 1, 3, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

    # Set the `disable_attention_bias` to 0.
    for pos_i in range(len(expected_ids[0])):
      for pos_j in range(len(expected_ids[0][pos_i])):
        if expected_ids[0][pos_i][pos_j] == disabled_attention_bias:
          expected_ids[0][pos_i][pos_j] = 0

    expected_ids = tf.constant(expected_ids)

    with self.cached_session() as sess:
      result = sess.run(
          tableformer_utils.get_relative_relation_ids(features,
                                                      disabled_attention_bias))
      self.assertAllEqual(sess.run(expected_ids), result)


if __name__ == '__main__':
  tf.test.main()
