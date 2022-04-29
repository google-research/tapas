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

from tapas.models.bert import modeling
from tapas.utils import attention_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class AttentionUtilsTest(tf.test.TestCase):

  def test_matches_token_type_id(self):
    with self.cached_session() as sess:
      row_ids = sess.run(
          tf.constant([[1, 2, 2], [5, 5, 6], [1, 2, 3], [4, 5, 6]]))
      result = attention_utils._matches_token_type_id(row_ids)
      expected_result = sess.run(
          tf.constant([
              [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
              [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
              [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          ]))
      self.assertAllEqual(result, expected_result)

  def test_comput_bucket_id(self):
    with self.cached_session() as sess:
      column_ids = tf.constant([[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])
      input_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
      bucket_ids = tf.constant([[0, 0, 0, 1, 2, 1, 2, 3, 1, 2, 3, 3, 4, 4, 4]])

      result = sess.run(
          attention_utils._compute_bucket_id(
              bucket_size=3,
              header_size=3,
              token_type_id=column_ids,
              input_mask=input_mask))
      expected_result = sess.run(bucket_ids)

      self.assertAllEqual(result, expected_result)

  def test_comput_bucket_id_on_distinct_columns(self):
    with self.cached_session() as sess:
      # When bucket size is 1 and columns_ids are a permutation of 0..n-1 then
      # the bucket_ids will match the column_ids
      column_ids = tf.constant([[0, 2, 3, 1, 5, 4]])
      input_mask = tf.constant([[1, 1, 1, 1, 1, 1]])
      bucket_ids = tf.constant([[0, 2, 3, 1, 5, 4]])

      result = sess.run(
          attention_utils._compute_bucket_id(
              bucket_size=1,
              header_size=1,
              token_type_id=column_ids,
              input_mask=input_mask))
      expected_result = sess.run(bucket_ids)

      self.assertAllEqual(result, expected_result)

  def test_comput_bucket_id_with_header(self):
    with self.cached_session() as sess:
      # Similar to the distinct column test, but now we have two header tokens
      column_ids = tf.constant([[0, 2, 3, 1, 5, 4]])
      input_mask = tf.constant([[1, 1, 1, 1, 1, 1]])
      bucket_ids = tf.constant([[0, 1, 2, 0, 4, 3]])

      result = sess.run(
          attention_utils._compute_bucket_id(
              bucket_size=1,
              header_size=2,
              token_type_id=column_ids,
              input_mask=input_mask))
      expected_result = sess.run(bucket_ids)

      self.assertAllEqual(result, expected_result)

  def test_compute_headwise_sparse_attention_mask(self):
    with self.cached_session() as sess:
      # Table Structure
      # Q1 Q2 A0 B0 C0
      #       A1 B1 C1
      #       A2 B2 C2 PAD1 PAD2 PAD3 PAD4

      row_ids = tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]])
      input_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
      segment_ids = tf.constant([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
      column_ids = tf.constant([[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])

      result = sess.run(
          attention_utils.compute_headwise_sparse_attention_mask(
              num_row_heads=2,
              num_column_heads=3,
              bucket_size=0,
              header_size=None,
              input_mask=input_mask,
              segment_ids=segment_ids,
              column_ids=column_ids,
              row_ids=row_ids))[0]

      #    Q1 Q2 A0 A1 A2 B0 B1 B2 C0 C1 C2 PADDING
      expected_row_result = [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Q1
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Q2
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A0
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A1
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A2
          [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # B0
          [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # B1
          [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # B2
          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # C0
          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # C1
          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # C2
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD1
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD2
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD3
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD4
      ]

      #    Q1 Q2 A0 A1 A2 B0 B1 B2 C0 C1 C2 PADDING
      expected_column_result = [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Q1
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Q2
          [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # A0
          [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # A1
          [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # A2
          [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # B0
          [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # B1
          [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # B2
          [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # C0
          [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # C1
          [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C2
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD1
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD2
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD3
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD4
      ]

      expected_result = sess.run(
          tf.constant([expected_row_result] * 2 + [expected_column_result] * 3))

      self.assertAllEqual(result, expected_result)

      # With bucket size 2 no extra attention should be pruned
      result = sess.run(
          attention_utils.compute_headwise_sparse_attention_mask(
              num_row_heads=2,
              num_column_heads=3,
              bucket_size=3,
              header_size=None,
              input_mask=input_mask,
              segment_ids=segment_ids,
              column_ids=column_ids,
              row_ids=row_ids))[0]

      # The attention of the padding tokens changes but it has no impact
      self.assertAllEqual(result[:, :, -4], expected_result[:, :, -4])

  def test_compute_sparse_attention_mask(self):
    with self.cached_session() as sess:
      # Table Structure
      # Q1 Q2 A0 B0 C0
      #       A1 B1 C1
      #       A2 B2 C2 PAD1 PAD2 PAD3 PAD4

      row_ids = tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]])
      input_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
      segment_ids = tf.constant([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
      column_ids = tf.constant([[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])

      result = sess.run(
          attention_utils.compute_sparse_attention_mask(
              input_mask=input_mask,
              segment_ids=segment_ids,
              column_ids=column_ids,
              row_ids=row_ids))[0]

      expected_result = sess.run(
          #    Q1 Q2 A0 A1 A2 B0 B1 B2 C0 C1 C2 PADDING
          tf.constant([
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Q1
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Q2
              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # A0
              [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # A1
              [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # A2
              [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # B0
              [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # B1
              [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],  # B2
              [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # C0
              [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],  # C1
              [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # C2
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD1
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD2
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD3
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # PAD4
          ]))

      self.assertAllEqual(result, expected_result)

  def compare_efficient_and_vanilla_sparse_attention(
      self, sess, row_ids, column_ids,
      embedding_dim, num_row_heads, num_column_heads,
      bucket_size, header_size, seq_length, input_size,
      first_segment_size, rows_sorted):
    tf.random.set_random_seed(42)
    num_attention_heads = num_row_heads + num_column_heads
    pad_size = seq_length - input_size
    second_segment_size = input_size - first_segment_size
    input_mask = tf.constant([[1] * input_size + [0] * pad_size])
    segment_ids = tf.constant([[0] * first_segment_size +
                               [1] * second_segment_size + [0] * pad_size])
    input_tensor = tf.random.normal(shape=[1, seq_length, 128])

    attention_mask = attention_utils.compute_headwise_sparse_attention_mask(
        num_row_heads=num_row_heads,
        num_column_heads=num_column_heads,
        bucket_size=bucket_size,
        header_size=header_size,
        input_mask=input_mask,
        segment_ids=segment_ids,
        column_ids=column_ids,
        row_ids=row_ids)

    expected_result_op = modeling.attention_layer(
        input_tensor,
        input_tensor,
        attention_mask=attention_mask,
        num_attention_heads=num_attention_heads,
        size_per_head=embedding_dim)[0][0, :input_size]

    result_ops = []
    for sort_after_projection in [True, False]:
      attention_layer = attention_utils.create_bucketed_attention_layer(
          input_mask=input_mask,
          input_header=tf.math.equal(segment_ids, 0),
          bucket_size=bucket_size,
          header_size=header_size,
          sort_after_projection=sort_after_projection,
          token_type_ids=[(num_row_heads, rows_sorted, row_ids),
                          (num_column_heads, False, column_ids)])

      result_ops.append(
          attention_layer(
              input_tensor,
              input_tensor,
              num_attention_heads=num_attention_heads,
              size_per_head=embedding_dim)[0][0, :input_size])

    sess.run(tf.global_variables_initializer())
    expected_result, result1, result2 = sess.run([expected_result_op] +
                                                 result_ops)
    self.assertAllClose(result1, expected_result)
    self.assertAllClose(result2, expected_result)

  def test_efficient_sparse_attention_matches_vanilla_version(self):
    # Tests that computing bucketed and non-bucketed attention for random
    # embeddings produces the same result.
    with self.cached_session() as sess:
      # Table Structure
      # Q1 Q2 A0 B0 C0
      #       A1 B1 C1
      #       A2 B2 C2 PAD1 PAD2 PAD3 PAD4

      row_ids = tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]])
      column_ids = tf.constant([[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]])

      self.compare_efficient_and_vanilla_sparse_attention(
          sess,
          row_ids=row_ids,
          column_ids=column_ids,
          embedding_dim=32,
          num_row_heads=2,
          num_column_heads=4,
          bucket_size=3,
          header_size=3,
          seq_length=15,
          input_size=11,
          first_segment_size=2,
          rows_sorted=True)

  def test_efficient_sparse_attention_random_ids_matches_vanilla_version(self):
    # Tests that computing bucketed and non-bucketed attention for random
    # attributes not mapping to real columns yield the same results.
    with self.cached_session() as sess:
      seq_length = 14
      row_ids = tf.random.uniform(
          shape=[1, seq_length], maxval=20, dtype=tf.int32)
      column_ids = tf.random.uniform(
          shape=[1, seq_length], maxval=20, dtype=tf.int32)
      self.compare_efficient_and_vanilla_sparse_attention(
          sess,
          row_ids=row_ids,
          column_ids=column_ids,
          embedding_dim=16,
          num_row_heads=5,
          num_column_heads=1,
          bucket_size=2,
          header_size=4,
          seq_length=seq_length,
          input_size=11,
          first_segment_size=2,
          rows_sorted=False)


if __name__ == "__main__":
  tf.test.main()
