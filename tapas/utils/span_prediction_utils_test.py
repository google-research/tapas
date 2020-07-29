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

from absl import logging
from tapas.utils import span_prediction_utils
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class SpanPredictionUtilsTest(tf.test.TestCase):

  def test_get_start_and_end_ids(self):
    label_ids = tf.constant([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1],
    ])
    start_ids, end_ids = span_prediction_utils._get_start_and_end_ids(label_ids)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      (start_ids_val, end_ids_val) = sess.run([start_ids, end_ids])
    self.assertAllClose(start_ids_val, [
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0.],
    ])
    self.assertAllClose(end_ids_val, [
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0., 1.],
    ])

  def test_get_span_logits(self):
    seq_length = 5
    batch_size = 2
    embedding_dim = 3
    embeddings = tf.random.normal(shape=(batch_size, seq_length, embedding_dim))

    start_ids = tf.constant([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    end_ids = tf.constant([
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    column_ids = tf.constant([
        [0, 1, 1, 1, 2],
        [0, 0, 1, 1, 1],
    ])
    row_ids = tf.constant([
        [0, 2, 2, 2, 2],
        [0, 3, 3, 3, 3],
    ])

    spans, span_logits, loss = span_prediction_utils._get_span_logits(
        embeddings,
        start_ids,
        end_ids,
        column_ids,
        row_ids,
        max_span_length=2,
    )

    span_mask = tf.where(
        span_logits > -1000.0,
        tf.ones_like(span_logits),
        tf.zeros_like(span_logits),
    )

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      (
          spans_value,
          span_logits_value,
          loss_value,
          span_mask,
      ) = sess.run([
          spans,
          span_logits,
          loss,
          span_mask,
      ])

    logging.info("span_value: %s", spans_value)
    logging.info("span_logits_value: %s", span_logits_value)
    logging.info("loss_value: %s", loss_value)

    self.assertAllClose(
        spans,
        [
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 1], [1, 2], [2, 3],
             [3, 4]],
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 1], [1, 2], [2, 3],
             [3, 4]],
        ],
    )
    self.assertAllClose(
        span_mask,
        [
            [0., 1., 1., 1., 1., 0., 1., 1., 0.],
            [0., 0., 1., 1., 1., 0., 0., 1., 1.],
        ],
    )
    self.assertGreater(loss_value, 0.0)

  def test_get_boundary_logits(self):
    seq_length = 5
    batch_size = 2
    embedding_dim = 3
    embeddings = tf.random.normal(shape=(batch_size, seq_length, embedding_dim))

    label_ids = tf.constant([
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
    ],
                            shape=(batch_size, seq_length))
    column_ids = tf.constant([
        [0, 1, 1, 1, 2],
        [0, 0, 1, 1, 1],
    ])
    row_ids = tf.constant([
        [0, 2, 2, 2, 2],
        [0, 3, 3, 3, 3],
    ])

    spans, span_logits, loss = span_prediction_utils.get_boundary_logits(
        embeddings, label_ids, column_ids, row_ids, max_span_length=2)

    all_finite = tf.reduce_all(tf.math.is_finite(span_logits))

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      (
          spans_value,
          span_logits_value,
          loss_value,
      ) = sess.run([
          spans,
          span_logits,
          loss,
      ])
      self.assertTrue(sess.run(all_finite))

    logging.info("spans_value: %s", spans_value)
    logging.info("span_logits_value: %s", span_logits_value)
    logging.info("loss_value: %s", loss_value)
    self.assertAllEqual(
        spans_value[0],
        spans_value[1],
    )
    self.assertAllEqual(spans_value[0], [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                                         [0, 1], [1, 2], [2, 3], [3, 4]])
    self.assertGreater(loss_value, 0.0)
    self.assertEqual(spans.shape, (2, 9, 2))
    self.assertEqual(span_logits.shape, (2, 9))
    self.assertEqual(loss.shape, ())


if __name__ == "__main__":
  tf.test.main()
