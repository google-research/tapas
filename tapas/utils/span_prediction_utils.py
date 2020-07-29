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
"""Utilities for predicting answer spans."""

import enum
from typing import Text, Tuple

from absl import logging

from tapas.models import tapas_classifier_model_utils as utils
from tapas.models.bert import modeling

import tensorflow.compat.v1 as tf

_classification_initializer = utils.classification_initializer

_NEG_INFINITY = -10000.0


class SpanPredictionMode(str, enum.Enum):
  NONE = "none"
  BOUNDARY = "boundary"
  SPAN = "span"


def _get_indexes(seq_length, batch_size):
  return tf.repeat(
      tf.expand_dims(tf.range(seq_length), axis=0), repeats=batch_size, axis=0)


def _get_valid_indexes(
    indexes,
    offset,
):
  """Restrict to indexes so that i+offset still is a valid index."""
  if offset == 0:
    return indexes
  return indexes[:, :-offset]


def _gather_indexes(
    indexes,
    flat_spans_2d,
    start_or_end,
):
  """Gathers indexes for start or end index.

  Where flat_spans_2d is a data-structure built to work with tf.gather_nd.
  It pairs a batch index with a start or end index.

  Args:
    indexes: <int32>[batch_size, seq_length].
    flat_spans_2d: <int32>[batch_size, num_spans * 2, 2].
    start_or_end: 0 for start index, 1 for end index.

  Returns:
    indexes: <int32>[batch_size, num_spans].
  """
  shape = modeling.get_shape_list(flat_spans_2d, expected_rank=3)
  batch_size = shape[0]
  num_spans = shape[1] // 2
  span_index = tf.gather_nd(params=indexes, indices=flat_spans_2d)
  span_index = tf.reshape(span_index, shape=(batch_size, num_spans, 2))
  span_index = tf.slice(
      span_index, begin=[0, 0, start_or_end], size=[batch_size, num_spans, 1])
  span_index = tf.squeeze(span_index, axis=2)
  return span_index


def _get_mask(condition, dtype=tf.float32):
  return tf.cast(
      tf.where(condition, tf.ones_like(condition), tf.zeros_like(condition)),
      dtype=dtype,
  )


def _get_id_mask(
    ids,
    flat_spans_2d,
    dtype=tf.float32,
):
  """Checks that all tokens in span have same id and that id > 0."""
  start_ids = _gather_indexes(ids, flat_spans_2d, 0)
  end_ids = _gather_indexes(ids, flat_spans_2d, 1)
  return _get_mask(
      tf.math.logical_and(tf.equal(start_ids, end_ids), start_ids > 0),
      dtype=dtype,
  )


def _get_start_and_end_ids(label_ids):
  """Compute span begin and end from label_ids.

  This will output multiple start / end indexes if there are multiple
  consecutive spans.

  Args:
    label_ids: <int32>[batch_size, seq_length]

  Returns:
    start_ids: <<float32>[batch_size, seq_length]
    end_ids: <float32>[batch_size, seq_length]
  """
  shape = modeling.get_shape_list(label_ids, expected_rank=2)
  batch_size = shape[0]
  # <int32>[batch_size, 1]
  padding = tf.zeros(shape=(batch_size, 1), dtype=tf.int32)
  # <int32>[batch_size, seq_length + 1]
  left_label_ids = tf.concat([label_ids, padding], axis=1)
  # <int32>[batch_size, seq_length + 1]
  right_label_ids = tf.concat([padding, label_ids], axis=1)

  start_index = tf.cast(left_label_ids - right_label_ids > 0, tf.float32)
  # <int32>[batch_size, seq_length]
  start_index = start_index[:, :-1]
  start_index = tf.cast(start_index, tf.float32)

  end_index = tf.cast(right_label_ids - left_label_ids > 0, tf.float32)
  # <int32>[batch_size, seq_length]
  end_index = end_index[:, 1:]
  end_index = tf.cast(end_index, tf.float32)
  return start_index, end_index


def compute_token_span_logits(
    embeddings,
    name,
):
  """Computes logits per token.

  Args:
    embeddings: <float>[batch_size, seq_length, hidden_dim] Output of the
      encoder layer.
    name: Name for variable.

  Returns:
    <float>[batch_size, seq_length] Logits per token.
  """
  hidden_size = embeddings.shape.as_list()[-1]
  output_weights = tf.get_variable(
      f"{name}_output_weights", [hidden_size],
      initializer=_classification_initializer())
  output_bias = tf.get_variable(
      f"{name}_output_bias", shape=(), initializer=tf.zeros_initializer())
  logits = (tf.einsum("bsj,j->bs", embeddings, output_weights) + output_bias)
  return logits


def _get_spans(
    seq_length,
    batch_size,
    max_span_length,
):
  """Computes all possible spans of a certain size."""
  indexes = _get_indexes(seq_length, batch_size)

  all_spans = []
  for offset in range(max_span_length):
    end_indexes = indexes + offset
    start_indexes = _get_valid_indexes(indexes, offset)
    end_indexes = _get_valid_indexes(end_indexes, offset)
    spans = tf.stack([start_indexes, end_indexes], axis=-1)
    all_spans.append(spans)

  # <int32>[batch_size, num_spans, 2]
  spans = tf.concat(all_spans, axis=1)
  num_spans = modeling.get_shape_list(spans, expected_rank=3)[1]

  batch_dims = tf.expand_dims(tf.range(0, batch_size), axis=1)
  # <int32>[batch_size, num_spans*2]
  batch_dims = tf.repeat(batch_dims, repeats=num_spans * 2, axis=1)
  flat_spans = tf.reshape(spans, shape=(batch_size, num_spans * 2))
  # <int32>[batch_size, num_spans*2, 2]
  flat_spans_2d = tf.stack([batch_dims, flat_spans], axis=2)
  return spans, num_spans, flat_spans_2d


def _get_span_mask(column_ids, row_ids, flat_spans_2d):
  """Computes a mask that is true for all spans that are within a table cell."""
  # <bool>[batch_size, num_spans]
  same_column = _get_id_mask(column_ids, flat_spans_2d, tf.bool)
  # <bool>[batch_size, num_spans]
  same_row = _get_id_mask(row_ids, flat_spans_2d, tf.bool)
  # <bool>[batch_size, num_spans]
  span_mask = tf.cast(same_column & same_row, tf.bool)

  logging.info("same_column: %s", same_column)
  logging.info("same_row: %s", same_row)
  logging.info("span_mask: %s", span_mask)
  return span_mask


def _mask_spans(span_mask, span_logits):
  masked_span_logits = tf.where(
      span_mask,
      span_logits,
      tf.ones_like(span_logits) * _NEG_INFINITY,
  )
  logging.info("mask_span_logits: %s", masked_span_logits)
  return masked_span_logits


def _get_span_logits(
    embeddings,
    start_ids,
    end_ids,
    column_ids,
    row_ids,
    max_span_length,
):
  """Computes span logits.

  Args:
    embeddings: <float32>[batch_size, seq_length, embedding_dim]
    start_ids: <int32>[batch_size, seq_length], 1 if span start there
    end_ids: <int32>[batch_size, seq_length], 1 if span ends there
    column_ids: <int32>[batch_size, seq_length]
    row_ids: <int32>[batch_size, seq_length]
    max_span_length: Max span to create.

  Returns:
    spans: <int32>[batch_size, num_spans, 2]
    span_logits: <float32>[batch_size, num_spans]
    loss: <float32>
  """

  shape = modeling.get_shape_list(embeddings, expected_rank=3)
  batch_size = shape[0]
  seq_length = shape[1]
  embedding_dim = shape[2]

  spans, num_spans, flat_spans_2d = _get_spans(seq_length, batch_size,
                                               max_span_length)

  # <float32>[batch_size, num_spans, embedding_dim, 2]
  embedding_layer = tf.gather_nd(params=embeddings, indices=flat_spans_2d)
  embedding_layer = tf.reshape(
      embedding_layer, shape=(batch_size, num_spans, embedding_dim * 2))

  # <float32>[batch_size, num_spans]
  span_logits = compute_token_span_logits(embedding_layer, "span")
  logging.info("span_logits: %s", span_logits)

  def _get_loss(masked_span_logits):
    # <bool>[batch_size, num_spans]
    is_span_start_correct = _gather_indexes(
        tf.cast(start_ids, tf.bool), flat_spans_2d, 0)
    # <bool>[batch_size, num_spans]
    is_span_end_correct = _gather_indexes(
        tf.cast(end_ids, tf.bool), flat_spans_2d, 1)
    # <bool>[batch_size, num_spans]
    # If the sum is 2.0 the span has a valid start and end index.
    span_correct = _get_mask(is_span_start_correct & is_span_end_correct)
    logging.info("span_correct: %s", span_correct)
    # <float32>[batch_size]
    has_correct_span = _get_mask(tf.reduce_sum(span_correct, axis=1) > 0)
    logging.info("has_correct_span: %s", has_correct_span)
    # <float32>[batch_size, num_spans]
    span_probs = tf.nn.log_softmax(masked_span_logits, axis=-1)
    logging.info("span_probs: %s", span_probs)
    # <float32>[batch_size]
    per_example_loss = -tf.reduce_sum(
        span_correct * span_probs, axis=-1) * has_correct_span
    # <float32>
    loss = tf.reduce_mean(per_example_loss)
    return loss

  span_mask = _get_span_mask(column_ids, row_ids, flat_spans_2d)
  masked_span_logits = _mask_spans(span_mask, span_logits)
  # <float32>[batch_size, num_spans]
  return spans, masked_span_logits, _get_loss(masked_span_logits)


def get_span_logits(
    embeddings,
    label_ids,
    column_ids,
    row_ids,
    max_span_length,
):
  start_ids, end_ids = _get_start_and_end_ids(label_ids)
  return _get_span_logits(
      embeddings,
      start_ids,
      end_ids,
      column_ids,
      row_ids,
      max_span_length,
  )


def _get_span_loss(logits, label_ids):
  """Compute simple soft max loss.

  This will work occur but not perfectly if presented with multiple spans.
  In particular, all possible combinations of correct start and end spans
  will be treated as correct. This should not over-genereate a lot since we do
  make sure that the spans don't cross cell boundaries.

  Args:
    logits: <float32[batch_size, seq_length]
    label_ids: <float32>[batch_size, seq_length]

  Returns:
    loss: <float32>
  """
  log_probs = tf.nn.log_softmax(logits, axis=-1)
  # <float32>[batch_size]
  per_example_loss = -tf.reduce_sum(label_ids * log_probs, axis=-1)
  loss = tf.reduce_mean(
      compute_masked_example_loss(
          label_ids,
          per_example_loss,
      ))
  return loss


def get_boundary_logits(
    embeddings,
    label_ids,
    column_ids,
    row_ids,
    max_span_length,
):
  """Computes span logits as sum of start and end index logits.

  Args:
    embeddings: <float32>[batch_size, seq_length, embedding_dim]
    label_ids: <int32>[batch_size, seq_length]
    column_ids: <int32>[batch_size, seq_length]
    row_ids: <int32>[batch_size, seq_length]
    max_span_length: Max span to create.

  Returns:
    spans: <int32>[batch_size, num_spans, 2]
    span_logits: <float32>[batch_size, num_spans]
    loss: <float32>
  """
  start_ids, end_ids = _get_start_and_end_ids(label_ids)
  start_logits = compute_token_span_logits(embeddings, name="start")
  end_logits = compute_token_span_logits(embeddings, name="end")
  loss = _get_span_loss(start_logits, start_ids)
  loss += _get_span_loss(end_logits, end_ids)

  shape = modeling.get_shape_list(label_ids, expected_rank=2)
  batch_size = shape[0]
  seq_length = shape[1]
  spans, _, flat_spans_2d = _get_spans(
      seq_length,
      batch_size,
      max_span_length,
  )

  span_logits = _gather_indexes(start_logits, flat_spans_2d, start_or_end=0)
  span_logits = _gather_indexes(end_logits, flat_spans_2d, start_or_end=1)
  span_mask = _get_span_mask(column_ids, row_ids, flat_spans_2d)
  span_logits = _mask_spans(span_mask, span_logits)
  return spans, span_logits, loss


def get_span_logits_by_mode(
    mode, embeddings, label_ids, column_ids, row_ids,
    max_span_length):
  """Delegates to the respective implementation of the mode."""
  if mode == SpanPredictionMode.BOUNDARY:
    return get_boundary_logits(
        embeddings,
        label_ids,
        column_ids,
        row_ids,
        max_span_length=max_span_length,
    )
  elif mode == SpanPredictionMode.SPAN:
    return get_span_logits(
        embeddings,
        label_ids,
        column_ids,
        row_ids,
        max_span_length=max_span_length,
    )
  else:
    raise ValueError("Unsupported mode: {mode}")


def compute_masked_example_loss(
    label_ids,
    per_example_loss,
):
  """Computes a mask that denotes whether the ith example has an answer.

  Args:
    label_ids: <int32>[batch_size, seq_length]
    per_example_loss: <float32>[batch_size]

  Returns:
    label_mask: <float32>[batch_size].
  """
  is_cell_supervision_available = tf.reduce_sum(label_ids, axis=1) > 0
  mask = tf.where(
      is_cell_supervision_available,
      tf.ones_like(per_example_loss),
      tf.zeros_like(per_example_loss),
  )
  mask = tf.stop_gradient(mask)
  return per_example_loss * mask
