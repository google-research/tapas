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
"""Helper methods to restrict attention in Transformer models."""

import dataclasses
import enum
import functools
import math
from typing import Any, Callable, List, Mapping, Optional, Text, Tuple

import tensorflow.compat.v1 as tf


_INFINITY = 10000  # Same value used in BERT


class RestrictAttentionMode(str, enum.Enum):
  FULL = 'full'
  SAME_COLUMN_OR_ROW = 'same_colum_or_row'
  HEADWISE_SAME_COLUMN_OR_ROW = 'headwise_same_colum_or_row'
  HEADWISE_EFFICIENT = 'headwise_efficient'
  TABLE_ATTENTION = 'table_attention'


def _matches_token_type_id(tensor):
  """Returns a 3D tensor indicating whether feature at position i and j match.

  Args:
    tensor: <int32>[batch_size, seq_length]

  Returns:
    tensor: <bool>[batch_size, seq_length, seq_length].
  """
  return tf.math.equal(
      tf.expand_dims(tensor, axis=1), tf.expand_dims(tensor, axis=2))


def compute_sparse_attention_mask(segment_ids, column_ids, row_ids, input_mask,
                                  **_):
  """Computes 3D attention matrix.

  Position i will attent to position j if:
  j has a 1 in the input mask and at least one of the following is true:
   * i and j are in the same column
   * i and j are in the same row
   * j has segment id 0 (e.g. is part of the question)
   * i has segment id 0 (e.g. is part of the question)

  Args:
    segment_ids: <int32>[batch_size, seq_length]
    column_ids: <int32>[batch_size, seq_length]
    row_ids: <int32>[batch_size, seq_length]
    input_mask: <int32>[batch_size, seq_length]

  Returns:
    attention_mask: <float32>[batch_size, seq_length, seq_length]
  """
  segment_zero = tf.math.equal(segment_ids, 0)
  mask_one = tf.math.equal(input_mask, 1)
  return tf.cast(
      tf.expand_dims(mask_one, axis=1) &
      (_matches_token_type_id(column_ids) | _matches_token_type_id(row_ids)
       | tf.expand_dims(segment_zero, axis=2)
       | tf.expand_dims(segment_zero, axis=1)), tf.float32)


def _compute_bucket_id(bucket_size, header_size, token_type_id,
                       input_mask):
  """Reorder inputs based on token_type index and compute the bucket index.

  Args:
    bucket_size: <int32> Only attend to position that fall in consecutive
      equally sized buckets, or to/from the first bucket.
    header_size: <int32> The size of the first bucket.
    token_type_id: <int32>[batch_size, seq_length]
    input_mask: <int32>[batch_size, seq_length]

  Returns:
    bucket_id: <int32>[batch_size, seq_lengh]
  """
  # Ensure that the padding tokens get sorted last
  masked_token_type_id = token_type_id + (-input_mask + 1) * 10000
  # Argsorting twice gives you the rank. Sorting once and using scatter could
  # be slightly faster, but not relevant for simulation code
  token_type_ranks = tf.argsort(
      tf.argsort(masked_token_type_id, stable=True), stable=True)
  # Elements from 0 to `header_size` - 1 are mapped to 0, the rest are dividied
  # evenly into groups of size `bucket_size` starting from 1.
  return tf.math.maximum((token_type_ranks - header_size) // bucket_size + 1, 0)


def _compute_cross_bucket_attention_mask(bucket_size, header_size,
                                         token_type_id, input_mask):
  """Create attention matrix simulating a reshaped input split in equal buckets.

  Args:
    bucket_size: <int32> Only attend to position that fall in consecutive
      equally sized buckets, or to/from the first bucket.
    header_size: <int32> The size of the first bucket.
    token_type_id: <int32>[batch_size, seq_length]
    input_mask: <int32>[batch_size, seq_length]

  Returns:
    cross_bucket_attention_mask: <int32>[batch_size, seq_lengh, seq_lengh]
  """
  if bucket_size < 1:
    return tf.expand_dims(tf.ones_like(token_type_id, dtype=tf.bool), axis=1)
  seq_length = token_type_id.shape[1]
  bucket_id = _compute_bucket_id(bucket_size, header_size, token_type_id,
                                 input_mask)
  bucket_zero = tf.math.equal(bucket_id, 0)
  bucket_diff = tf.abs(
      tf.expand_dims(bucket_id, axis=1) - tf.expand_dims(bucket_id, axis=2))
  # Allow second and last position to attend to each other since this makes the
  # efficient implementation easier.
  return (tf.math.less_equal(bucket_diff, 1)
          | tf.math.equal(bucket_diff, (seq_length // bucket_size) - 2)
          | tf.expand_dims(bucket_zero, axis=1)
          | tf.expand_dims(bucket_zero, axis=2))


def compute_headwise_sparse_attention_mask(num_row_heads,
                                           num_column_heads,
                                           bucket_size, header_size,
                                           segment_ids, column_ids, row_ids,
                                           input_mask, **_):
  """Computes 4D attention matrix, that varies by head.

  Position i will attent to position j on head h:
  j has a 1 in the input mask and at least one of the following is true:
   * i and j are in the same row and h < num_row_heads
   * i and j are in the same columns and h >= num_row_heads
   * j has segment id 0 (e.g. is part of the question)
   * i has segment id 0 (e.g. is part of the question)
  If using bucket_size we additionally impose the restriction that when sorting
  by the corresponding attribute (columns or rows) and splitting input in
  equal size buckets, only tokens on consecutive buckets or the first bucket
  attend to/from each other.

  Args:
    num_row_heads: <int32> Number of heads that attend within a row
    num_column_heads: <int32> Number of heads that attend within a column
    bucket_size: <int32> Only attend to position that fall in consecutive
      equally sized buckets, or to/from the first bucket.
    header_size: Optional<int32> The size of the first bucket. Will use
      `bucket_size` if None is passed.
    segment_ids: <int32>[batch_size, seq_length]
    column_ids: <int32>[batch_size, seq_length]
    row_ids: <int32>[batch_size, seq_length]
    input_mask: <int32>[batch_size, seq_length]

  Returns:
    attention_mask: <float32>[batch_size, num_heads, seq_length, seq_length]
  """
  # <bool>[batch_size, seq_length]
  segment_zero = tf.math.equal(segment_ids, 0)
  # <bool>[batch_size, seq_length, seq_length]
  to_or_from_segument_zero = (
      tf.expand_dims(segment_zero, axis=2)
      | tf.expand_dims(segment_zero, axis=1))
  # <bool>[batch_size, 1, seq_length]
  mask_one = tf.expand_dims(tf.math.equal(input_mask, 1), axis=1)

  if header_size is None:
    header_size = bucket_size

  row_bucket_mask = _compute_cross_bucket_attention_mask(
      bucket_size, header_size, row_ids, input_mask)

  column_bucket_mask = _compute_cross_bucket_attention_mask(
      bucket_size, header_size, column_ids, input_mask)

  # <bool>[batch_size, seq_length, seq_length]
  row_wise = (
      mask_one & row_bucket_mask &
      (_matches_token_type_id(row_ids) | to_or_from_segument_zero))

  # <bool>[batch_size, seq_length, seq_length]
  column_wise = (
      mask_one & column_bucket_mask &
      (_matches_token_type_id(column_ids) | to_or_from_segument_zero))

  # <bool>[batch_size, 2, seq_length, seq_length]
  rows_and_columns = tf.stack([row_wise, column_wise], axis=1)

  # Repeat the row-wise and column-wise attention the correct number of times
  # <bool>[batch_size, num_row_heads + num_column_heads, seq_length, seq_length]
  result = tf.repeat(
      rows_and_columns, repeats=[num_row_heads, num_column_heads], axis=1)

  return tf.cast(result, tf.float32)


@dataclasses.dataclass
class _BucketedTensor():
  """Results of sorting and reshaping a tensor into equally sized buckets.

  Different views of the tensor will be created to facilitate attention across
  tokens from different buckets. This can be constructed from a rank 3 or rank
  4 tensor, using the method _create_bucketed_tensor. Rank 3 integer tensors
  correspond to token attributes, like segment or input mask. Rank 4 float
  tensors correspond to token embeddings, be it keys, queries or values. First
  two dimensions are always batch_size and num_heads:

  Attributes:
    full: <float32>[..., seq_length, embedding_size] or <int32>[..., 1,
      seq_length]. Original tensor without any bucketing.
    head: <float32>[..., bucket_size, embedding_size] or <int32>[..., 1,
      bucket_size]. Values in first bucket.
    tail: <float32>[..., num_buckets - 1, bucket_size, embedding_size] or
      <int32>[..., num_buckets - 1, bucket_size]. Buckets after the first.
    window: <float32>[..., num_buckets - 1, 3 * bucket_size, embedding_size] or
      <int32>[..., num_buckets - 1, 3 * bucket_size]. This tensor contains same
      as rest but also a rotation to the left and right, in order to achieve
      attention to the previous and next bucket.
  """
  full: tf.Tensor
  head: tf.Tensor
  tail: tf.Tensor
  window: tf.Tensor

  def expand_from(self, view):
    return tf.expand_dims(getattr(self, view), axis=-1)

  def expand_to(self, view):
    return tf.expand_dims(getattr(self, view), axis=-2)


def _create_bucketed_tensor(tensor,
                            name,
                            order_indices,
                            bucket_size,
                            header_size,
                            num_heads,
                            sort_after_projection,
                            embedding_size = None):
  """Builds a _BucketedTensor view based on order indexes and a bucket size.

  Args:
    tensor: Input tensor of rank 3 or 4. First free dimensions are batch_size,
      num_heads and seq_length. Optional last one is an embedding size.
    name: Variable scope to use for all the ops in this function
    order_indices: <int32>[batch_size, num_heads, seq_length] Indices to reorder
      the tensor for each head before bucketing.
    bucket_size: Size of buckets that will be created after re-ordering.
    header_size: Size of the first bucket that will attend to/from everything.
    num_heads: Number of heads. Must match the second dimension.
    sort_after_projection: Whether to sort input tensor.
    embedding_size: Optional dimension of embeddings for rank 4 tensors. Only
      needed for rank 4 `tensor`.

  Returns:
    _BucketedTensor
  """
  rank = tensor.shape.rank
  if rank not in [3, 4]:
    raise ValueError('Input tensor nees to have rank 3 or 4')

  with tf.variable_scope(name):
    sorted_tensor = (
        tf.gather(tensor, order_indices, batch_dims=2)
        if sort_after_projection else tensor)
    sorted_tensor_tail = sorted_tensor[:, :, header_size:, Ellipsis]
    sorted_tensor_head = sorted_tensor[:, :, :header_size, Ellipsis]
    seq_length = tensor.shape[2] - header_size
    new_shape = [-1, num_heads, seq_length // bucket_size, bucket_size]

    if rank == 3:
      # For token_ids keeping the dummy dimension makes the matching easier.
      full = tf.expand_dims(tensor, axis=2)
      head = tf.expand_dims(sorted_tensor_head, axis=2)
    else:
      new_shape.append(embedding_size)
      full = tensor
      head = sorted_tensor_head

    tail = tf.reshape(sorted_tensor_tail, shape=new_shape)

    def _rotate(t, offset):
      return tf.concat([t[:, :, offset:, Ellipsis], t[:, :, :offset, Ellipsis]],
                       axis=2,
                       name=f'rotate_{offset}')

    # Creates a helper tensor to attend to past and future buckets.
    # Because of the rotation the second and last buckets will attend to each
    # other. While this could be explicitly masked, it doesn't really hurt.
    # <>[batch_size, num_heads, num_buckets - 1, 3 * bucket_size, ...]
    window = tf.concat([tail, _rotate(tail, 1), _rotate(tail, -1)], axis=3)
  return _BucketedTensor(full, head, tail, window)


def _additive_mask(mask):
  """Converts a [0, 1] mask to [-infinity, 0] mask to be added to logits."""
  return (tf.cast(mask, tf.int32) - 1) * _INFINITY


def _dense_layer_3d(input_tensor, activation,
                    name, num_heads, size_per_head,
                    initializer_range, sort_after_projection):
  """Linear layer that computes a 2d tensor for every position."""
  last_dim = input_tensor.shape[-1]
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    # To preserve compatibility with BERT checkpoints we keep the same shapes.
    w = tf.get_variable(
        name='kernel',
        shape=[last_dim, num_heads * size_per_head],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    w = tf.reshape(w, [last_dim, num_heads, size_per_head])
    b = tf.get_variable(
        name='bias',
        shape=[num_heads * size_per_head],
        initializer=tf.zeros_initializer)
    b = tf.reshape(b, [1, num_heads, 1, size_per_head])

    # When input is already sorted, it contains the Heads axis
    input_shape = 'BSI' if sort_after_projection else 'BHSI'
    ret = tf.einsum(f'{input_shape},IHE->BHSE', input_tensor, w) + b
    return ret if activation is None else activation(ret)


def _dropout(tensor, dropout_prob):
  """Dropout layer."""
  if not dropout_prob:
    return tensor

  return tf.nn.dropout(tensor, rate=dropout_prob)


def _compute_bucketed_attention_mask(from_view, to_view,
                                     ids,
                                     attend_always):
  """Compute mask to add to attention scores between BucketedTensors."""
  # Allow attention when the ids match, or one of the tokens is in the header.
  mask = (ids.expand_to(to_view) < _INFINITY) & (
      tf.math.equal(ids.expand_from(from_view), ids.expand_to(to_view))
      | attend_always.expand_from(from_view) | attend_always.expand_to(to_view))

  if not ({'tail', 'window'} & {from_view, to_view}):
    # When there are no buckets in from or to tensors, the axis is removed.
    # This happens when first attends to full.
    mask = tf.squeeze(mask, axis=2)

  return tf.cast(_additive_mask(mask), tf.float32)


def _attention_scores(from_view, to_view, additive_mask,
                      queries, keys):
  """Computes masked attention scores between two views of bucketed tensors."""
  from_buckets = 'N' if from_view in ('tail', 'window') else ''
  to_buckets = 'N' if to_view in ('tail', 'window') else ''
  result_buckets = from_buckets or to_buckets
  # Computes unmasked attention scores. If either from or to views have a
  # num_bucket dimension, we keep it in the output.
  scores = tf.einsum(
      f'BH{from_buckets}FE,BH{to_buckets}TE->BH{result_buckets}FT',
      getattr(queries, from_view),
      getattr(keys, to_view),
      name=f'query_key_{from_view}_{to_view}')

  return scores + additive_mask


def create_bucketed_attention_layer(
    input_mask, input_header, bucket_size, header_size,
    token_type_ids,
    sort_after_projection):
  """Returns a drop-in replacement for attention_layer using sparsse attention.

  Args:
    input_mask: int32<batch_size, seq_length> The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    input_header: bool<batch_size, seq_length> The values should be 1 or 0.
      Attention will not be restricted to or from tokens where header is 1.
    bucket_size: int. Size of sections where self attention happens.
    header_size: Size of the first bucket that will attend to/from everything.
      If None is passed will use the same as `bucket_size`.
    token_type_ids: List<(int, bool, <int32>[batch_size, seq_length])> contains
      the number of heads for each token type, whether they are sorted, and the
      ids of each position. Attention is restricted between tokens with the same
      type id and this field is used to sort/bucket. ids must be non-negative.
    sort_after_projection: Sorting can happen on the layer input or after
      applying the projection to keys, queries and values. Depending on the
      accelerator, one option could be more convenient.

  Returns:
    Function with same signature as `attention_layer`.
    See `_bucketed_attention_layer`.
  """
  type_counts = [cnt for cnt, _, _ in token_type_ids]
  num_heads = sum(type_counts)

  # Ensure that the padding tokens get sorted last.
  additive_mask = _additive_mask(input_mask)

  type_sorted = [is_sorted for _, is_sorted, _ in token_type_ids]
  type_ids_masked = [ids - additive_mask for _, _, ids in token_type_ids]
  type_ids_argsorted = [
      None if is_sorted else tf.argsort(ids, stable=True)
      for is_sorted, ids in zip(type_sorted, type_ids_masked)
  ]

  def _gather_and_repeat(tensors, name):
    """Reorder each tensor using its type_id and copies it once per head."""
    if len(tensors) == 1:
      tensors = tensors * len(type_ids_argsorted)
    with tf.variable_scope(name):
      gathered_tensors = []
      for tensor, indices in zip(tensors, type_ids_argsorted):
        gathered_tensors.append(
            tensor if indices is None else tf
            .gather(tensor, indices, batch_dims=1))

      return tf.repeat(
          tf.stack(gathered_tensors, axis=1), repeats=type_counts, axis=1)

  # <int32>[batch_size, num_types, seq_length]
  type_ids = tf.stack(type_ids_masked, axis=1)
  type_order = tf.argsort(type_ids, stable=True)
  type_rank = tf.argsort(type_order, stable=True)

  # For each head the correct order to sort the embeddings.
  # <int32>[batch_size, num_attention_heads, seq_length].
  type_order_repeated = tf.repeat(type_order, repeats=type_counts, axis=1)

  # For each head the inverse of the correct order to sort the embeddings.
  # <int32>[batch_size, num_attention_heads, seq_length].
  type_rank_repeated = tf.repeat(type_rank, repeats=type_counts, axis=1)
  input_header = input_header & tf.math.equal(input_mask, 1)

  if sort_after_projection:
    # No need to sort in this case since this will happen in the BucketedTensor.
    type_ids_repeated = tf.repeat(type_ids, repeats=type_counts, axis=1)
    attend_always_repeated = tf.repeat(
        tf.expand_dims(input_header, axis=1), repeats=num_heads, axis=1)
  else:
    type_ids_repeated = _gather_and_repeat(type_ids_masked, 'type_id_gather')
    attend_always_repeated = _gather_and_repeat([input_header],
                                                'attend_always_gather')

  if header_size is None:
    header_size = bucket_size

  args = {
      'order_indices': type_order_repeated,
      'bucket_size': bucket_size,
      'header_size': header_size,
      'num_heads': num_heads,
      'sort_after_projection': sort_after_projection,
  }
  ids = _create_bucketed_tensor(type_ids_repeated, 'type_id', **args)
  attend_always = _create_bucketed_tensor(attend_always_repeated,
                                          'attend_always', **args)
  head_full_mask = _compute_bucketed_attention_mask('head', 'full', ids,
                                                    attend_always)
  tail_head_mask = _compute_bucketed_attention_mask('tail', 'head', ids,
                                                    attend_always)
  tail_window_mask = _compute_bucketed_attention_mask('tail', 'window', ids,
                                                      attend_always)
  return functools.partial(
      _bucketed_attention_layer,
      sort_after_projection=sort_after_projection,
      gather_and_repeat=_gather_and_repeat,
      bucketed_tensor_args=args,
      header_size=header_size,
      type_ranks=type_rank_repeated,
      head_full_mask=head_full_mask,
      tail_head_mask=tail_head_mask,
      tail_window_mask=tail_window_mask)


def _bucketed_attention_layer(
    from_tensor,
    to_tensor,
    bucketed_tensor_args,
    header_size,
    gather_and_repeat,
    sort_after_projection,
    type_ranks,
    head_full_mask,
    tail_head_mask,
    tail_window_mask,
    num_attention_heads,
    size_per_head = 512,
    query_act = None,
    key_act = None,
    value_act = None,
    attention_mask=None,
    input_mask=None,
    attention_probs_dropout_prob = 0.0,
    initializer_range = 0.02,
    softmax_temperature = 1.0,
    to_proj_length = None):
  """Performs sparse multi-headed attention from `from_tensor` to `to_tensor`.

  If `from_tensor` and `to_tensor` are the same, then this is self-attention.
  Each timestep in `from_tensor` attends to the corresponding sequence in
  `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list of
  tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  We use the `token_type_ids` attributes to reorder and reshape the matrix to
  guarantee that tokens with the same token types will be positioned nearby.
  Then, the query and key tensors are dot-producted and scaled for positions
  in consecutive buckets or to/from the first bucket. These are softmaxed to
  obtain attention probabilities. The value tensors are then interpolated by
  these probabilities, then concatenated back to a single tensor and returned.

  In order to make the computation simple we create auxiliary views of the
  Q, K and V tensors. The `head` and `tail` views contain the first bucket
  and all the rest respectively. The `window` view is a concatenation of the
  `tail` view as well as the same view rotated in both directions so that each
  bucket can be aligned with the one before and after it in one matrix op.

  Scalar dimensions referenced here:
    B = `batch_size` (number of sequences)
    S = `seq_length`
    F = `from_tensor` bucket_size length
    T = `to_tensor` bucket_size length
    R = `header_size`
    H = `num_attention_heads`
    N = `num_buckets`
    I = `input_embedding_size`
    E = `embedding_size`

  In practice, the multi-headed attention are done with tf.einsum as follows:
    from_tensor: [BSI]
    to_tensor: [BSI]
    Wq, Wk, Wv: [IHE]
    Q:[BHSE] = einsum('BSI,IHE->BHSE', from_tensor, Wq)
    K:[BHSE] = einsum('BSI,IHE->BHSE', to_tensor, Wk)
    V:[BHSE] = einsum('BSI,IHE->BHSE', to_tensor, Wv)

    Q, K, V = BucketedTensor(Q), BucketedTensor(K), BucketedTensor(V)

    attention_prob_from_head:[BHFT] = softmax(einsum('BHFE,BHTE->BHFT',
      Q.first, K.full) / sqrt(E))

    attention_score_to_head:[BHNFT] = einsum('BHNFE,BHTE->BHNFT',
      Q.tail, K.first) / sqrt(E)
    attention_score_to_window:[BHNFT] = einsum('BHNFE,BHNTE->BHNFT',
      Q.tail, K.window) / sqrt(E)
    attention_prob_from_taiil = softmax(concat(
      attention_score_to_head, attention_score_to_window))

    attention_prob_to_head = attention_prob_from_tail[:, :, :, :, :R]
    attention_prob_to_window = attention_prob_from_tail[:, :, :, :, R:]

    context_layer_from_head:[BHFE] = einsum('BHFT,BHTE->BHFE',
      attention_prob_from_head, V.full)
    context_layer_to_head:[BHNFE] = einsum('BHNFT,BHTE->BHNFE',
      attention_prob_to_head, V.head)
    context_layer_to_window:[BHNFE] = einsum('BHNFT,BHNTE->BHNFE',
      attention_prob_to_window, V.window)

    context_layer = concat(
      context_layer_from_head,
      reshape(context_layer_to_head + context_layer_to_window))

    return reorder(context_layer)

  Args:
    from_tensor: <float32>[batch_size, from_seq_length, from_width].
    to_tensor: <float32>[batch_size, to_seq_length, to_width].
    bucketed_tensor_args: dict of arguments to build bucketed_tensors.
    header_size: Size of the first bucket that will attend to/from everything.
      If None is passed will use the same as `bucket_size`.
    gather_and_repeat: Helper function that sorts input tensor only if necessary
      for that head. Used when `sort_after_projection` is False.
    sort_after_projection: Sorting can happen on the layer input or after
      applying the projection to keys, queries and values. Depending on the
      accelerator, one option could be more convenient.
    type_ranks: <int32>[batch_size, num_heads, seq_length]
    head_full_mask: <float32>[batch_size, num_heads, header_size, seq_length]
      additive mask between head and full tokens.
    tail_head_mask: <float32>[batch_size, num_heads, num_buckets, bucket_size,
      header_size] additive mask between tail and head tokens.
    tail_window_mask: <float32>[batch_size, num_heads, num_buckets, bucket_size,
      3 * bucket_size] additive mask between tail and window tokens.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Dimension of embeddings for keys, queries and values.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_mask: (optional) Unsupported attribute, added for duck typing.
    input_mask: (optional) Unsupported attribute, added for duck typing.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    softmax_temperature: The temperature for the softmax attention.
    to_proj_length: (optional) Unsupported attribute, added for duck typing.

  Returns:
    context_layer: <float32>[batch_size, from_seq_length, num_attention_heads,
                            size_per_head],
    (
      attention_prob_from_head: <float32>[batch_size, num_attention_heads,
        header_size, to_seq_length],
      attention_prob_from_tail: <float32>[batch_size, num_attention_heads,
        num_buckets, bucket_size, header_size + 3 * bucket_size]
    )

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  del attention_mask
  if to_proj_length is not None:
    raise ValueError(f'Unsupported parameter: to_proj_length: {to_proj_length}')
  del input_mask
  num_heads = num_attention_heads
  seq_length = from_tensor.shape[1]
  args = dict(embedding_size=size_per_head, **bucketed_tensor_args)

  if not sort_after_projection:
    same_sequences = from_tensor is to_tensor
    from_tensor = gather_and_repeat([from_tensor], 'from_tensor_gather')
    if same_sequences:
      to_tensor = from_tensor
    else:
      to_tensor = gather_and_repeat([to_tensor], 'to_tensor_gather')

  # <float32>[batch_size, num_heads, seq_length, key_size]
  query_layer = _dense_layer_3d(from_tensor, query_act, 'query', num_heads,
                                size_per_head, initializer_range,
                                sort_after_projection)
  key_layer = _dense_layer_3d(to_tensor, key_act, 'key', num_heads,
                              size_per_head, initializer_range,
                              sort_after_projection)
  value_layer = _dense_layer_3d(to_tensor, value_act, 'value', num_heads,
                                size_per_head, initializer_range,
                                sort_after_projection)

  keys = _create_bucketed_tensor(key_layer, 'key', **args)
  values = _create_bucketed_tensor(value_layer, 'value', **args)
  queries = _create_bucketed_tensor(query_layer, 'query', **args)
  tau = softmax_temperature * math.sqrt(float(size_per_head))

  # Attention probs from the first bucket to all the rest.
  # <float32>[batch_size, num_heads, header_size, seq_length]
  attention_scores_from_head = _attention_scores('head', 'full', head_full_mask,
                                                 queries, keys)
  attention_prob_from_head = tf.nn.softmax(attention_scores_from_head / tau)

  # Self attention probs from the second bucket onwards as well as to the first.
  # <float32>[batch_size, num_heads,
  #           num_buckets, bucket_size, header_size + 3 * bucket_size]
  attention_scores_from_tail = tf.concat([
      _attention_scores('tail', 'head', tail_head_mask, queries, keys),
      _attention_scores('tail', 'window', tail_window_mask, queries, keys)
  ],
                                         axis=-1)
  attention_prob_from_tail = tf.nn.softmax(attention_scores_from_tail / tau)
  # <float32>[batch_size, num_heads, num_buckets, bucket_size, header_size]
  attention_prob_to_head = attention_prob_from_tail[:, :, :, :, :header_size]
  # <float32>[batch_size, num_heads, num_buckets, bucket_size, 3 * bucket_size]
  attention_prob_to_window = attention_prob_from_tail[:, :, :, :, header_size:]

  # <float32>[batch_size, num_heads, num_buckets, bucket_size, size_per_head]
  context_layer_to_head = tf.einsum(
      'BHNFT,BHTE->BHNFE',
      _dropout(attention_prob_to_head, attention_probs_dropout_prob),
      values.head,
      name='attention_value_to_head')

  # <float32>[batch_size, num_heads, num_buckets, bucket_size, size_per_head]
  context_layer_to_window = tf.einsum(
      'BHNFT,BHNTE->BHNFE',
      _dropout(attention_prob_to_window, attention_probs_dropout_prob),
      values.window,
      name='attention_value_to_window')

  # Change shape back into full sequence length.
  # <float32>[batch_size, num_heads, seq_length - header_size, size_per_head]
  context_layer_from_tail = tf.reshape(
      context_layer_to_head + context_layer_to_window,
      shape=[-1, num_heads, seq_length - header_size, size_per_head])

  # <float32>[batch_size, num_heads, header_size, size_per_head]
  context_layer_from_head = tf.einsum(
      'BHFT,BHTE->BHFE',
      _dropout(attention_prob_from_head, attention_probs_dropout_prob),
      values.full,
      name='attention_value_from_head')

  context_layer = tf.concat([context_layer_from_head, context_layer_from_tail],
                            axis=2)

  # Reorder applying the inverse permutation.
  context_layer = tf.gather(
      context_layer, type_ranks, batch_dims=2, name='reorder')

  # These ops are only used/executed for testing or inspecting attention probs.
  attention_probs = (attention_prob_from_head, attention_prob_from_tail)

  # Transpose to match the standard attention API.
  return tf.transpose(context_layer, perm=[0, 2, 1, 3]), attention_probs
