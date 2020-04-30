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
"""A tensor supporting reduction over irregularly grouped entries.

A segmented tensor is a tensor with a set of indices {0, ..., num_segments - 1}
and an index map that assigns an index to each element of the tensor. Two
elements with the same index are considered grouped together. The set of all
elements with index `k` is called the segment over k.

Segmented tensors support reductions over segments (reduce_mean, reduce_sum,
etc.). A typical example is performing reductions over table cells:

```
# Prepare a tf.Tensor with table values.
values = ...

# Prepare the table indices, either by rows or columns. The shape of `row_ids`
# and `col_ids` has to be a prefix of the shape of `values`.
row_index = segmented_tensor.IndexMap(indices=row_ids, num_segments=max_rows)
col_index = segmented_tensor.IndexMap(indices=col_ids, num_segments=max_cols)

# Combine the indices to get a table indexed by cell. The result has
# `num_segments` equal to row_ids * col_ids.
cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)

# Compute the averages per cell. The result is a `Tensor` with shape
# [max_rows * max_cols, ..] together with an index map on it. The index map is
# equal to range(max_rows * max_cols).
cell_averages, _ = segmented_tensor.reduce_mean(values, cell_index)

# Gather the results to get back a Tensor with the same shape as `cell_index`.
# If there are multiple elements in the same cell they will have the same value.
token_to_its_cell_average = segmented_tensor.gather(cell_averages, cell_index)
```

Batching is supported by setting `batch_dims`. The first `batch_dims` dimensions
will be treated as the batch. Elements of different batches are never grouped
together, not even if they have the same index.
"""

import tensorflow.compat.v1 as tf


class IndexMap(object):
  """Index grouping entries within a tensor."""

  def __init__(self, indices, num_segments, batch_dims=0):
    """Creates an index.

    Args:
      indices: <int32> Tensor of indices, same shape as `values`.
      num_segments: <int32> Scalar tensor, the number of segments. All elements
        in a batched segmented tensor must have the same number of segments
        (although many segments can be empty).
      batch_dims: Python integer, the number of batch dimensions. The first
        `batch_dims` dimensions of a SegmentedTensor are treated as batch
        dimensions. Segments in different batch elements are always distinct
        even if they have the same index.
    """
    self.indices = tf.convert_to_tensor(indices)
    self.num_segments = tf.convert_to_tensor(num_segments)
    self.batch_dims = batch_dims

  def batch_shape(self):
    return tf.shape(self.indices)[:self.batch_dims]


class ProductIndexMap(IndexMap):
  """The product of two indices."""

  def __init__(self, outer_index, inner_index):
    """Combines indices i and j into pairs (i, j).

    The result is an index where each segment (i, j) is the intersection of
    segments i and j. For example if the inputs represent table cells indexed by
    respectively rows and columns the output will be a table indexed by
    (row, column) pairs, i.e. by cell.

    The implementation combines indices {0, .., n - 1} and {0, .., m - 1} into
    {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segements` * `inner_index.num_segments`.

    Args:
      outer_index: IndexMap.
      inner_index: IndexMap, must have the same shape as `outer_index`.
    """
    if outer_index.batch_dims != inner_index.batch_dims:
      raise ValueError('outer_index.batch_dims and inner_index.batch_dims '
                       'must be the same.')

    super(ProductIndexMap, self).__init__(
        indices=(inner_index.indices +
                 outer_index.indices * inner_index.num_segments),
        num_segments=inner_index.num_segments * outer_index.num_segments,
        batch_dims=inner_index.batch_dims)
    self.outer_index = outer_index
    self.inner_index = inner_index

  def project_outer(self, index):
    """Projects an index with the same index set onto the outer components."""
    return IndexMap(
        indices=tf.floor_div(index.indices, self.inner_index.num_segments),
        num_segments=self.outer_index.num_segments,
        batch_dims=index.batch_dims)

  def project_inner(self, index):
    """Projects an index with the same index set onto the inner components."""
    return IndexMap(
        indices=tf.floormod(index.indices, self.inner_index.num_segments),
        num_segments=self.inner_index.num_segments,
        batch_dims=index.batch_dims)


def gather(values, index, name='segmented_gather'):
  """Gathers from `values` using the index map.

  For each element in the domain of the index map this operation looks up a
  value for that index in `values`. Two elements from the same segment always
  get assigned the same value.

  Args:
    values: [B1, ..., Bn, num_segments, V1, ...] Tensor with segment values.
    index: [B1, ..., Bn, I1, ..., Ik] IndexMap.
    name: Name for the TensorFlow operation.

  Returns:
    [B1, ..., Bn, I1, ..., Ik, V1, ...] Tensor with the gathered values.
  """
  return tf.gather(
      values, index.indices, batch_dims=index.batch_dims, name=name)


def flatten(index, name='segmented_flatten'):
  """Flattens a batched index map to a 1d index map.

  This operation relabels the segments to keep batch elements distinct. The k-th
  batch element will have indices shifted by `num_segments` * (k - 1). The
  result is a tensor with `num_segments` multiplied by the number of elements
  in the batch.

  Args:
    index: IndexMap to flatten.
    name: Name for the TensorFlow operation.

  Returns:
    The flattened IndexMap.
  """
  with tf.variable_scope(name):
    batch_size = tf.reduce_prod(index.batch_shape())
    offset = tf.range(batch_size) * index.num_segments
    offset = tf.reshape(offset, index.batch_shape())
    for _ in range(index.batch_dims, index.indices.shape.rank):
      offset = tf.expand_dims(offset, -1)

    indices = offset + index.indices
    return IndexMap(
        indices=tf.reshape(indices, [-1]),
        num_segments=index.num_segments * batch_size,
        batch_dims=0)


def range_index_map(batch_shape, num_segments, name='range_index_map'):
  """Constructs an index map equal to range(num_segments)."""
  with tf.variable_scope(name):
    batch_shape = tf.convert_to_tensor(batch_shape)
    batch_shape.shape.assert_has_rank(1)
    num_segments = tf.convert_to_tensor(num_segments)
    num_segments.shape.assert_has_rank(0)

    indices = tf.range(num_segments)
    shape = tf.concat([
        tf.ones_like(batch_shape, dtype=tf.int32),
        tf.expand_dims(num_segments, axis=0)
    ],
                      axis=0)
    indices = tf.reshape(indices, shape)
    multiples = tf.concat([batch_shape, [1]], axis=0)
    indices = tf.tile(indices, multiples)
    return IndexMap(
        indices=indices,
        num_segments=num_segments,
        batch_dims=batch_shape.shape.as_list()[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
  """Applies a segment reduction segment-wise."""
  with tf.variable_scope(name):
    # Flatten the batch dimensions, as segments ops do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = tf.shape(values)[index.indices.shape.rank:]
    flattened_shape = tf.concat([[-1], vector_shape], axis=0)
    flat_values = tf.reshape(values, flattened_shape)
    segment_means = segment_reduce_fn(
        data=flat_values,
        segment_ids=flat_index.indices,
        num_segments=flat_index.num_segments)

    # Unflatten the values.
    new_shape = tf.concat(
        [index.batch_shape(), [index.num_segments], vector_shape], axis=0)
    output_values = tf.reshape(segment_means, new_shape)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def reduce_mean(values, index, name='segmented_reduce_mean'):
  """Averages a tensor over its segments.

  Outputs 0 for empty segments.

  This operations computes the mean over segments, with support for:
    - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
      a batch can have different indices.
    - Vectorization using the last dimension [V1, V2, ...]. If they are present
      the output will be a mean of vectors rather than scalars.

  Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

  Args:
    values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
      averaged.
    index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
    name: Name for the TensorFlow ops.

  Returns:
    A pair (output_values, output_index) where `output_values` is a tensor
    of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..] and `index` is an
    IndexMap with shape [B1, B2, ..., Bn, num_segments].
  """
  return _segment_reduce(values, index, tf.math.unsorted_segment_mean, name)


def reduce_sum(values, index, name='segmented_reduce_sum'):
  """Sums a tensor over its segments.

  Outputs 0 for empty segments.

  This operations computes the sum over segments, with support for:
    - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
      a batch can have different indices.
    - Vectorization using the last dimension [V1, V2, ...]. If they are present
      the output will be a sum of vectors rather than scalars.

  Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

  Args:
    values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
      averaged.
    index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
    name: Name for the TensorFlow ops.

  Returns:
    A pair (output_values, output_index) where `output_values` is a tensor
    of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..] and `index` is an
    IndexMap with shape [B1, B2, ..., Bn, num_segments].
  """
  return _segment_reduce(values, index, tf.math.unsorted_segment_sum, name)


def reduce_max(values, index, name='segmented_reduce_sum'):
  """Computes the maximum over segments.

  This operations computes the maximum over segments, with support for:
    - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
      a batch can have different indices.
    - Vectorization using the last dimension [V1, V2, ...]. If they are present
      the output will be an element-wise maximum of vectors rather than scalars.

  Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

  Args:
    values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
      averaged.
    index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
    name: Name for the TensorFlow ops.

  Returns:
    A pair (output_values, output_index) where `output_values` is a tensor
    of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..] and `index` is an
    IndexMap with shape [B1, B2, ..., Bn, num_segments].
  """
  return _segment_reduce(values, index, tf.math.unsorted_segment_max, name)
