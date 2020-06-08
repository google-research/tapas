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

from absl.testing import absltest
import numpy as np
from tapas.models import segmented_tensor
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class SegmentedTensorTest(tf.test.TestCase):

  def _prepare_tables(self):
    """Prepares two tables, both with three distinct rows.

    The first table has two columns:
      1.0, 2.0 | 3.0
      2.0, 0.0 | 1.0
      1.0, 3.0 | 4.0

    The second table has three columns:
      1.0 | 2.0 | 3.0
      2.0 | 0.0 | 1.0
      1.0 | 3.0 | 4.0

    Returns:
      SegmentedTensors with the tables.
    """
    values = tf.constant([
        [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
        [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
    ])
    row_index = segmented_tensor.IndexMap(
        indices=[
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        ],
        num_segments=3,
        batch_dims=1)
    col_index = segmented_tensor.IndexMap(
        indices=[
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ],
        num_segments=3,
        batch_dims=1)
    return values, row_index, col_index

  def test_product_index(self):
    _, row_index, col_index = self._prepare_tables()
    cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)
    row_index_proj = cell_index.project_outer(cell_index)
    col_index_proj = cell_index.project_inner(cell_index)

    with self.session() as sess:
      ind = sess.run(cell_index.indices)
      self.assertEqual(sess.run(cell_index.num_segments), 9)

      # Projections should give back the original indices.
      self.assertAllEqual(
          sess.run(row_index.indices), sess.run(row_index_proj.indices))
      self.assertEqual(
          sess.run(row_index.num_segments),
          sess.run(row_index_proj.num_segments))
      self.assertEqual(row_index.batch_dims, row_index_proj.batch_dims)
      self.assertAllEqual(
          sess.run(col_index.indices), sess.run(col_index_proj.indices))
      self.assertEqual(col_index.batch_dims, col_index_proj.batch_dims)

      # The first and second "column" are identified in the first table.
      for i in range(3):
        self.assertEqual(ind[0, i, 0], ind[0, i, 1])
        self.assertNotEqual(ind[0, i, 0], ind[0, i, 2])

      # All rows are distinct in the first table.
      for i, i_2 in zip(range(3), range(3)):
        for j, j_2 in zip(range(3), range(3)):
          if i != i_2 and j != j_2:
            self.assertNotEqual(ind[0, i, j], ind[0, i_2, j_2])

      # All cells are distinct in the second table.
      for i, i_2 in zip(range(3), range(3)):
        for j, j_2 in zip(range(3), range(3)):
          if i != i_2 or j != j_2:
            self.assertNotEqual(ind[1, i, j], ind[1, i_2, j_2])

  def test_flatten(self):
    _, row_index, col_index = self._prepare_tables()
    row_index_flat = segmented_tensor.flatten(row_index)
    col_index_flat = segmented_tensor.flatten(col_index)

    shape = [3, 4, 5]
    batched_index = segmented_tensor.IndexMap(
        indices=tf.fill(shape, tf.constant(0, dtype=tf.int32)),
        num_segments=1,
        batch_dims=3)
    batched_index_flat = segmented_tensor.flatten(batched_index)

    with self.session() as sess:
      self.assertAllEqual(
          sess.run(row_index_flat.indices),
          [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
      self.assertAllEqual(
          sess.run(col_index_flat.indices),
          [0, 0, 1, 0, 0, 1, 0, 0, 1, 3, 4, 5, 3, 4, 5, 3, 4, 5])
      self.assertEqual(
          sess.run(batched_index_flat.num_segments), np.prod(shape))
      self.assertAllEqual(
          sess.run(batched_index_flat.indices), range(np.prod(shape)))

  def test_reduce_sum(self):
    values, row_index, col_index = self._prepare_tables()
    cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)
    row_sum, _ = segmented_tensor.reduce_sum(values, row_index)
    col_sum, _ = segmented_tensor.reduce_sum(values, col_index)
    cell_sum, _ = segmented_tensor.reduce_sum(values, cell_index)
    with self.session() as sess:
      self.assertAllClose(sess.run(row_sum), [[6.0, 3.0, 8.0], [6.0, 3.0, 8.0]])
      self.assertAllClose(sess.run(col_sum), [[9.0, 8.0, 0.0], [4.0, 5.0, 8.0]])
      self.assertAllClose(
          sess.run(cell_sum), [[3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 4.0, 4.0, 0.0],
                               [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0]])

  def test_reduce_mean(self):
    values, row_index, col_index = self._prepare_tables()
    cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)
    row_mean, _ = segmented_tensor.reduce_mean(values, row_index)
    col_mean, _ = segmented_tensor.reduce_mean(values, col_index)
    cell_mean, _ = segmented_tensor.reduce_mean(values, cell_index)
    with self.session() as sess:
      self.assertAllClose(
          sess.run(row_mean), [[6.0 / 3.0, 3.0 / 3.0, 8.0 / 3.0],
                               [6.0 / 3.0, 3.0 / 3.0, 8.0 / 3.0]])
      self.assertAllClose(
          sess.run(col_mean),
          [[9.0 / 6.0, 8.0 / 3.0, 0.0], [4.0 / 3.0, 5.0 / 3.0, 8.0 / 3.0]])
      self.assertAllClose(
          sess.run(cell_mean),
          [[3.0 / 2.0, 3.0, 0.0, 2.0 / 2.0, 1.0, 0.0, 4.0 / 2.0, 4.0, 0.0],
           [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0]])

  def test_reduce_max(self):
    values = [2, 1, 0, 3]
    index = segmented_tensor.IndexMap(indices=[0, 1, 0, 1], num_segments=2)
    maximum, _ = segmented_tensor.reduce_max(values, index)
    with self.session() as sess:
      self.assertAllEqual(sess.run(maximum), [2, 3])

  def test_reduce_sum_vectorized(self):
    values = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    index = segmented_tensor.IndexMap(
        indices=[0, 0, 1], num_segments=2, batch_dims=0)
    sums, new_index = segmented_tensor.reduce_sum(values, index)
    with self.session() as sess:
      self.assertAllClose(sess.run(sums), [[3.0, 5.0, 7.0], [3.0, 4.0, 5.0]])
      self.assertAllEqual(sess.run(new_index.indices), [0, 1])
      self.assertEqual(sess.run(new_index.num_segments), 2)
      self.assertEqual(new_index.batch_dims, 0)

  def test_gather(self):
    values, row_index, col_index = self._prepare_tables()
    cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)

    # Compute sums and then gather. The result should have the same shape as
    # the original table and each element should contain the sum the values in
    # its cell.
    sums, _ = segmented_tensor.reduce_sum(values, cell_index)
    cell_sum = segmented_tensor.gather(sums, cell_index)
    cell_sum.shape.assert_is_compatible_with(values.shape)

    with self.session() as sess:
      self.assertAllClose(
          sess.run(cell_sum),
          [[[3.0, 3.0, 3.0], [2.0, 2.0, 1.0], [4.0, 4.0, 4.0]],
           [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]]])

  def test_gather_vectorized(self):
    values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    index = segmented_tensor.IndexMap(
        indices=[[0, 1], [1, 0]], num_segments=2, batch_dims=1)
    result = segmented_tensor.gather(values, index)
    with self.session() as sess:
      self.assertAllEqual(
          sess.run(result), [[[1, 2], [3, 4]], [[7, 8], [5, 6]]])

  def test_range_index_map(self):
    batch_shape = [3, 4]
    num_segments = 5
    index = segmented_tensor.range_index_map(batch_shape, num_segments)
    with self.session() as sess:
      self.assertEqual(num_segments, sess.run(index.num_segments))
      self.assertEqual(2, index.batch_dims)
      indices = sess.run(index.indices)
      self.assertAllEqual(indices.shape, [3, 4, 5])
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(indices[i, j, :], range(num_segments))


if __name__ == '__main__':
  absltest.main()
