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
"""TAPAS BERT model utils for classification."""

from tapas.models import segmented_tensor
import tensorflow.compat.v1 as tf

EPSILON_ZERO_DIVISION = 1e-10
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


def classification_initializer():
  """Classification layer initializer."""
  return tf.truncated_normal_initializer(stddev=0.02)


def compute_token_logits(output_layer, temperature,
                         init_cell_selection_weights_to_zero):
  """Computes logits per token.

  Args:
    output_layer: <float>[batch_size, seq_length, hidden_dim] Output of the
      encoder layer.
    temperature: float Temperature for the Bernoulli distribution.
    init_cell_selection_weights_to_zero: Whether the initial weights should be
      set to 0. This ensures that all tokens have the same prior probability.

  Returns:
    <float>[batch_size, seq_length] Logits per token.
  """
  hidden_size = output_layer.shape.as_list()[-1]
  output_weights = tf.get_variable(
      "output_weights", [hidden_size],
      initializer=tf.zeros_initializer()
      if init_cell_selection_weights_to_zero else classification_initializer())
  output_bias = tf.get_variable(
      "output_bias", shape=(), initializer=tf.zeros_initializer())
  logits = (tf.einsum("bsj,j->bs", output_layer, output_weights) +
            output_bias) / temperature
  return logits


# TODO(eisenjulian): Move more methods from tapas_classifier_model
def compute_column_logits(output_layer,
                          cell_index,
                          cell_mask,
                          init_cell_selection_weights_to_zero,
                          allow_empty_column_selection):
  """Computes logits for each column.

  Args:
    output_layer: <float>[batch_size, seq_length, hidden_dim] Output of the
      encoder layer.
    cell_index: segmented_tensor.IndexMap [batch_size, seq_length] Index that
      groups tokens into cells.
    cell_mask: <float>[batch_size, max_num_rows * max_num_cols] Input mask per
      cell, 1 for cells that exists in the example and 0 for padding.
    init_cell_selection_weights_to_zero: Whether the initial weights should be
      set to 0. This is also applied to column logits, as they are used to
      select the cells. This ensures that all columns have the same prior
      probability.
    allow_empty_column_selection: Allow to select no column.

  Returns:
    <float>[batch_size, max_num_cols] Logits per column. Logits will be set to
      a very low value (such that the probability is 0) for the special id 0
      (which means "outside the table") or columns that do not apear in the
      table.
  """
  hidden_size = output_layer.shape.as_list()[-1]
  column_output_weights = tf.get_variable(
      "column_output_weights", [hidden_size],
      initializer=tf.zeros_initializer()
      if init_cell_selection_weights_to_zero else classification_initializer())
  column_output_bias = tf.get_variable(
      "column_output_bias", shape=(), initializer=tf.zeros_initializer())
  token_logits = (
      tf.einsum("bsj,j->bs", output_layer, column_output_weights) +
      column_output_bias)

  # Average the logits per cell and then per column.
  # Note that by linearity it doesn't matter if we do the averaging on the
  # embeddings or on the logits. For performance we do the projection first.
  # [batch_size, max_num_cols * max_num_rows]
  cell_logits, cell_logits_index = segmented_tensor.reduce_mean(
      token_logits, cell_index)

  column_index = cell_index.project_inner(cell_logits_index)
  # [batch_size, max_num_cols]
  column_logits, out_index = segmented_tensor.reduce_sum(
      cell_logits * cell_mask, column_index)
  cell_count, _ = segmented_tensor.reduce_sum(cell_mask, column_index)
  column_logits /= cell_count + EPSILON_ZERO_DIVISION

  # Mask columns that do not appear in the example.
  is_padding = tf.logical_and(cell_count < 0.5,
                              tf.not_equal(out_index.indices, 0))
  column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(is_padding, tf.float32)

  if not allow_empty_column_selection:
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(
        tf.equal(out_index.indices, 0), tf.float32)

  return column_logits
