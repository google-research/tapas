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
"""TAPAS BERT model for classification."""

import enum
import json
from typing import Iterable, Text, Optional, List

import dataclasses

from tapas.datasets import dataset
from tapas.datasets import table_dataset
from tapas.models import segmented_tensor
from tapas.models.bert import modeling
from tapas.models.bert import optimization
from tapas.models.bert import table_bert
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


_EPSILON_ZERO_DIVISION = 1e-10
_CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


class AverageApproximationFunction(str, enum.Enum):
  RATIO = "ratio"
  FIRST_ORDER = "first_order"
  SECOND_ORDER = "second_order"


@dataclasses.dataclass
class TapasClassifierConfig:
  """Helper class for configuration of Tapas model.

  bert_config: Config object for general bert hyper params.
  init_checkpoint: Location of the model checkpoint.
  learning_rate: Optimizer learning rate.
  num_train_steps: Total number of training steps for optimizer schedule.
  num_warmup_steps: Number of training steps to warm up optimizer.
  use_tpu: Use TPU for training.
  positive_weight: Weight for positive labels.
  num_aggregation_labels: The number of aggregation classes to predict.
  num_classification_labels: The number of classes to predict.
  aggregation_loss_importance: Importance weight for the aggregation loss.
  use_answer_as_supervision: Whether to use the answer as the only supervision
    for aggregation examples.
  answer_loss_importance: Importance weight for the regression loss.
  use_normalized_answer_loss: Normalize loss by max of predicted and expected
    value.
  huber_loss_delta: Delta parameter used to calculate the regression loss.
  temperature: Scales cell logits to control the skewness of probabilities.
  agg_temperature: Scales aggregation logits to control the skewness of
    probabilities.
  use_gumbel_for_cells: Applies Gumbel-Softmax to cell selection.
  use_gumbel_for_agg: Applies Gumbel-Softmax to aggregation selection.
  average_approximation_function: Method to calculate expected average of
    cells in the relaxed case.
  cell_select_pref: Preference for cell selection in ambiguous cases.
  answer_loss_cutoff: Ignore examples with answer loss larger than cutoff.
  grad_clipping: If not None, clip the gradient norm to this value
  max_num_rows: Maximum number of rows.
  max_num_columns: Maximum number of columns.
  average_logits_per_cell: Wheher to average logits per cell.
  select_one_column: Whether to constrain the model to only select cells from
    a single column.
  allow_empty_column_selection: Allow not to select any column.
  disabled_features: Set of embeddings ids to disable in the input layer.
    Posible values are "segment_ids", "column_ids", "row_ids", "prev_label_ids",
    "column_ranks", "inv_column_ranks", "numeric_relations"
  init_cell_selection_weights_to_zero: Whether to initialize cell selection.
    weights to 0 so that the initial probabilities are 50%.
  disable_position_embeddings: Disable positional embeddings in the input layer.
  disable_per_token_loss: Disable any (strong or weak) supervision on cells.
  """

  bert_config: modeling.BertConfig
  init_checkpoint: Text
  learning_rate: float
  num_train_steps: Optional[int]
  num_warmup_steps: Optional[int]
  use_tpu: bool
  positive_weight: float
  num_aggregation_labels: int
  num_classification_labels: int
  aggregation_loss_importance: float
  use_answer_as_supervision: bool
  answer_loss_importance: Optional[float]
  use_normalized_answer_loss: bool
  huber_loss_delta: Optional[float]
  temperature: float
  agg_temperature: float
  use_gumbel_for_cells: bool
  use_gumbel_for_agg: bool
  average_approximation_function: AverageApproximationFunction
  cell_select_pref: Optional[float]
  answer_loss_cutoff: Optional[float]
  grad_clipping: Optional[float]
  max_num_rows: int
  max_num_columns: int
  average_logits_per_cell: bool
  select_one_column: bool
  allow_empty_column_selection: bool = True
  disabled_features: Optional[List[Text]] = None
  init_cell_selection_weights_to_zero: bool = False
  disable_position_embeddings: bool = False
  disable_per_token_loss: bool = False

  def to_json_string(self):
    """Serializes this instance to a JSON string."""

    class EnhancedJSONEncoder(json.JSONEncoder):

      def default(self, o):
        if dataclasses.is_dataclass(o):
          return dataclasses.asdict(o)
        if isinstance(o, modeling.BertConfig):
          return o.to_dict()
        return super().default(o)

    return json.dumps(self, indent=2, sort_keys=True, cls=EnhancedJSONEncoder)

  def to_json_file(self, json_file):
    """Serializes this instance to a JSON file."""
    with tf.io.gfile.GFile(json_file, "w") as writer:
      writer.write(self.to_json_string() + "\n")

  @classmethod
  def from_dict(cls, json_object, for_prediction=False):
    """Constructs a config from a Python dictionary of parameters."""
    json_object = dict(json_object)
    # Overwrite json bert config with config object.
    json_object["bert_config"] = modeling.BertConfig.from_dict(
        json_object["bert_config"])
    # Delete deprecated option, if present.
    # TODO See of we can filter everything that's not an argument.
    if "restrict_attention" in json_object:
      del json_object["restrict_attention"]
    if for_prediction:
      # Disable training-only option to reduce input requirements.
      json_object["use_answer_as_supervision"] = False
    return TapasClassifierConfig(**json_object)

  @classmethod
  def from_json_file(cls, json_file, for_prediction=False):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text), for_prediction=for_prediction)


def _get_probs(dist):
  """Helper to extract probability from a distribution."""
  # In tensorflow_probabiliy 0.7 this attribute was filled on __init__ method
  if dist.probs is not None:
    return dist.probs
  # From 0.8 onwards the probs is not filled and a function should be used
  return dist.probs_parameter()


def _classification_initializer():
  """Classification layer initializer."""
  return tf.truncated_normal_initializer(stddev=0.02)


def _calculate_aggregation_logits(output_layer_aggregation, output_weights_agg,
                                  output_bias_agg):
  """Calculates the aggregation logits.

  Args:
    output_layer_aggregation: <float32>[batch_size, hidden_size]
    output_weights_agg: <float32>[num_aggregation_labels, hidden_size_agg]
    output_bias_agg: <float32>[num_aggregation_labels]

  Returns:
    logits_aggregation: <float32>[batch_size, num_aggregation_labels]
  """
  logits_aggregation = tf.matmul(
      output_layer_aggregation, output_weights_agg, transpose_b=True)
  logits_aggregation = tf.nn.bias_add(logits_aggregation, output_bias_agg)
  return logits_aggregation


def _calculate_aggregation_loss_known(logits_aggregation, aggregate_mask,
                                      aggregation_function_id,
                                      config):
  """Calculates aggregation loss when its type is known during training.

  In the weakly supervised setting, the only known information is that for
  cell selection examples, "no aggregation" should be predicted. For other
  examples (those that require aggregation), no loss is accumulated.
  In the setting where aggregation type is always known, standard cross entropy
  loss is accumulated for all examples.

  Args:
    logits_aggregation: <float32>[batch_size, num_aggregation_labels]
    aggregate_mask: <float32>[batch_size]
    aggregation_function_id: <int32>[batch_size, 1]
    config: Configuration for Tapas model.

  Returns:
    aggregation_loss_known: <float32>[batch_size, num_aggregation_labels]
  """
  if config.use_answer_as_supervision:
    # Prepare "no aggregation" targets for cell selection examples.
    target_aggregation = tf.zeros_like(aggregate_mask, dtype=tf.int32)
  else:
    # Use aggregation supervision as the target.
    target_aggregation = aggregation_function_id

  one_hot_labels = tf.one_hot(
      target_aggregation, depth=config.num_aggregation_labels, dtype=tf.float32)
  log_probs = tf.nn.log_softmax(logits_aggregation, axis=-1)
  # <float32>[batch_size]
  per_example_aggregation_intermediate = -tf.reduce_sum(
      one_hot_labels * log_probs, axis=-1)
  if config.use_answer_as_supervision:
    # Accumulate loss only for examples requiring cell selection
    # (no aggregation).
    return per_example_aggregation_intermediate * (1 - aggregate_mask)
  else:
    return per_example_aggregation_intermediate


def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
  """Calculates aggregation loss in the case of answer supervision."""
  dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
  # Index 0 correponds to "no aggregation".
  aggregation_ops_total_mass = tf.reduce_sum(
      _get_probs(dist_aggregation)[:, 1:], axis=1)
  # Predict some aggregation in case of an answer that needs aggregation.
  # This increases the probability of all aggregation functions, in a way
  # similar to MML, but without considering whether the function gives the
  # correct answer.
  return -tf.log(aggregation_ops_total_mass) * aggregate_mask


def _calculate_aggregation_loss(logits_aggregation, aggregate_mask,
                                aggregation_function_id,
                                config):
  """Calculates the aggregation loss per example."""
  per_example_aggregation_loss = _calculate_aggregation_loss_known(
      logits_aggregation, aggregate_mask, aggregation_function_id, config)

  if config.use_answer_as_supervision:
    # Add aggregation loss for numeric answers that need aggregation.
    per_example_aggregation_loss += _calculate_aggregation_loss_unknown(
        logits_aggregation, aggregate_mask)
  return config.aggregation_loss_importance * per_example_aggregation_loss


def _calculate_expected_result(dist_per_cell, numeric_values,
                               numeric_values_scale, input_mask_float,
                               logits_aggregation,
                               config):
  """Calculate the expected result given cell and aggregation probabilities."""
  if config.use_gumbel_for_cells:
    gumbel_dist = tfp.distributions.RelaxedBernoulli(
        # The token logits where already divided by the temperature and used for
        # computing cell selection errors so we need to multiply it again here
        config.temperature,
        logits=dist_per_cell.logits_parameter() * config.temperature)
    scaled_probability_per_cell = gumbel_dist.sample()
  else:
    scaled_probability_per_cell = _get_probs(dist_per_cell)

  # <float32>[batch_size, seq_length]
  scaled_probability_per_cell = (scaled_probability_per_cell /
                                 numeric_values_scale) * input_mask_float
  count_result = tf.reduce_sum(scaled_probability_per_cell, axis=1)
  numeric_values_masked = tf.where(
      tf.is_nan(numeric_values), tf.zeros_like(numeric_values),
      numeric_values)  # Mask non-numeric table values to zero.
  sum_result = tf.reduce_sum(
      scaled_probability_per_cell * numeric_values_masked, axis=1)
  avg_approximation = config.average_approximation_function
  if avg_approximation == AverageApproximationFunction.RATIO:
    average_result = sum_result / (count_result + _EPSILON_ZERO_DIVISION)
  elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
    # The sum of all probabilities exept that correspond to other cells
    ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) \
        - scaled_probability_per_cell + 1
    average_result = tf.reduce_sum(
        numeric_values_masked * scaled_probability_per_cell / ex, axis=1)
  elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
    # The sum of all probabilities exept that correspond to other cells
    ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) \
        - scaled_probability_per_cell + 1
    pointwise_var = scaled_probability_per_cell * \
        (1 - scaled_probability_per_cell)
    var = tf.reduce_sum(pointwise_var, axis=1, keepdims=True) - pointwise_var
    multiplier = (var / tf.math.square(ex) + 1) / ex
    average_result = tf.reduce_sum(
        numeric_values_masked * scaled_probability_per_cell * multiplier,
        axis=1)
  else:
    tf.logging.error("Invalid average_approximation_function: %s",
                     config.average_approximation_function)

  if config.use_gumbel_for_agg:
    gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(
        config.agg_temperature, logits=logits_aggregation[:, 1:])
    # <float32>[batch_size, num_aggregation_labels - 1]
    aggregation_op_only_probs = gumbel_dist.sample()
  else:
    # <float32>[batch_size, num_aggregation_labels - 1]
    aggregation_op_only_probs = tf.nn.softmax(
        logits_aggregation[:, 1:] / config.agg_temperature, axis=-1)
  all_results = tf.concat([
      tf.expand_dims(sum_result, axis=1),
      tf.expand_dims(average_result, axis=1),
      tf.expand_dims(count_result, axis=1)
  ],
                          axis=1)
  expected_result = tf.reduce_sum(
      all_results * aggregation_op_only_probs, axis=1)
  return expected_result


def _calculate_regression_loss(answer, aggregate_mask, dist_per_cell,
                               numeric_values, numeric_values_scale,
                               input_mask_float, logits_aggregation,
                               config):
  """Calculates the regression loss per example.

  Args:
    answer: <float32>[batch_size, 1]
    aggregate_mask: <float32>[batch_size]
    dist_per_cell: Cell selection distribution for each cell.
    numeric_values: <float32>[batch_size, seq_length]
    numeric_values_scale: <float32>[batch_size, seq_length]
    input_mask_float: <float32>[batch_size, seq_length]
    logits_aggregation: <float32>[batch_size, num_aggregation_labels]
      probabilities.
    config: Configuration for Tapas model.

  Returns:
    per_example_answer_loss_scaled: <float32>[batch_size]. Scales answer loss
    for each example in the batch.
    large_answer_loss_mask: <float32>[batch_size]. A mask which is 1 for
    examples for which their answer loss is larger than the answer_loss_cutoff.
  """
  # <float32>[batch_size]
  expected_result = _calculate_expected_result(dist_per_cell, numeric_values,
                                               numeric_values_scale,
                                               input_mask_float,
                                               logits_aggregation, config)
  answer_masked = tf.where(tf.is_nan(answer), tf.zeros_like(answer), answer)
  # <float32>[batch_size]

  if config.use_normalized_answer_loss:
    normalizer = tf.stop_gradient(
        tf.math.maximum(
            tf.math.abs(expected_result), tf.math.abs(answer_masked)) +
        _EPSILON_ZERO_DIVISION)
    normalized_answer_masked = answer_masked / normalizer
    normalized_expected_result = expected_result / normalizer
    per_example_answer_loss = tf.losses.huber_loss(
        normalized_answer_masked * aggregate_mask,
        normalized_expected_result * aggregate_mask,
        delta=tf.cast(config.huber_loss_delta, tf.float32),
        reduction=tf.losses.Reduction.NONE)
  else:
    per_example_answer_loss = tf.losses.huber_loss(
        answer_masked * aggregate_mask,
        expected_result * aggregate_mask,
        delta=tf.cast(config.huber_loss_delta, tf.float32),
        reduction=tf.losses.Reduction.NONE)
  if config.answer_loss_cutoff is None:
    large_answer_loss_mask = tf.ones_like(
        per_example_answer_loss, dtype=tf.float32)
  else:
    large_answer_loss_mask = tf.where(
        per_example_answer_loss > config.answer_loss_cutoff,
        tf.zeros_like(per_example_answer_loss, dtype=tf.float32),
        tf.ones_like(per_example_answer_loss, dtype=tf.float32))
  per_example_answer_loss_scaled = config.answer_loss_importance * (
      per_example_answer_loss * aggregate_mask)
  return per_example_answer_loss_scaled, large_answer_loss_mask


def _calculate_aggregate_mask(answer, output_layer_aggregation, output_bias_agg,
                              output_weights_agg, cell_select_pref, label_ids):
  """Finds examples where the model should select cells with no aggregation.

  Returns a mask that determines for which examples should the model select
  answers directly from the table, without any aggregation function. If the
  answer is a piece of text the case is unambiguous as aggregation functions
  only apply to numbers. If the answer is a number but does not appear in the
  table then we must use some aggregation case. The ambiguous case is when the
  answer is a number that also appears in the table. In this case we use the
  aggregation function probabilities predicted by the model to decide whether
  to select or aggregate. The threshold for this is a hyperparameter
  `cell_select_pref`.

  Args:
    answer: <float32>[batch_size, 1]
    output_layer_aggregation: <float32>[batch_size, hidden_size]
    output_bias_agg: <float32>[num_aggregation_labels]
    output_weights_agg: <float32>[num_aggregation_labels, hidden_size_agg]
    cell_select_pref: Preference for cell selection in ambiguous cases.
    label_ids: <int32>[batch_size, seq_length]

  Returns:
    aggregate_mask: <float32>[batch_size] A mask set to 1 for examples that
      should use aggregation functions.
  """
  # <float32>[batch_size]
  aggregate_mask_init = tf.cast(tf.logical_not(tf.is_nan(answer)), tf.float32)
  logits_aggregation = _calculate_aggregation_logits(output_layer_aggregation,
                                                     output_weights_agg,
                                                     output_bias_agg)
  dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
  aggregation_ops_total_mass = tf.reduce_sum(
      _get_probs(dist_aggregation)[:, 1:], axis=1)
  # Cell selection examples according to current model.
  is_pred_cell_selection = aggregation_ops_total_mass <= cell_select_pref
  # Examples with non-empty cell selection supervision.
  is_cell_supervision_available = tf.reduce_sum(label_ids, axis=1) > 0
  aggregate_mask = tf.where(
      tf.logical_and(is_pred_cell_selection, is_cell_supervision_available),
      tf.zeros_like(aggregate_mask_init, dtype=tf.float32), aggregate_mask_init)
  aggregate_mask = tf.stop_gradient(aggregate_mask)
  return aggregate_mask


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
      if init_cell_selection_weights_to_zero else _classification_initializer())
  output_bias = tf.get_variable(
      "output_bias", shape=(), initializer=tf.zeros_initializer())
  logits = (tf.einsum("bsj,j->bs", output_layer, output_weights) +
            output_bias) / temperature
  return logits


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
      if init_cell_selection_weights_to_zero else _classification_initializer())
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
  column_logits /= cell_count + _EPSILON_ZERO_DIVISION

  # Mask columns that do not appear in the example.
  is_padding = tf.logical_and(cell_count < 0.5,
                              tf.not_equal(out_index.indices, 0))
  column_logits += _CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(is_padding, tf.float32)

  if not allow_empty_column_selection:
    column_logits += _CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(
        tf.equal(out_index.indices, 0), tf.float32)

  return column_logits


def compute_classification_logits(num_classification_labels, output_layer):
  """Computes logits for each classification of the sequence.

  Args:
    num_classification_labels: int Number of class to predict
    output_layer: <float>[batch_size, hidden_dim] Output of the
      encoder layer.

  Returns:
    <float>[batch_size, num_classification_labels] Logits per class.
  """
  hidden_size_agg = output_layer.shape[-1].value
  output_weights_cls = tf.get_variable(
      "output_weights_cls",
      shape=[num_classification_labels, hidden_size_agg],
      initializer=_classification_initializer())
  output_bias_cls = tf.get_variable(
      "output_bias_cls",
      shape=[num_classification_labels],
      initializer=tf.zeros_initializer())
  logits_classification = tf.matmul(
      output_layer, output_weights_cls, transpose_b=True)
  logits_classification = tf.nn.bias_add(logits_classification, output_bias_cls)
  return logits_classification


def _single_column_cell_selection_loss(token_logits, column_logits, label_ids,
                                       cell_index, col_index, cell_mask):
  """Computes the loss for cell selection constrained to a single column.

  The loss is a hierarchical log-likelihood. The model first predicts a column
  and then selects cells within that column (conditioned on the column). Cells
  outside the selected column are never selected.

  Args:
    token_logits: <float>[batch_size, seq_length] Logits per token.
    column_logits: <float>[batch_size, max_num_cols] Logits per column.
    label_ids: <int32>[batch_size, seq_length] Labels per token.
    cell_index: segmented_tensor.IndexMap [batch_size, seq_length] Index that
      groups tokens into cells.
    col_index: segmented_tensor.IndexMap [batch_size, seq_length] Index that
      groups tokens into columns.
    cell_mask: <float>[batch_size, max_num_rows * max_num_cols] Input mask per
      cell, 1 for cells that exists in the example and 0 for padding.

  Returns:
    selection_loss_per_example: <float>[batch_size] Loss for each example.
    logits: <float>[batch_size, seq_length] New logits which are only allowed
      to select cells in a single column. Logits outside of the most likely
      column according to `column_logits` will be set to a very low value
      (such that the probabilities are 0).
  """
  # First find the column we should select. We use the column with maximum
  # number of selected cells.
  labels_per_column, _ = segmented_tensor.reduce_sum(
      tf.cast(label_ids, tf.float32), col_index)
  column_label = tf.argmax(labels_per_column, axis=-1, output_type=tf.int32)
  # Check if there are no selected cells in the column. In that case the model
  # should predict the special column id 0, which means "select nothing".
  no_cell_selected = tf.equal(tf.reduce_max(labels_per_column, axis=-1), 0)
  column_label = tf.where(no_cell_selected, tf.zeros_like(column_label),
                          column_label)

  column_dist = tfp.distributions.Categorical(logits=column_logits)
  column_loss_per_example = -column_dist.log_prob(column_label)

  # Reduce the labels and logits to per-cell from per-token.
  logits_per_cell, _ = segmented_tensor.reduce_mean(token_logits, cell_index)
  labels_per_cell, labels_index = segmented_tensor.reduce_max(
      tf.cast(label_ids, tf.int32), cell_index)

  # Mask for the selected column.
  column_id_for_cells = cell_index.project_inner(labels_index).indices
  column_mask = tf.cast(
      tf.equal(column_id_for_cells, tf.expand_dims(column_label, axis=1)),
      tf.float32)

  # Compute the log-likelihood for cells, but only for the selected column.
  cell_dist = tfp.distributions.Bernoulli(logits=logits_per_cell)
  cell_log_prob = cell_dist.log_prob(labels_per_cell)
  cell_loss = -tf.reduce_sum(cell_log_prob * column_mask * cell_mask, axis=1)
  # We need to normalize the loss by the number of cells in the column.
  cell_loss /= tf.reduce_sum(
      column_mask * cell_mask, axis=1) + _EPSILON_ZERO_DIVISION

  selection_loss_per_example = column_loss_per_example
  selection_loss_per_example += tf.where(
      no_cell_selected, tf.zeros_like(selection_loss_per_example), cell_loss)

  # Set the probs outside the selected column (selected by the *model*)
  # to 0. This ensures backwards compatibility with models that select
  # cells from multiple columns.
  selected_column_id = tf.argmax(column_logits, axis=-1, output_type=tf.int32)
  selected_column_mask = tf.cast(
      tf.equal(column_id_for_cells, tf.expand_dims(selected_column_id,
                                                   axis=-1)), tf.float32)
  # Never select cells with the special column id 0.
  selected_column_mask = tf.where(
      tf.equal(column_id_for_cells, 0), tf.zeros_like(selected_column_mask),
      selected_column_mask)
  logits_per_cell += _CLOSE_ENOUGH_TO_LOG_ZERO * (
      1.0 - cell_mask * selected_column_mask)
  logits = segmented_tensor.gather(logits_per_cell, cell_index)

  return selection_loss_per_example, logits


def _get_classification_outputs(config,
                                is_training, output_layer,
                                output_layer_aggregation, label_ids, input_mask,
                                table_mask, aggregation_function_id, answer,
                                numeric_values, numeric_values_scale, row_ids,
                                column_ids, classification_class_index):
  """Creates a classification model.

  Args:
    config: Configuration for Tapas model.
    is_training: Whether the model is training.
    output_layer: <float32>[batch_size, seq_length, hidden_size]
    output_layer_aggregation: <float32>[batch_size, hidden_size]
    label_ids: <int32>[batch_size, seq_length]
    input_mask: <int32>[batch_size, seq_length]
    table_mask: <int32>[batch_size, seq_length]
    aggregation_function_id: <int32>[batch_size, 1]
    answer: <float32>[batch_size, 1]
    numeric_values: <float32>[batch_size, seq_length]
    numeric_values_scale: <float32>[batch_size, seq_length]
    row_ids: <int32>[batch_size, seq_length]
    column_ids: <int32>[batch_size, seq_length]
    classification_class_index: <int32>[batch, 1]


  Returns:
    total_loss: the overall loss
    logits: <float32>[batch_size, seq_length]
    logits_aggregation: <float32>[batch_size, num_aggregation_labels]
    dist.probs: <float32>[batch_size, seq_length]
    logits_classification <float32>[batch_size, num_classification_labels]
  """
  if is_training:
    # I.e., 0.1 dropout
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

  # Construct indices for the table.
  row_index = segmented_tensor.IndexMap(
      indices=tf.minimum(row_ids, config.max_num_rows - 1),
      num_segments=config.max_num_rows,
      batch_dims=1)
  col_index = segmented_tensor.IndexMap(
      indices=tf.minimum(column_ids, config.max_num_columns - 1),
      num_segments=config.max_num_columns,
      batch_dims=1)
  cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)

  # Masks.
  # <float32>[batch_size, seq_length]
  input_mask_float = tf.cast(input_mask, tf.float32)
  table_mask_float = tf.cast(table_mask, tf.float32)
  # Mask for cells that exist in the table (i.e. that are not padding).
  cell_mask, _ = segmented_tensor.reduce_mean(input_mask_float, cell_index)

  # Compute logits per token. These are used to select individual cells.
  logits = compute_token_logits(
      output_layer=output_layer,
      temperature=config.temperature,
      init_cell_selection_weights_to_zero=\
        config.init_cell_selection_weights_to_zero)

  # Compute logits per column. These are used to select a column.
  if config.select_one_column:
    column_logits = compute_column_logits(
        output_layer=output_layer,
        cell_index=cell_index,
        cell_mask=cell_mask,
        init_cell_selection_weights_to_zero=\
          config.init_cell_selection_weights_to_zero,
        allow_empty_column_selection=config.allow_empty_column_selection)

  # TODO(pawelnow): Extract this into a function.
  # Compute aggregation function logits.
  do_model_aggregation = config.num_aggregation_labels > 0
  if do_model_aggregation:
    hidden_size_agg = output_layer_aggregation.shape[-1].value
    output_weights_agg = tf.get_variable(
        "output_weights_agg",
        shape=[config.num_aggregation_labels, hidden_size_agg],
        initializer=_classification_initializer())
    output_bias_agg = tf.get_variable(
        "output_bias_agg",
        shape=[config.num_aggregation_labels],
        initializer=tf.zeros_initializer())

  do_model_classification = config.num_classification_labels > 0
  logits_cls = None
  if do_model_classification:
    logits_cls = compute_classification_logits(config.num_classification_labels,
                                               output_layer_aggregation)

  with tf.variable_scope("loss"):
    total_loss = 0.0
    is_supervised = not do_model_aggregation or \
        not config.use_answer_as_supervision

    ### Semi-supervised cell selection in case of no aggregation
    #############################################################

    # If the answer (the denotation) appears directly in the table we might
    # select the answer without applying any aggregation function. There are
    # some ambiguous cases, see _calculate_aggregate_mask for more info.
    # `aggregate_mask` is 1 for examples where we chose to aggregate and 0
    #  for examples where we chose to select the answer directly.
    # `label_ids` encodes the positions of the answer appearing in the table.
    if is_supervised:
      aggregate_mask = None
    else:
      # <float32>[batch_size]
      aggregate_mask = _calculate_aggregate_mask(
          answer=answer,
          output_layer_aggregation=output_layer_aggregation,
          output_bias_agg=output_bias_agg,
          output_weights_agg=output_weights_agg,
          cell_select_pref=config.cell_select_pref,
          label_ids=label_ids)

    ### Cell selection log-likelihood
    ###################################

    if config.average_logits_per_cell:
      logits_per_cell, _ = segmented_tensor.reduce_mean(logits, cell_index)
      logits = segmented_tensor.gather(logits_per_cell, cell_index)
    dist_per_token = tfp.distributions.Bernoulli(logits=logits)

    selection_loss_per_example = None
    if not config.select_one_column:
      weight = tf.where(
          label_ids == 0, tf.ones_like(label_ids, dtype=tf.float32),
          config.positive_weight *\
          tf.ones_like(label_ids, dtype=tf.float32))
      selection_loss_per_token = -dist_per_token.log_prob(label_ids) * weight
      selection_loss_per_example = (
          tf.reduce_sum(selection_loss_per_token * input_mask_float, axis=1) /
          (tf.reduce_sum(input_mask_float, axis=1) + _EPSILON_ZERO_DIVISION))
    else:
      selection_loss_per_example, logits = _single_column_cell_selection_loss(
          token_logits=logits,
          column_logits=column_logits,
          label_ids=label_ids,
          cell_index=cell_index,
          col_index=col_index,
          cell_mask=cell_mask)
      dist_per_token = tfp.distributions.Bernoulli(logits=logits)

    ### Logits for the aggregation function
    #########################################

    logits_aggregation = None
    if do_model_aggregation:
      logits_aggregation = _calculate_aggregation_logits(
          output_layer_aggregation, output_weights_agg, output_bias_agg)

    ### Classification loss
    ###############################
    if do_model_classification:
      one_hot_labels = tf.one_hot(
          classification_class_index,
          depth=config.num_classification_labels,
          dtype=tf.float32)
      log_probs = tf.nn.log_softmax(logits_cls, axis=-1)
      # <float32>[batch_size]
      per_example_classification_intermediate = -tf.reduce_sum(
          one_hot_labels * log_probs, axis=-1)

      cls_loss = tf.reduce_mean(per_example_classification_intermediate)
      # with tf.control_dependencies([tf.print(cls_loss)]):
      total_loss += cls_loss

    ### Supervised cell selection
    ###############################

    if config.disable_per_token_loss:
      pass
    elif is_supervised:
      total_loss += tf.reduce_mean(selection_loss_per_example)
    else:
      # For the not supervissed case, do not assign loss for cell selection
      total_loss += tf.reduce_mean(selection_loss_per_example *
                                   (1.0 - aggregate_mask))

    ### Semi-supervised regression loss and supervised loss for aggregations
    #########################################################################

    if do_model_aggregation:
      # Note that `aggregate_mask` is None if the setting is supervised.
      per_example_additional_loss = _calculate_aggregation_loss(
          logits_aggregation, aggregate_mask, aggregation_function_id, config)

      if config.use_answer_as_supervision:
        # Add regression loss for numeric answers which require aggregation.
        answer_loss, large_answer_loss_mask = _calculate_regression_loss(
            answer, aggregate_mask, dist_per_token, numeric_values,
            numeric_values_scale, table_mask_float, logits_aggregation, config)
        per_example_additional_loss += answer_loss
        # Zero loss for examples with answer_loss > cutoff.
        per_example_additional_loss *= large_answer_loss_mask

      total_loss += tf.reduce_mean(per_example_additional_loss)

    return (total_loss, logits, logits_aggregation,
            _get_probs(dist_per_token) * input_mask_float, logits_cls)


def _calculate_eval_metrics_fn(loss, label_ids, logits, input_mask,
                               aggregation_function_id,
                               logits_aggregation,
                               classification_class_index,
                               logits_cls):
  """Calculates metrics for both cells and aggregation functions."""
  logits.shape.assert_has_rank(2)
  label_ids.shape.assert_has_rank(2)

  # <int32>[batch size, seq_length]
  predictions = tf.where(logits >= 0, tf.ones_like(logits, dtype=tf.int32),
                         tf.zeros_like(logits, dtype=tf.int32))
  input_mask_float = tf.cast(input_mask, tf.float32)

  loss = tf.metrics.mean(values=loss)
  accuracy = tf.metrics.accuracy(
      labels=label_ids, predictions=predictions, weights=input_mask_float)

  # <bool>[batch size, seq_length]
  token_correct = tf.logical_or(
      tf.equal(label_ids, predictions),
      tf.logical_not(tf.cast(input_mask, tf.bool)))
  # <bool>[batch size]
  per_sequence_accuracy = tf.reduce_all(token_correct, axis=1)
  sequence_accuracy = tf.metrics.mean(values=per_sequence_accuracy)

  probs = tf.sigmoid(logits)
  precision = tf.metrics.precision(
      labels=label_ids, predictions=predictions, weights=input_mask_float)
  recall = tf.metrics.recall(
      labels=label_ids, predictions=predictions, weights=input_mask_float)
  auc = tf.metrics.auc(labels=label_ids, predictions=probs)
  mean_label = tf.metrics.mean(
      values=tf.cast(label_ids, tf.float32), weights=input_mask_float)

  metrics = {
      "eval_loss": loss,
      "eval_accuracy": accuracy,
      "eval_sequence_accuracy": sequence_accuracy,
      "eval_precision": precision,
      "eval_recall": recall,
      "eval_auc": auc,
      "eval_mean_label": mean_label,
  }

  if logits_cls is not None:
    # <int32>[batch size]
    predictions_cls = tf.argmax(logits_cls, axis=-1, output_type=tf.int32)
    accuracy_cls = tf.metrics.accuracy(
        labels=classification_class_index, predictions=predictions_cls)
    metrics.update({
        "eval_classification_accuracy": accuracy_cls,
    })

  if logits_aggregation is not None:
    # <int32>[batch size]
    predictions_agg = tf.argmax(
        logits_aggregation, axis=-1, output_type=tf.int32)
    accuracy_agg = tf.metrics.accuracy(
        labels=aggregation_function_id, predictions=predictions_agg)
    # <bool>[batch size]
    per_sequence_agg_accuracy = tf.equal(aggregation_function_id,
                                         predictions_agg)
    # Whether cells and aggregation function predictions are both correct.
    per_sequence_joint_accuracy = tf.logical_and(per_sequence_agg_accuracy,
                                                 per_sequence_accuracy)
    joint_accuracy = tf.metrics.mean(values=per_sequence_joint_accuracy)
    metrics.update({
        "eval_aggregation_accuracy": accuracy_agg,
        "eval_joint_accuracy": joint_accuracy,
    })
  return metrics


def model_fn_builder(config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    del labels  # Unused.

    tf.logging.info("*** Features ***")
    for name in sorted(features):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    label_ids = features["label_ids"]
    input_mask = features["input_mask"]
    row_ids = features["row_ids"]
    column_ids = features["column_ids"]
    # Table cells only, without question tokens and table headers.
    table_mask = tf.where(row_ids > 0, tf.ones_like(row_ids),
                          tf.zeros_like(row_ids))
    do_model_aggregation = config.num_aggregation_labels > 0
    aggregation_function_id = (
        tf.squeeze(features["aggregation_function_id"], axis=[1])
        if do_model_aggregation else None)

    do_model_classification = config.num_classification_labels > 0
    classification_class_index = (
        tf.squeeze(features["classification_class_index"], axis=[1])
        if do_model_classification else None)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = table_bert.create_model(
        features=features,
        mode=mode,
        bert_config=config.bert_config,
        disabled_features=config.disabled_features,
        disable_position_embeddings=config.disable_position_embeddings)


    if config.use_answer_as_supervision:
      answer = tf.squeeze(features["answer"], axis=[1])
      numeric_values = features["numeric_values"]
      numeric_values_scale = features["numeric_values_scale"]
    else:
      answer = None
      numeric_values = None
      numeric_values_scale = None

    (total_loss, logits, logits_aggregation, probabilities,
     logits_cls) = _get_classification_outputs(
         config=config,
         output_layer=model.get_sequence_output(),
         output_layer_aggregation=model.get_pooled_output(),
         label_ids=label_ids,
         input_mask=input_mask,
         table_mask=table_mask,
         aggregation_function_id=aggregation_function_id,
         answer=answer,
         numeric_values=numeric_values,
         numeric_values_scale=numeric_values_scale,
         is_training=is_training,
         row_ids=row_ids,
         column_ids=column_ids,
         classification_class_index=classification_class_index)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    init_checkpoint = config.init_checkpoint
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if config.use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss,
          config.learning_rate,
          config.num_train_steps,
          config.num_warmup_steps,
          config.use_tpu,
          gradient_accumulation_steps=params.get("gradient_accumulation_steps",
                                                 1),
          grad_clipping=config.grad_clipping)

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = (_calculate_eval_metrics_fn, [
          total_loss, label_ids, logits, input_mask, aggregation_function_id,
          logits_aggregation, classification_class_index, logits_cls
      ])
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      predictions = {
          "probabilities": probabilities,
          "column_ids": features["column_ids"],
          "row_ids": features["row_ids"],
          "segment_ids": features["segment_ids"],
          "question_id_ints": features["question_id_ints"],
      }
      # TODO Remove once the data has been updated.
      if "question_id" in features:
        # Only available when predicting on GPU.
        predictions["question_id"] = features["question_id"]
      if do_model_aggregation:
        predictions.update({
            "gold_aggr":
                features["aggregation_function_id"],
            "pred_aggr":
                tf.argmax(logits_aggregation, axis=-1, output_type=tf.int32)
        })
      if do_model_classification:
        predictions.update({
            "gold_cls": features["classification_class_index"],
            "pred_cls": tf.argmax(logits_cls, axis=-1, output_type=tf.int32)
        })
        if config.num_classification_labels == 2:
          predictions.update(
              {"logits_cls": logits_cls[:, 1] - logits_cls[:, 0]})
        else:
          predictions.update({"logits_cls": logits_cls})
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def input_fn(
    name,
    file_patterns,
    data_format,
    compression_type,
    is_training,
    max_seq_length,
    max_predictions_per_seq,
    add_aggregation_function_id,
    add_classification_labels,
    add_answer,
    include_id,
    params,
):
  """Returns an input_fn compatible with the tf.estimator API."""
  parse_example_fn = table_dataset.parse_table_examples(
      max_seq_length=max_seq_length,
      max_predictions_per_seq=max_predictions_per_seq,
      task_type=table_dataset.TableTask.CLASSIFICATION,
      add_aggregation_function_id=add_aggregation_function_id,
      add_classification_labels=add_classification_labels,
      add_answer=add_answer,
      include_id=include_id,
      add_candidate_answers=False,
      max_num_candidates=0,
      params=params)
  ds = dataset.read_dataset(
      parse_example_fn,
      name=name,
      file_patterns=file_patterns,
      data_format=data_format,
      compression_type=compression_type,
      is_training=is_training,
      params=params)
  return ds
