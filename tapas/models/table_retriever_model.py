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
"""Table retireval model based on TAPAS instances."""

import collections
import json
import re
from typing import Iterable, Text, Optional, List

from absl import logging

import dataclasses

from tapas.datasets import dataset
from tapas.datasets import table_dataset
from tapas.models.bert import modeling
from tapas.models.bert import optimization
from tapas.models.bert import table_bert
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

# Used to mask the logits of the repeated elements
_INF = 10000.0


@dataclasses.dataclass
class RetrieverConfig:
  """Helper class for configuration of Retriever model.

  bert_config: Config object for general bert hyper params.
  init_checkpoint: Location of the model checkpoint.
  learning_rate: Optimizer learning rate.
  num_train_steps: Total number of training steps for optimizer schedule.
  num_warmup_steps: Number of training steps to warm up optimizer.
  use_tpu: Use TPU for training.
  grad_clipping: If not None, clip the gradient norm to this value.
  down_projection_dim: If not None, project query/table representation.
  init_from_single_encoder: If true, a single encoder weights are duplicated.
  max_query_length: The query is capped to this length.
  use_out_of_core_negatives: Use all the negatives when using many TPU cores.
  mask_repeated_tables: Mask tables that are repeated within a batch
  mask_repeated_questions: Mask texts that are repeated within a batch
  ignore_table_content: Only use title and header.
  disable_features: Ignored input features.
  use_mined_negatives: If true, use mined negatives given as add'l features.
  """
  bert_config: modeling.BertConfig
  init_checkpoint: Text
  learning_rate: float
  num_train_steps: Optional[int]
  num_warmup_steps: Optional[int]
  use_tpu: bool
  grad_clipping: Optional[float]
  down_projection_dim: int
  init_from_single_encoder: bool
  max_query_length: int
  use_out_of_core_negatives: bool = False
  mask_repeated_tables: bool = False
  mask_repeated_questions: bool = False
  ignore_table_content: bool = False
  disabled_features: List[Text] = dataclasses.field(default_factory=list)
  use_mined_negatives: bool = False

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
  def from_dict(cls, json_object):
    """Constructs a config from a Python dictionary of parameters."""
    json_object = dict(json_object)
    # Overwrite json bert config with config object.
    json_object["bert_config"] = modeling.BertConfig.from_dict(
        json_object["bert_config"])
    # Delete deprecated option, if present.
    # TODO See of we can filter everything that's not an argument.
    if "restrict_attention" in json_object:
      del json_object["restrict_attention"]
    return RetrieverConfig(**json_object)

  @classmethod
  def from_json_file(cls, json_file: Text):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))


def _calculate_eval_metrics_fn(loss, logits, labels):
  """Calculates metrics for both cells and aggregation functions."""

  loss = tf.metrics.mean(values=loss)
  accuracy = tf.metrics.accuracy(
      labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1))

  metrics = {
      "eval_loss": loss,
      "eval_accuracy": accuracy,
  }
  return metrics


def _apply_reapated_text_masking(
    config: RetrieverConfig,
    question_hash: tf.Tensor,
    question_hash_transposed: tf.Tensor,
    labels: tf.Tensor,
    logits: tf.Tensor,
) -> tf.Tensor:
  """Applies repated text masking.

  Args:
    config: Retriever config.
    question_hash: <int64>[global_batch_size, 1]
    question_hash_transposed: <int64>[1, batch_size]
    labels: <int64>[batch_size, global_batch_size * num_tables]
    logits: <float>[batch_size, global_batch_size * num_tables]

  Returns:
    Masked logits (same shape / dtype).
  """
  # Make sure not all hashes are 0.
  # This indicates the "question_hash" feature wasn't set.
  assert_op = tf.assert_equal(
      tf.math.reduce_all(tf.math.equal(question_hash, 0)), [False])
  with tf.control_dependencies([assert_op]):
    logging.vlog(2, "question_hash: %s", question_hash)
    logging.vlog(2, "question_hash_transposed: %s", question_hash_transposed)
    logging.vlog(2, "labels: %s", labels)
    logging.vlog(2, "logits: %s", logits)
    # <bool>[batch_size, global_batch_size]
    repeated_texts = tf.math.equal(question_hash, question_hash_transposed)
    if config.use_mined_negatives:
      batch_size = repeated_texts.shape[0]
      global_batch_size = repeated_texts.shape[1]
      num_tables = logits.shape[1] // global_batch_size
      # <bool>[batch_size, global_batch_size * num_tables]
      repeated_texts = tf.concat([
          repeated_texts,
          tf.zeros(
              shape=(batch_size, (num_tables - 1) * global_batch_size),
              dtype=tf.bool)
      ],
                                 axis=1)
    repeated_texts = (
        repeated_texts
        # Makes sure original correct question pair isn't masked
        & tf.math.equal(labels, 0))
    logging.vlog(2, "repeated texts: %s", repeated_texts)
  ops = []
  if logging.vlog_is_on(2):
    ops.append(
        tf.print(
            "repeated texts content:",
            question_hash,
            repeated_texts,
            output_stream=logging.info,
        ))
  with tf.control_dependencies(ops):
    return tf.where(repeated_texts, tf.zeros_like(logits) - _INF, logits)


def _get_assignment_map_from_checkpoint(tvars, init_checkpoint,
                                        init_from_single_encoder):
  """Computes the assignment of checkpoint variables to graph variables.

  Args:
    tvars: Computation graph variables.
    init_checkpoint: Checkpoint variables.
    init_from_single_encoder: Whether to initialize an additional encoder from
      checkpoint variables.

  Returns:
    assignments: A list of mappings, where each mapping holds assignments from
    checkpoint variable to computation graph variable.
  """

  def _fill_assignments(assignment_map, initialized_variable_names,
                        var_name_from_init, var_name):
    """Assigns a checkpoint variable to init some graph variable."""
    if var_name not in name_to_variable:
      tf.logging.info(
          "%s is not part of the computation graph, thus cannot be initialized",
          var_name)
      return
    assignment_map[var_name_from_init] = var_name
    initialized_variable_names[var_name] = 1
    initialized_variable_names[var_name + ":0"] = 1

  # Collect all computation graph variables.
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  # Assign checkpoint variables that match graph variables.
  init_vars = tf.train.list_variables(init_checkpoint)
  assignment_map = collections.OrderedDict()
  initialized_variable_names = {}
  for x in init_vars:
    (name, var) = (x[0], x[1])
    _fill_assignments(assignment_map, initialized_variable_names, name, name)
  assignments = [assignment_map]

  # Load variables from init checkpoint to an additional encoder.
  if init_from_single_encoder:
    assignment_map_additional_encoder = collections.OrderedDict()
    for x in init_vars:
      (name, var) = (x[0], x[1])
      # Replace "varname/..." with "varname_1/... to get the name of the second
      # instance of the variable.
      name_additional = re.sub(r"(.*?)/", r"\1_1/", name, count=1)
      _fill_assignments(assignment_map_additional_encoder,
                        initialized_variable_names, name, name_additional)
    assignments.append(assignment_map_additional_encoder)

  return (assignments, initialized_variable_names)


@dataclasses.dataclass
class ModelBuilderData:
  # <float32>[global_batch_size * num_tables, down_projection_dim]
  table_rep: tf.Tensor
  # <int64>[global_batch_size * num_tables]
  table_id_hash: tf.Tensor
  # <int64>[global_batch_size, 1]
  question_hash: tf.Tensor
  # <float32>[batch_size, global_batch_size * num_tables]
  labels: tf.Tensor


def get_updates_for_use_tpu_with_out_of_core_negatives(
    data: ModelBuilderData) -> ModelBuilderData:
  """Recovers data in case use_tpu and out_of_core_negatives are used."""
  raise NotImplementedError


def model_fn_builder(config: RetrieverConfig):
  """Returns `model_fn` closure for TPUEstimator."""

  def _get_projection_matrix(name, num_columns):
    return tf.get_variable(
        name,
        shape=[config.down_projection_dim, num_columns],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

  def _get_type_representation(hidden_representation, prejection):
    return tf.matmul(hidden_representation, prejection, transpose_b=True)

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    del labels, params  # Unused.

    # TODO(thomasmueller) Add support for this.
    if (config.use_out_of_core_negatives and config.use_mined_negatives and
        config.mask_repeated_questions):
      raise ValueError("Unsupported combination of options.")

    tf.logging.info("*** Features ***")
    for name in sorted(features):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    if config.ignore_table_content:
      features["input_mask"] = tf.where(features["row_ids"] > 0,
                                        tf.zeros_like(features["input_mask"]),
                                        tf.ones_like(features["input_mask"]))

    features_table_names = [
        "input_ids", "input_mask", "segment_ids", "column_ids", "row_ids",
        "prev_label_ids", "column_ranks", "inv_column_ranks",
        "numeric_relations"
    ]
    features_table = {}
    for name in features_table_names:
      if config.use_mined_negatives:
        # split each feature in half, and concat vertically.
        feature_positive_table, feature_negative_table = tf.split(
            features[name], num_or_size_splits=2, axis=1)
        features_table[name] = tf.concat(
            [feature_positive_table, feature_negative_table], axis=0)
      else:
        features_table[name] = features[name]

    tf.logging.info("*** Features table ***")
    for name in sorted(features_table):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    table_model = table_bert.create_model(
        features=features_table,
        disabled_features=config.disabled_features,
        mode=mode,
        bert_config=config.bert_config,
    )

    # Arrange features for query, such that it is assigned with an empty table.
    empty_table_features = tf.zeros_like(
        features["question_input_mask"])[:, :config.max_query_length]
    features_query = {
        "input_ids":
            features["question_input_ids"][:, :config.max_query_length],
        "input_mask":
            features["question_input_mask"][:, :config.max_query_length],
        "segment_ids":
            empty_table_features,
        "column_ids":
            empty_table_features,
        "row_ids":
            empty_table_features,
        "prev_label_ids":
            empty_table_features,
        "column_ranks":
            empty_table_features,
        "inv_column_ranks":
            empty_table_features,
        "numeric_relations":
            empty_table_features,
    }
    query_model = table_bert.create_model(
        features=features_query,
        disabled_features=config.disabled_features,
        mode=mode,
        bert_config=config.bert_config,
    )

    table_hidden_representation = table_model.get_pooled_output()
    query_hidden_representation = query_model.get_pooled_output()
    if config.down_projection_dim > 0:
      table_projection = _get_projection_matrix(
          "table_projection", num_columns=table_hidden_representation.shape[1])
      query_projection = _get_projection_matrix(
          "text_projection", num_columns=query_hidden_representation.shape[1])

      # <float32>[batch_size * num_tables, down_projection_dim]
      table_rep = _get_type_representation(table_hidden_representation,
                                           table_projection)
      # <float32>[batch_size, down_projection_dim]
      query_rep = _get_type_representation(query_hidden_representation,
                                           query_projection)
    else:
      table_rep = table_hidden_representation
      query_rep = query_hidden_representation

    batch_size = tf.shape(query_rep)[0]
    # Identity matrix, as gold logits are on the diagonal.
    labels_single_table = tf.eye(batch_size)
    if config.use_mined_negatives:
      # <int64>[batch_size, batch_size * num_tables]
      labels = tf.concat(
          [labels_single_table,
           tf.zeros_like(labels_single_table)], axis=1)
    else:
      labels = labels_single_table

    # <int64>[batch_size * num_tables]
    table_id_hash = tf.reshape(
        tf.transpose(features["table_id_hash"]), shape=[-1])
    # <int64>[batch_size, 1]
    table_id_hash_transposed = features["table_id_hash"][..., :1]

    # <int64>[batch_size, 1]
    question_hash = features["question_hash"]

    # <int64>[1, batch_size]
    question_hash_transposed = tf.transpose(question_hash)
    if config.use_tpu and config.use_out_of_core_negatives:
      data = get_updates_for_use_tpu_with_out_of_core_negatives(
          ModelBuilderData(table_rep, table_id_hash, question_hash, labels))
      table_rep = data.table_rep
      table_id_hash = data.table_id_hash
      question_hash = data.question_hash
      labels = data.labels
    # <float32>[batch_size, batch_size|global_batch_size * num_tables]
    logits = tf.matmul(query_rep, table_rep, transpose_b=True)
    if config.mask_repeated_tables:
      # Matrix of non-trivial repeated tables
      # <bool>[batch_size, batch_size|global_batch_size * num_tables]
      repeated_tables = tf.math.equal(
          tf.expand_dims(table_id_hash, axis=0),
          table_id_hash_transposed) & tf.math.equal(labels, 0)
      logits = tf.where(repeated_tables, tf.zeros_like(logits) - _INF, logits)

    if config.mask_repeated_questions:
      logits = _apply_reapated_text_masking(
          config,
          question_hash,
          question_hash_transposed,
          labels,
          logits,
      )

    total_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    init_checkpoint = config.init_checkpoint
    if init_checkpoint:
      (assignment_maps,
       initialized_variable_names) = _get_assignment_map_from_checkpoint(
           tvars, init_checkpoint, config.init_from_single_encoder)

      if config.use_tpu:

        def tpu_scaffold():
          for assignment_map in assignment_maps:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        for assignment_map in assignment_maps:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss,
          config.learning_rate,
          config.num_train_steps,
          config.num_warmup_steps,
          config.use_tpu,
          grad_clipping=config.grad_clipping)

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf_estimator.ModeKeys.EVAL:
      eval_metrics = (_calculate_eval_metrics_fn, [total_loss, logits, labels])
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      if config.use_mined_negatives:
        table_rep_gold, _ = tf.split(table_rep, num_or_size_splits=2, axis=0)
      else:
        table_rep_gold = table_rep

      predictions = {
          "query_rep": query_rep,
          "table_rep": table_rep_gold,
      }
      # Only available when predicting on GPU.
      if "table_id" in features:
        predictions["table_id"] = features["table_id"]
      if "question_id" in features:
        predictions["query_id"] = features["question_id"]
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def input_fn(name: Text, file_patterns: Iterable[Text], data_format: Text,
             is_training: bool, max_seq_length: int, include_id: bool,
             compression_type: Text, use_mined_negatives: bool, params):
  """Returns an input_fn compatible with the tf.estimator API."""
  task_type = (
      table_dataset.TableTask.RETRIEVAL_NEGATIVES
      if use_mined_negatives else table_dataset.TableTask.RETRIEVAL)
  parse_example_fn = table_dataset.parse_table_examples(
      max_seq_length=max_seq_length,
      max_predictions_per_seq=None,
      task_type=task_type,
      add_aggregation_function_id=False,
      add_classification_labels=False,
      add_answer=False,
      include_id=include_id,
      add_candidate_answers=False,
      max_num_candidates=0,
      params=params)
  ds = dataset.read_dataset(
      parse_example_fn,
      name=name,
      file_patterns=file_patterns,
      data_format=data_format,
      is_training=is_training,
      compression_type=compression_type,
      params=dict(params, max_eval_count=None),
  )
  return ds
