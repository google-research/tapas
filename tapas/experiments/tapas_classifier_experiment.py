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
"""TAPAS classifier experiment."""

import functools
import json
import os
import traceback
from typing import Mapping, Optional, Text

from absl import app
from absl import flags
import numpy as np
from tapas.experiments import prediction_utils
from tapas.models import tapas_classifier_model
from tapas.scripts import calc_metrics_utils
from tapas.scripts import prediction_utils as pred_utils
from tapas.utils import attention_utils
from tapas.utils import experiment_utils  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
FLAGS = flags.FLAGS

flags.DEFINE_string("data_format", "tfrecord", "The input data format.")

flags.DEFINE_string(
    "compression_type",
    "GZIP",
    "Compression to use when reading tfrecords. '' for no compression.",
)

flags.DEFINE_multi_string(
    "input_file_train", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_multi_string(
    "input_file_eval", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_multi_string(
    "input_file_predict", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "prediction_output_dir", None,
    "If not none or empty writes predictions to this directory. Otherwise "
    "writes predictions to model_dir.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")


flags.DEFINE_integer("max_num_columns", 32,
                     "Maximum number of columns in a table.")

flags.DEFINE_integer("max_num_rows", 64, "Maximum number of rows in a table.")

flags.DEFINE_bool("average_logits_per_cell", False,
                  "Whether to average logits in the model.")

flags.DEFINE_integer("minutes_to_sleep_before_predictions", 5,
                     "Time in minutes to sleep before starting to predict.")

flags.DEFINE_float("positive_weight", 10.0, "Weight for positive labels.")

flags.DEFINE_float("cell_classification_threshold", 0.5,
                   "Threshold for selecting cells when predicting.")

flags.DEFINE_bool("output_token_probabilities", False,
                  "Whether to add token probabilities to predictions.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string(
    "eval_name", None,
    "Name of the current evaluation set. Will be used in Tensorboard.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer(
    "evaluated_checkpoint_step", None,
    "The step for a specific model checkpoint to be evaluated. If None, then"
    "all checkpoints are used.")

flags.DEFINE_bool(
    "do_sequence_prediction", False,
    "Whether to run sequence prediction. (Relevant for SQA-like datasets.)")

flags.DEFINE_integer(
    "num_aggregation_labels", 0,
    "The number of aggregation classes to predict. If 0, then"
    "aggregation function prediction is not modeled.")

flags.DEFINE_integer(
    "num_classification_labels", 0,
    "The number of classes to predict. If 0, then"
    "classification prediction is not modeled.")

flags.DEFINE_enum(
    "span_prediction",
    "none",
    ["none", "boundary", "span"],
    "Add span prediction output.",
)

flags.DEFINE_float(
    "aggregation_loss_importance", 1.0, "Weight that determines the aggregation"
    "loss importance in comparison to that of the cell loss. Only used if"
    "num_aggregation_labels > 0")

flags.DEFINE_bool(
    "use_answer_as_supervision", True, "Whether to consider the answer as"
    "supervision for aggregation examples, which corresponds to the weakly"
    "supervised setting. Only used if num_aggregation_labels > 0")

flags.DEFINE_float(
    "answer_loss_importance", 1.0, "Weight that determines the answer"
    "loss importance in comparison to that of the cell loss. Only used if"
    "use_answer_as_supervision is True.")

flags.DEFINE_bool(
    "use_normalized_answer_loss", False,
    "If true normalize loss by the max of predicted and expected value.")

flags.DEFINE_float(
    "huber_loss_delta", 25.0, "The delta used to calculate the huber loss."
    "Only used if use_answer_as_supervision is True.")

flags.DEFINE_float(
    "answer_loss_cutoff", 100.0, "If set, model weights will not be"
    "updated for examples with a scalar answer loss larger than this cutoff."
    "Only used if use_answer_as_supervision is True.")

flags.DEFINE_float(
    "cell_select_pref", 0.5, "A value in [0,1] that describes the"
    "behavior in ambiguous cases where the answer is a single float that"
    "appears in the table. A value of 1 means that the model would always"
    "prefer cell selection in this case. Only used if use_answer_as_supervision"
    "is True.")

flags.DEFINE_float(
    "temperature", 1.0, "Used to scale cell logits. A lower temperature"
    "produces probabilities that looks like stepwise functions. A value of 1.0"
    "means no scaling.")

flags.DEFINE_float(
    "agg_temperature", 1.0, "Used to scale agg logits. A lower temperature"
    "produces probabilities that looks like stepwise functions. A value of 1.0"
    "means no scaling.")

flags.DEFINE_bool(
    "use_gumbel_for_cells", False, "Whether gumbel-softmax distribution should "
    "be used for cell selection.")

flags.DEFINE_bool(
    "use_gumbel_for_agg", False, "Whether gumbel-softmax distribution should "
    "be used for aggregation function selection.")

flags.DEFINE_enum(
    "average_approximation_function", "ratio",
    ["ratio", "first_order", "second_order"],
    "To calculate the expected value of the average of the cells, one can use "
    "different approximations, all of them biased")

flags.DEFINE_float(
    "grad_clipping", None, "If not None, gradients greater in absolute value"
    "than this number are clipped.")

flags.DEFINE_string("eval_interactions_file", None,
                    "The file that contains interactions protos.")
flags.DEFINE_string("predict_interactions_file", None,
                    "The file that contains interactions protos.")

flags.DEFINE_bool(
    "init_cell_selection_weights_to_zero", False,
    "Whether to initialize cell selection weights to 0 "
    "so that the initial probabilities are 50%.")

flags.DEFINE_bool(
    "select_one_column", False,
    "Whether the model should be forced to select cells from only one column.")

flags.DEFINE_string("table_pruning_config_file", None,
                    "To specify the config for table pruning.")

flags.DEFINE_enum(
    "restrict_attention_mode",
    "full",
    [
        "full",
        "same_colum_or_row",
        "headwise_same_colum_or_row",
        "headwise_efficient",
        "table_attention"
    ],
    "Options to restrict attention if tokens are in the same row/column.")

flags.DEFINE_integer(
    "restrict_attention_bucket_size", 0, "For sparse attention modes, further "
    "restrict attention to consecutive buckets of uniform size.")

flags.DEFINE_integer(
    "restrict_attention_header_size", None, "For sparse attention modes, size "
    "of the first section that will attend to/from everything else.")

flags.DEFINE_float(
    "restrict_attention_row_heads_ratio", 0.5, "For sparse attention modes, "
    "proportion of heads that should focus on rows vs columns.")

flags.DEFINE_bool("cell_cross_entropy", False,
                  "Whether to use cross entropy for cell selection loss.")

flags.DEFINE_bool(
    "cell_cross_entropy_hard_em", False,
    "When using cell cross entropy, whether to use hard or soft "
    "EM for cases with more than one ground truth cell.")


flags.DEFINE_string(
    "classification_label_weight", "",
    "CSV List of label:weight to multiply the loss term for each example.")

flags.DEFINE_bool(
    "reset_output_cls",
    False,
    "If true, reset classification output layer on init from checkpoint. This "
    "is useful when fine-tuning a classifier model from a model pre-trained "
    "with a different number of classes.",
)

flags.DEFINE_bool(
    "compute_weighted_denotation_accuracy", False,
    "Whether to compute weighted denotation accuracy for QA experiments.")

flags.DEFINE_list("disabled_features", [],
                  "Features that should be disabled (for ablation studies).")

flags.DEFINE_bool("disable_position_embeddings", False,
                  "Whether to disable position embeddings.")

flags.DEFINE_bool("reset_position_index_per_cell", False,
                  "Whether to restart position indexes at every cell.")

flags.DEFINE_bool("disable_per_token_loss", False,
                  "Disable any (strong or weak) supervision on cells.")

flags.DEFINE_bool("allow_empty_column_selection", False,
                  "If false, disallow selecting no column.")

flags.DEFINE_bool("mask_examples_without_labels", False,
                  "If false, mask examples without answers.")

flags.DEFINE_integer(
    "proj_value_length",
    -1,
    "If > 0, down-project key and values in self attention computation.",
)

flags.DEFINE_bool(
    "attention_bias_use_relative_scalar_only", True,
    "Whether to use only relative scalar (no embedding) for attention bias in"
    "TableFormer. Takes effect only if `restrict_attention_mode` is"
    "\"table_attention\".")

flags.DEFINE_integer(
    "attention_bias_disabled", 0,
    "Attention bias that should be disabled (for ablation studies) in"
    "TableFormer. Each number maps to a specific relationship id. Mapping can"
    "be found in tapas/utils/tableformer_utils.py. Takes effect only if"
    "`restrict_attention_mode` is \"table_attention\".")



def _get_projection_length(proj_length):
  return proj_length if proj_length and proj_length > 0 else None


def _predict_and_export_metrics(
    mode,
    name,
    input_fn,
    estimator,
    current_step,
    checkpoint,
    output_dir,
    do_model_aggregation,
    do_model_classification,
    output_token_answers,
):
  """Exports model predictions and calculates denotation metric.

  Args:
    mode: Prediction mode. Can be "predict" or "eval".
    name: Used as name appendix if not default.
    input_fn: Function to generate exmaples to passed into `estimator.predict`.
    estimator: The Estimator instance.
    current_step: Current checkpoint step to be evaluated.
    checkpoint: Path to the checkpoint to be evaluated.
    output_dir: Path to save predictions generated by the checkpoint.
    do_model_aggregation: Whether model does aggregation.
    do_model_classification: Whether model does classification.
    output_token_answers: If true, output answer coordinates.

  Raises:
    ValueError: if an invalid mode is passed.
  """
  # Predict for each new checkpoint.
  tf.logging.info(
      "Running predictor for step %d (%s).",
      current_step,
      checkpoint,
  )
  if mode == "predict":
    input_file = FLAGS.input_file_predict
    interactions_file = FLAGS.predict_interactions_file
  elif mode == "eval":
    input_file = FLAGS.input_file_eval
    interactions_file = FLAGS.eval_interactions_file
  else:
    raise ValueError(f"Invalid mode {mode}")
  result = estimator.predict(
      input_fn=input_fn,
      checkpoint_path=checkpoint,
  )

  base_name = mode
  if name:
    base_name = f"{mode}_{name}"

  output_predict_file = os.path.join(output_dir,
                                     f"{base_name}_results_{current_step}.tsv")
  prediction_utils.write_predictions(
      result,
      output_predict_file,
      do_model_aggregation,
      do_model_classification,
      FLAGS.cell_classification_threshold,
      FLAGS.output_token_probabilities,
      output_token_answers=output_token_answers,
  )

  if FLAGS.do_sequence_prediction:
    examples_by_position = prediction_utils.read_classifier_dataset(
        predict_data=input_file,
        data_format=FLAGS.data_format,
        compression_type=FLAGS.compression_type,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=FLAGS.use_answer_as_supervision)
    result_sequence = prediction_utils.compute_prediction_sequence(
        estimator=estimator, examples_by_position=examples_by_position)
    output_predict_file_sequence = os.path.join(
        FLAGS.model_dir,
        base_name + "_results_sequence_{}.tsv").format(current_step)
    prediction_utils.write_predictions(
        result_sequence, output_predict_file_sequence, do_model_aggregation,
        do_model_classification, FLAGS.cell_classification_threshold,
        FLAGS.output_token_probabilities, output_token_answers)



def main(_):
  do_model_aggregation = FLAGS.num_aggregation_labels > 0
  do_model_classification = FLAGS.num_classification_labels > 0
  bert_config = experiment_utils.bert_config_from_flags()
  total_steps = experiment_utils.num_train_steps()
  tapas_config = tapas_classifier_model.TapasClassifierConfig(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=total_steps,
      num_warmup_steps=experiment_utils.num_warmup_steps(),
      use_tpu=FLAGS.use_tpu,
      positive_weight=FLAGS.positive_weight,
      num_aggregation_labels=FLAGS.num_aggregation_labels,
      num_classification_labels=FLAGS.num_classification_labels,
      aggregation_loss_importance=FLAGS.aggregation_loss_importance,
      use_answer_as_supervision=FLAGS.use_answer_as_supervision,
      answer_loss_importance=FLAGS.answer_loss_importance,
      use_normalized_answer_loss=FLAGS.use_normalized_answer_loss,
      huber_loss_delta=FLAGS.huber_loss_delta,
      temperature=FLAGS.temperature,
      agg_temperature=FLAGS.agg_temperature,
      use_gumbel_for_cells=FLAGS.use_gumbel_for_cells,
      use_gumbel_for_agg=FLAGS.use_gumbel_for_agg,
      average_approximation_function=(
          tapas_classifier_model.AverageApproximationFunction(
              FLAGS.average_approximation_function)),
      cell_select_pref=FLAGS.cell_select_pref,
      answer_loss_cutoff=FLAGS.answer_loss_cutoff,
      grad_clipping=FLAGS.grad_clipping,
      classification_label_weight={
          int(pair.split(":")[0]): float(pair.split(":")[1])
          for pair in FLAGS.classification_label_weight.split(",")
          if pair
      },
      table_pruning_config_file=FLAGS.table_pruning_config_file,
      restrict_attention_mode=(attention_utils.RestrictAttentionMode(
          FLAGS.restrict_attention_mode)),
      restrict_attention_bucket_size=FLAGS.restrict_attention_bucket_size,
      restrict_attention_header_size=FLAGS.restrict_attention_header_size,
      restrict_attention_row_heads_ratio=(
          FLAGS.restrict_attention_row_heads_ratio),
      mask_examples_without_labels=FLAGS.mask_examples_without_labels,
      cell_cross_entropy_hard_em=FLAGS.cell_cross_entropy_hard_em,
      cell_cross_entropy=FLAGS.cell_cross_entropy,
      reset_output_cls=FLAGS.reset_output_cls,
      disabled_features=FLAGS.disabled_features,
      max_num_rows=FLAGS.max_num_rows,
      max_num_columns=FLAGS.max_num_columns,
      average_logits_per_cell=FLAGS.average_logits_per_cell,
      init_cell_selection_weights_to_zero=FLAGS
      .init_cell_selection_weights_to_zero,
      select_one_column=FLAGS.select_one_column,
      allow_empty_column_selection=FLAGS.allow_empty_column_selection,
      disable_position_embeddings=FLAGS.disable_position_embeddings,
      disable_per_token_loss=FLAGS.disable_per_token_loss,
      reset_position_index_per_cell=FLAGS.reset_position_index_per_cell,
      span_prediction=tapas_classifier_model.SpanPredictionMode(
          FLAGS.span_prediction),
      proj_value_length=_get_projection_length(FLAGS.proj_value_length),
      attention_bias_disabled=FLAGS.attention_bias_disabled,
      attention_bias_use_relative_scalar_only=FLAGS
      .attention_bias_use_relative_scalar_only,
  )

  model_fn = tapas_classifier_model.model_fn_builder(tapas_config)
  estimator = experiment_utils.build_estimator(model_fn)

  if tapas_config.classification_label_weight:
    if any(x < 0 for x in tapas_config.classification_label_weight.values()):
      raise ValueError("Label weights cannot be negative in input: "
                       f"{tapas_config.classification_label_weight}.")
    if any(x < 0 or x >= tapas_config.num_classification_labels
           for x in tapas_config.classification_label_weight.keys()):
      raise ValueError("Invalid label in label weights for input: "
                       f"{tapas_config.classification_label_weight}.")

  if FLAGS.do_train:
    tf.io.gfile.makedirs(FLAGS.model_dir)
    # Copy the table pruning config if pruning is used.
    if FLAGS.table_pruning_config_file:
      table_pruning_path = os.path.join(FLAGS.model_dir,
                                        "table_pruning_config.textproto")
      tf.io.gfile.copy(
          FLAGS.table_pruning_config_file, table_pruning_path, overwrite=True)
    bert_config.to_json_file(os.path.join(FLAGS.model_dir, "bert_config.json"))
    tapas_config.to_json_file(
        os.path.join(FLAGS.model_dir, "tapas_config.json"))
    train_input_fn = functools.partial(
        tapas_classifier_model.input_fn,
        name="train",
        file_patterns=FLAGS.input_file_train,
        data_format=FLAGS.data_format,
        compression_type=FLAGS.compression_type,
        is_training=True,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=FLAGS.use_answer_as_supervision,
        include_id=False)
    estimator.train(input_fn=train_input_fn, max_steps=total_steps)

  eval_input_fn = functools.partial(
      tapas_classifier_model.input_fn,
      name="eval",
      file_patterns=FLAGS.input_file_eval,
      data_format=FLAGS.data_format,
      compression_type=FLAGS.compression_type,
      is_training=False,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      add_aggregation_function_id=do_model_aggregation,
      add_classification_labels=do_model_classification,
      add_answer=FLAGS.use_answer_as_supervision,
      include_id=not FLAGS.use_tpu)
  if FLAGS.do_eval:
    eval_name = FLAGS.eval_name if FLAGS.eval_name is not None else "default"
    for _, checkpoint in experiment_utils.iterate_checkpoints(
        model_dir=estimator.model_dir,
        total_steps=total_steps,
        marker_file_prefix=os.path.join(estimator.model_dir,
                                        f"eval_{eval_name}"),
        minutes_to_sleep=FLAGS.minutes_to_sleep_before_predictions):
      tf.logging.info("Running eval: %s", eval_name)
      result = estimator.evaluate(
          input_fn=eval_input_fn,
          steps=FLAGS.num_eval_steps,
          name=eval_name,
          checkpoint_path=checkpoint)
      tf.logging.info("Eval result:\n%s", result)

  if FLAGS.do_predict:
    predict_input_fn = functools.partial(
        tapas_classifier_model.input_fn,
        name="predict",
        file_patterns=FLAGS.input_file_predict,
        data_format=FLAGS.data_format,
        compression_type=FLAGS.compression_type,
        is_training=False,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=FLAGS.use_answer_as_supervision,
        include_id=not FLAGS.use_tpu)

    if FLAGS.prediction_output_dir:
      prediction_output_dir = FLAGS.prediction_output_dir
      tf.io.gfile.makedirs(prediction_output_dir)
    else:
      prediction_output_dir = estimator.model_dir

    marker_file_prefix = os.path.join(prediction_output_dir, "predict")
    # When two separate jobs are launched we don't want conflicting markers.
    if FLAGS.input_file_predict is not None:
      marker_file_prefix += "_test"
    if FLAGS.input_file_eval is not None:
      marker_file_prefix += "_dev"

    for current_step, checkpoint in experiment_utils.iterate_checkpoints(
        model_dir=estimator.model_dir,
        total_steps=total_steps,
        single_step=FLAGS.evaluated_checkpoint_step,
        marker_file_prefix=marker_file_prefix):

      try:
        if FLAGS.input_file_predict is not None:
          _predict_and_export_metrics(
              mode="predict",
              name=FLAGS.eval_name,
              input_fn=predict_input_fn,
              output_dir=prediction_output_dir,
              estimator=estimator,
              checkpoint=checkpoint,
              current_step=current_step,
              do_model_classification=do_model_classification,
              do_model_aggregation=do_model_aggregation,
              output_token_answers=not FLAGS.disable_per_token_loss)
        if FLAGS.input_file_eval is not None:
          _predict_and_export_metrics(
              mode="eval",
              name=FLAGS.eval_name,
              input_fn=eval_input_fn,
              output_dir=prediction_output_dir,
              estimator=estimator,
              checkpoint=checkpoint,
              current_step=current_step,
              do_model_classification=do_model_classification,
              do_model_aggregation=do_model_aggregation,
              output_token_answers=not FLAGS.disable_per_token_loss)
      except ValueError:
        tf.logging.error("Error getting predictions for checkpoint %s: %s",
                         checkpoint, traceback.format_exc())


if __name__ == "__main__":
  app.run(main)
