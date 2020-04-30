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
"""TAPAS classifier experiment."""

import functools
import os
import time

from absl import app
from absl import flags
from tapas.experiments import prediction_utils
from tapas.models import tapas_classifier_model
from tapas.scripts import calc_metrics_utils
from tapas.utils import experiment_utils  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf

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

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string(
    "eval_name", "default",
    "Name of the current evaluation set. Will be used in Tensorboard.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

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

flags.DEFINE_list("disabled_features", [],
                  "Features that should be disabled (for ablation studies).")
flags.DEFINE_bool("disable_position_embeddings", False,
                  "Whether to disable position embeddings.")

flags.DEFINE_bool("disable_per_token_loss", False,
                  "Disable any (strong or weak) supervision on cells.")

flags.DEFINE_bool("allow_empty_column_selection", False,
                  "If false, disallow selecting no column.")


def main(_):
  do_model_aggregation = FLAGS.num_aggregation_labels > 0
  do_model_classification = FLAGS.num_classification_labels > 0
  bert_config = experiment_utils.bert_config_from_flags()
  tapas_config = tapas_classifier_model.TapasClassifierConfig(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=experiment_utils.num_train_steps(),
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
      average_approximation_function=tapas_classifier_model.\
        AverageApproximationFunction(FLAGS.average_approximation_function),
      cell_select_pref=FLAGS.cell_select_pref,
      answer_loss_cutoff=FLAGS.answer_loss_cutoff,
      grad_clipping=FLAGS.grad_clipping,
      disabled_features=FLAGS.disabled_features,
      max_num_rows=FLAGS.max_num_rows,
      max_num_columns=FLAGS.max_num_columns,
      average_logits_per_cell=FLAGS.average_logits_per_cell,
      init_cell_selection_weights_to_zero=FLAGS
      .init_cell_selection_weights_to_zero,
      select_one_column=FLAGS.select_one_column,
      allow_empty_column_selection=FLAGS.allow_empty_column_selection,
      disable_position_embeddings=FLAGS.disable_position_embeddings,
      disable_per_token_loss=FLAGS.disable_per_token_loss)

  model_fn = tapas_classifier_model.model_fn_builder(tapas_config)
  estimator = experiment_utils.build_estimator(model_fn)

  if FLAGS.do_train:
    tf.io.gfile.makedirs(FLAGS.model_dir)
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
    estimator.train(
        input_fn=train_input_fn, max_steps=experiment_utils.num_train_steps())

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
      include_id=True)
  if FLAGS.do_eval:
    prev_checkpoint = None
    while True:
      checkpoint = estimator.latest_checkpoint()

      if checkpoint == prev_checkpoint:
        if FLAGS.minutes_to_sleep_before_predictions > 0:
          tf.logging.info("Sleeping %d mins before evaluation",
                          FLAGS.minutes_to_sleep_before_predictions)
          time.sleep(FLAGS.minutes_to_sleep_before_predictions * 60)
          continue

      tf.logging.info("Running eval: %s", FLAGS.eval_name)
      result = estimator.evaluate(
          input_fn=eval_input_fn,
          steps=FLAGS.num_eval_steps,
          name=FLAGS.eval_name)
      tf.logging.info("Eval result:\n%s", result)

      current_step = int(os.path.basename(checkpoint).split("-")[1])
      if experiment_utils.num_train_steps(
      ) is None or current_step >= experiment_utils.num_train_steps():
        tf.logging.info("Evaluation finished after training step %d",
                        current_step)
        break

      prev_checkpoint = checkpoint

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
        include_id=True)

    prev_checkpoint = None
    while True:
      checkpoint = estimator.latest_checkpoint()

      if checkpoint == prev_checkpoint:
        tf.logging.info("Sleeping 5 mins before predicting")
        time.sleep(5 * 60)
        continue

      current_step = int(os.path.basename(checkpoint).split("-")[1])

      def _predict_and_export_metrics(
          mode,
          input_fn,
          input_file,
          interactions_file,
      ):
        """Exports model predictions and calculates denotation metric."""
        # Predict for each new checkpoint.
        tf.logging.info("Running predictor for step %d.", current_step)
        result = estimator.predict(input_fn=input_fn)
        if FLAGS.prediction_output_dir:
          output_dir = FLAGS.prediction_output_dir
          tf.io.gfile.makedirs(output_dir)
        else:
          output_dir = FLAGS.model_dir
        output_predict_file = os.path.join(
            output_dir, f"{mode}_results_{current_step}.tsv")
        prediction_utils.write_predictions(result, output_predict_file,
                                           do_model_aggregation,
                                           do_model_classification,
                                           FLAGS.cell_classification_threshold)

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
              mode + "_results_sequence_{}.tsv").format(current_step)
          prediction_utils.write_predictions(
              result_sequence, output_predict_file_sequence,
              do_model_aggregation, do_model_classification,
              FLAGS.cell_classification_threshold)


      if FLAGS.input_file_predict is not None:
        _predict_and_export_metrics(
            mode="predict",
            input_fn=predict_input_fn,
            input_file=FLAGS.input_file_predict,
            interactions_file=FLAGS.predict_interactions_file)
      _predict_and_export_metrics(
          mode="eval",
          input_fn=eval_input_fn,
          input_file=FLAGS.input_file_eval,
          interactions_file=FLAGS.eval_interactions_file)

      num_train_steps = experiment_utils.num_train_steps()
      if num_train_steps is None or current_step >= num_train_steps:
        tf.logging.info("Predictor finished after training step %d",
                        current_step)
        break
      prev_checkpoint = checkpoint


if __name__ == "__main__":
  app.run(main)
