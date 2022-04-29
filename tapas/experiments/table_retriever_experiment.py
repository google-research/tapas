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
"""Table retriever experiment."""

import csv
import functools
import os
import traceback
from typing import Text, Optional

from absl import app
from absl import flags
from tapas.models import table_retriever_model
from tapas.scripts import eval_table_retriever_utils
from tapas.utils import experiment_utils  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("data_format", "tfrecord", "The input data format.")

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

flags.DEFINE_integer("minutes_to_sleep_before_predictions", 5,
                     "Time in minutes to sleep before starting to predict.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string(
    "eval_name", "default",
    "Name of the current evaluation set. Will be used in Tensorboard.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_float(
    "grad_clipping", None, "If not None, gradients greater in absolute value"
    "than this number are clipped.")

flags.DEFINE_integer(
    "down_projection_dim", 0, "Representation dimension of the query/table"
    "after down projection. If smaller than 1, no projection occurs.")

flags.DEFINE_bool(
    "init_from_single_encoder", True, "If true, expects to load"
    "a checkpoint of a single encoder, that would be used to"
    "initialize both encoders.")

flags.DEFINE_integer("max_query_length", 128,
                     "The query is capped to this length.")

flags.DEFINE_string(
    "compression_type",
    "",
    "Compression to use when reading tfrecords. '' for no compression.",
)

flags.DEFINE_integer(
    "evaluated_checkpoint_step", None,
    "The step for a specific model checkpoint to be evaluated. If None, then"
    "all checkpoints are used.")

flags.DEFINE_string(
    "evaluated_checkpoint_metric", None,
    "The metric used to chose a model checkpoint to be evaluated. If None, then"
    "all checkpoints are used.")

flags.DEFINE_bool("use_out_of_core_negatives", False,
                  "If true, use all the negatives when"
                  "using many TPU cores.")

flags.DEFINE_bool("mask_repeated_tables", False,
                  "If true, mask tables that are repeated within a batch.")

flags.DEFINE_bool("mask_repeated_questions", False,
                  "If true, mask questions that are repeated within a batch.")

flags.DEFINE_bool(
    "ignore_table_content", False,
    "If true, use only the table headers to represent the table.")

flags.DEFINE_bool(
    "use_mined_negatives", False,
    "If true, use mined negatives that should be given as"
    "additional table features.")

flags.DEFINE_list("disabled_features", [],
                  "Features that should be disabled (for ablation studies).")


def _get_test_input_fn(name, input_file):
  """Gets input_fn for eval/predict modes."""
  if input_file is None:
    return None
  input_fn = functools.partial(
      table_retriever_model.input_fn,
      name=name,
      file_patterns=input_file,
      data_format=FLAGS.data_format,
      is_training=False,
      max_seq_length=FLAGS.max_seq_length,
      compression_type=FLAGS.compression_type,
      use_mined_negatives=FLAGS.use_mined_negatives,
      include_id=True)
  return input_fn


def _predict_and_export_metrics(
    mode,
    input_fn,
    checkpoint_path,
    step,
    estimator,
    output_dir,
):
  """Exports model predictions and calculates precision@k."""
  tf.logging.info("Running predictor for step %d.", step)
  result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
  output_predict_file = os.path.join(output_dir, f"{mode}_results_{step}.tsv")
  write_predictions(result, output_predict_file)

  # Compute precision@k.
  if (not FLAGS.evaluated_checkpoint_step or
      not FLAGS.evaluated_checkpoint_metric):
    p_at_k = eval_table_retriever_utils.eval_precision_at_k(
        query_prediction_files=output_predict_file,
        table_prediction_files=output_predict_file,
        make_tables_unique=True)
    experiment_utils.save_metrics(output_dir, mode, step, p_at_k)


def write_predictions(predictions,
                      output_predict_file):
  """Writes predictions to an output TSV file.

  Predictions header: [query_id, query_rep, table_id, table_rep]
  Args:
    predictions: model predictions
    output_predict_file: Path for wrinting the predicitons.
  """
  with tf.io.gfile.GFile(output_predict_file, "w") as write_file:
    header = [
        "query_id",
        "query_rep",
        "table_id",
        "table_rep",
    ]
    writer = csv.DictWriter(write_file, fieldnames=header, delimiter="\t")
    writer.writeheader()

    for prediction in predictions:
      query_id = prediction["query_id"]
      table_id = prediction["table_id"]
      query_rep = prediction["query_rep"]
      table_rep = prediction["table_rep"]

      prediction_to_write = {
          "query_id": query_id[0].decode("utf-8"),
          "query_rep": query_rep.tolist(),
          "table_id": table_id[0].decode("utf-8"),
          "table_rep": table_rep.tolist(),
      }
      writer.writerow(prediction_to_write)


def main(_):
  bert_config = experiment_utils.bert_config_from_flags()
  total_steps = experiment_utils.num_train_steps()
  retriever_config = table_retriever_model.RetrieverConfig(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=total_steps,
      num_warmup_steps=experiment_utils.num_warmup_steps(),
      use_tpu=FLAGS.use_tpu,
      grad_clipping=FLAGS.grad_clipping,
      down_projection_dim=FLAGS.down_projection_dim,
      init_from_single_encoder=FLAGS.init_from_single_encoder,
      max_query_length=FLAGS.max_query_length,
      mask_repeated_tables=FLAGS.mask_repeated_tables,
      mask_repeated_questions=FLAGS.mask_repeated_questions,
      use_out_of_core_negatives=FLAGS.use_out_of_core_negatives,
      ignore_table_content=FLAGS.ignore_table_content,
      disabled_features=FLAGS.disabled_features,
      use_mined_negatives=FLAGS.use_mined_negatives,
  )

  model_fn = table_retriever_model.model_fn_builder(retriever_config)
  estimator = experiment_utils.build_estimator(model_fn)

  if FLAGS.do_train:
    tf.io.gfile.makedirs(FLAGS.model_dir)
    bert_config.to_json_file(os.path.join(FLAGS.model_dir, "bert_config.json"))
    retriever_config.to_json_file(
        os.path.join(FLAGS.model_dir, "tapas_config.json"))
    train_input_fn = functools.partial(
        table_retriever_model.input_fn,
        name="train",
        file_patterns=FLAGS.input_file_train,
        data_format=FLAGS.data_format,
        is_training=True,
        max_seq_length=FLAGS.max_seq_length,
        compression_type=FLAGS.compression_type,
        use_mined_negatives=FLAGS.use_mined_negatives,
        include_id=False)
    estimator.train(input_fn=train_input_fn, max_steps=total_steps)

  eval_input_fn = _get_test_input_fn("eval", FLAGS.input_file_eval)
  if FLAGS.do_eval:
    if eval_input_fn is None:
      raise ValueError("No input_file_eval specified!")
    for _, checkpoint in experiment_utils.iterate_checkpoints(
        model_dir=estimator.model_dir,
        total_steps=total_steps,
        marker_file_prefix=os.path.join(estimator.model_dir,
                                        f"eval_{FLAGS.eval_name}"),
        minutes_to_sleep=FLAGS.minutes_to_sleep_before_predictions):
      tf.logging.info("Running eval: %s", FLAGS.eval_name)
      try:
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=FLAGS.num_eval_steps,
            name=FLAGS.eval_name,
            checkpoint_path=checkpoint)
        tf.logging.info("Eval result:\n%s", result)
      except (ValueError, tf.errors.NotFoundError):
        tf.logging.error("Error getting predictions for checkpoint %s: %s",
                         checkpoint, traceback.format_exc())

  if FLAGS.do_predict:
    predict_input_fn = _get_test_input_fn("predict", FLAGS.input_file_predict)
    if FLAGS.prediction_output_dir:
      prediction_output_dir = FLAGS.prediction_output_dir
      tf.io.gfile.makedirs(prediction_output_dir)
    else:
      prediction_output_dir = estimator.model_dir

    marker_file_prefix = os.path.join(prediction_output_dir, "predict")
    # When two separate jobs are launched we don't want conflicting markers.
    if predict_input_fn is not None:
      marker_file_prefix += "_test"
    if eval_input_fn is not None:
      marker_file_prefix += "_dev"

    single_step = FLAGS.evaluated_checkpoint_step
    if FLAGS.evaluated_checkpoint_metric:
      single_step = experiment_utils.get_best_step_for_metric(
          estimator.model_dir, FLAGS.evaluated_checkpoint_metric)
    for current_step, checkpoint in experiment_utils.iterate_checkpoints(
        model_dir=estimator.model_dir,
        total_steps=total_steps,
        marker_file_prefix=marker_file_prefix,
        single_step=single_step):
      try:
        if predict_input_fn is not None:
          _predict_and_export_metrics(
              mode="predict",
              input_fn=predict_input_fn,
              checkpoint_path=checkpoint,
              step=current_step,
              estimator=estimator,
              output_dir=prediction_output_dir)

        if eval_input_fn is not None:
          _predict_and_export_metrics(
              mode="eval",
              input_fn=eval_input_fn,
              checkpoint_path=checkpoint,
              step=current_step,
              estimator=estimator,
              output_dir=prediction_output_dir)
      except (ValueError, tf.errors.NotFoundError):
        tf.logging.error("Error getting predictions for checkpoint %s: %s",
                         checkpoint, traceback.format_exc())


if __name__ == "__main__":
  app.run(main)
