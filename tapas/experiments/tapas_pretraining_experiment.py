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
"""TAPAS pretraining experiment."""

import functools
import os
import time

from absl import app
from absl import flags
from tapas.models import tapas_pretraining_model
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

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string(
    "eval_name", "default",
    "Name of the current evaluation set. Will be used in Tensorboard.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_list("disabled_features", [],
                  "Features that should be disabled (for ablation studies).")
flags.DEFINE_bool("disable_position_embeddings", False,
                  "Whether to disable position embeddings.")


def main(_):
  bert_config = experiment_utils.bert_config_from_flags()
  model_fn = tapas_pretraining_model.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=experiment_utils.num_train_steps(),
      num_warmup_steps=experiment_utils.num_warmup_steps(),
      use_tpu=FLAGS.use_tpu,
      disabled_features=FLAGS.disabled_features,
      disable_position_embeddings=FLAGS.disable_position_embeddings)
  estimator = experiment_utils.build_estimator(model_fn)

  if FLAGS.do_train:
    tf.io.gfile.makedirs(FLAGS.model_dir)
    bert_config.to_json_file(os.path.join(FLAGS.model_dir, "bert_config.json"))
    train_input_fn = functools.partial(
        tapas_pretraining_model.input_fn,
        name="train",
        file_patterns=FLAGS.input_file_train,
        data_format=FLAGS.data_format,
        compression_type=FLAGS.compression_type,
        is_training=True,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq)
    estimator.train(
        input_fn=train_input_fn, max_steps=experiment_utils.num_train_steps())

  if FLAGS.do_eval:
    eval_input_fn = functools.partial(
        tapas_pretraining_model.input_fn,
        name="eval",
        file_patterns=FLAGS.input_file_eval,
        data_format=FLAGS.data_format,
        compression_type=FLAGS.compression_type,
        is_training=False,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq)

    current_step = 0
    prev_checkpoint = None
    while True:
      checkpoint = estimator.latest_checkpoint()

      if checkpoint == prev_checkpoint:
        tf.logging.info("Sleeping 5 mins before evaluation")
        time.sleep(5 * 60)
        continue

      tf.logging.info("Running eval: %s", FLAGS.eval_name)
      result = estimator.evaluate(
          input_fn=eval_input_fn,
          steps=FLAGS.num_eval_steps,
          name=FLAGS.eval_name)
      tf.logging.info("Eval result:\n%s", result)

      current_step = int(os.path.basename(checkpoint).split("-")[1])
      if current_step >= experiment_utils.num_train_steps():
        tf.logging.info("Evaluation finished after training step %d",
                        current_step)
        break

      prev_checkpoint = checkpoint


if __name__ == "__main__":
  app.run(main)
