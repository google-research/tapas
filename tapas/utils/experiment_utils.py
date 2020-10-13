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
"""Experiment utilities."""

import datetime
import json
import os
import time
from typing import Iterable, Optional, Text, Tuple, Mapping

from absl import flags
from absl import logging
from tapas.models.bert import modeling
import tensorflow.compat.v1 as tf


flags.DEFINE_string("model_dir", None,
                    "The directory where the model will be stored.")
flags.DEFINE_string("session_master", None, "Tensorflow master address.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("tf_random_seed", None, "Random seed for tensorflow")

flags.DEFINE_integer("train_batch_size", 128, "Batch size for training set.")
flags.DEFINE_integer("eval_batch_size", 32, "Batch size for evaluation set.")
flags.DEFINE_integer("predict_batch_size", 32, "Batch size for inference.")
flags.DEFINE_integer("gradient_accumulation_steps", 1,
                     "Accumulate gradients across multiple steps.")

# Training parameters.
flags.DEFINE_integer("num_train_examples", None,
                     "Number of examples seen during training.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("warmup_ratio", 0.01, "Proportion of warmup steps.")

# Evaluation parameters.
flags.DEFINE_integer("num_eval_steps", None,
                     "Number of steps to take during evaluation.")
flags.DEFINE_integer("eval_throttle_secs", 600,
                     "Minimum number of seconds to wait between evaluations")
flags.DEFINE_integer("eval_start_delay_secs", 120,
                     "Number of seconds to wait before starting evaluations.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_float("bert_config_attention_probs_dropout_prob", None,
                   "Override for BertConfig.attention_probs_dropout_prob")
flags.DEFINE_float("bert_config_hidden_dropout_prob", None,
                   "Override for BertConfig.hidden_dropout_prob")
flags.DEFINE_float("bert_config_initializer_range", None,
                   "Override for BertConfig.initializer_range")
flags.DEFINE_float("bert_config_softmax_temperature", None,
                   "Override for BertConfig.softmax_temperature")

# Checkpointing.
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "Number of steps between checkpoint saves.")
flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "Max number of checkpoints to keep")
flags.DEFINE_float(
    "keep_checkpoint_every_n_hours", 4.0,
    "Additional checkpoints every n hours in addition to those"
    "kept by keep_checkpoint_max")

# TPU flags.
flags.DEFINE_bool("use_tpu", None, "Use TPU model.")
flags.DEFINE_integer("tpu_iterations_per_loop", 1000,
                     "TPU batch iterations per loop.")
flags.DEFINE_integer(
    "num_tpu_cores", None,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# Cloud TPU flags.
tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

FLAGS = flags.FLAGS


def bert_config_from_flags():
  """Reads the BERT config from flags."""
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.bert_config_attention_probs_dropout_prob is not None:
    bert_config.attention_probs_dropout_prob = (
        FLAGS.bert_config_attention_probs_dropout_prob)
  if FLAGS.bert_config_hidden_dropout_prob is not None:
    bert_config.hidden_dropout_prob = FLAGS.bert_config_hidden_dropout_prob
  if FLAGS.bert_config_initializer_range is not None:
    bert_config.initializer_range = FLAGS.bert_config_initializer_range
  if FLAGS.bert_config_softmax_temperature is not None:
    bert_config.softmax_temperature = FLAGS.bert_config_softmax_temperature
  return bert_config


def num_train_steps():
  if FLAGS.num_train_examples is None:
    return None
  return FLAGS.num_train_examples // FLAGS.train_batch_size


def num_warmup_steps():
  num_steps = num_train_steps()
  if num_steps is None:
    return None
  return int(num_steps * FLAGS.warmup_ratio)


def build_estimator(model_fn):
  """Builds a TPUEstimator using the common experiment flags."""
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.tpu_iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  return tf.estimator.tpu.TPUEstimator(
      params={"gradient_accumulation_steps": FLAGS.gradient_accumulation_steps},
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size // \
          FLAGS.gradient_accumulation_steps,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)


def iterate_checkpoints(
    model_dir,
    marker_file_prefix,
    total_steps,
    single_step = None,
    minutes_to_sleep = 5,
):
  """Iterate over checkpoints as they appear until the final one is reached.

  By default iterates over all checkpoints until completion, if `single_step`
  argument is used only that checkpoint is returned, whether it exists or not.
  The `marker_file_prefix` is used to write empty files as a checkpoint is
  successfully yielded to prevent repeated work if the job is restarted.

  Args:
    model_dir: Location where checkpoints live.
    marker_file_prefix: Location to write an empty file that marks the
      checkpoint as processed successfully. Ignored when `single_step` is used.
    total_steps: After each read over checkpoints, finish if `total_steps` is
      reached or `None` is passed.
    single_step: If specified, only return the checkpoint for this step.
    minutes_to_sleep: Number of minutes to spleep between iterations.

  Yields:
    A tuple with a step number and checkpoint path
  """
  if single_step is not None:
    checkpoint = os.path.join(model_dir, f"model.ckpt-{single_step}")
    yield single_step, checkpoint
    return
  done = set()
  while True:
    state = tf.train.get_checkpoint_state(model_dir)
    checkpoints = state.all_model_checkpoint_paths if state is not None else []
    found_pending_checkpoint = False
    for checkpoint in checkpoints:
      step = int(os.path.basename(checkpoint).split("-")[1])
      if step in done:
        continue
      done.add(step)
      if not tf.gfile.Exists(f"{checkpoint}.index"):
        tf.logging.info(f"Skipping step {step} since checkpoint is missing")
        continue
      marker_file = f"{marker_file_prefix}-{step}.done"
      if tf.gfile.Exists(marker_file):
        tf.logging.info(f"Skipping step {step} since marker file was found")
        continue
      yield step, checkpoint
      # To force the file to be created we need to write something in it.
      with tf.io.gfile.GFile(marker_file, "w") as f:
        f.write(datetime.datetime.now().isoformat() + "\n")
      found_pending_checkpoint = True
      # We will restart the loop since the some checkpoints might have been
      # deleted in the meantime.
      break

    if checkpoints and (total_steps is None or step >= total_steps):
      tf.logging.info(f"Checkpoint loop finished after step {step}")
      return

    if not found_pending_checkpoint and minutes_to_sleep > 0:
      tf.logging.info(f"Sleeping {minutes_to_sleep} mins before next loop")
      time.sleep(minutes_to_sleep * 60)


def save_metrics(
    model_dir,
    mode,
    step,
    metrics,
):
  """Save metrics to file."""
  metric_file_path = os.path.join(model_dir, f"{mode}_metrics_{step}.json")
  logging.info("Writing metrics to: %s", metric_file_path)
  with tf.io.gfile.GFile(metric_file_path, "w") as f:
    metrics = dict(metrics, step=step)
    f.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def get_best_step_for_metric(
    model_dir,
    metric,
):
  """Finds measurements for a metric and returns the step with the best value.

  Args:
    model_dir: Location where the model was trained.
    metric: Metric to select.

  Raises:
    ValueError: if no metric files are found.

  Returns:
    Step that maximizes the metric.
  """
  filepattern = os.path.join(model_dir, "eval_metrics_*.json")
  measurements = []
  for path in tf.io.gfile.glob(filepattern):
    with tf.io.gfile.GFile(path) as f:
      measurement = json.load(f)
      measurements.append((measurement[metric], measurement["step"]))

  if not measurements:
    raise ValueError(f"Metrics missing in {model_dir}")

  return max(measurements)[1]


