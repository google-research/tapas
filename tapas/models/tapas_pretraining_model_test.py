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

from absl.testing import parameterized
import numpy as np
from tapas.datasets import table_dataset
from tapas.datasets import table_dataset_test_utils
from tapas.models import tapas_pretraining_model
from tapas.models.bert import modeling
import tensorflow.compat.v1 as tf


class TapasPretrainingModelTest(parameterized.TestCase, tf.test.TestCase):

  def _generator_kwargs(self):
    return dict(
        max_seq_length=10,
        max_predictions_per_seq=5,
        task_type=table_dataset.TableTask.PRETRAINING,
        add_aggregation_function_id=False,
        add_classification_labels=False,
        add_answer=False,
        include_id=False,
        vocab_size=10,
        segment_vocab_size=3,
        num_columns=3,
        num_rows=2,
        add_candidate_answers=False,
        max_num_candidates=10)

  def _create_estimator(self, params):
    tf.logging.info("Setting random seed to {}".format(42))
    np.random.seed(42)

    # Small bert model for testing.
    bert_config = modeling.BertConfig.from_dict({
        "vocab_size": 10,
        "type_vocab_size": [3, 256, 256, 2, 256, 256, 10],
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 128,
        "intermediate_size": 512,
    })
    model_fn = tapas_pretraining_model.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=params["init_checkpoint"],
        learning_rate=params["learning_rate"],
        num_train_steps=params["num_train_steps"],
        num_warmup_steps=params["num_warmup_steps"],
        use_tpu=params["use_tpu"])

    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=params["use_tpu"],
        model_fn=model_fn,
        config=tf.estimator.tpu.RunConfig(
            model_dir=self.get_temp_dir(),
            save_summary_steps=params["num_train_steps"],
            save_checkpoints_steps=params["num_train_steps"]),
        train_batch_size=params["batch_size"],
        predict_batch_size=params["batch_size"],
        eval_batch_size=params["batch_size"])

    return estimator

  @parameterized.named_parameters(("no_checkpoint", False),
                                  ("with_checkpoint", True))
  def test_build_model_train_and_evaluate(self, load_checkpoint):
    """Tests that we can train, save, load and evaluate the model."""
    params = {
        "batch_size": 2,
        "init_checkpoint": None,
        "learning_rate": 5e-5,
        "num_train_steps": 50,
        "num_warmup_steps": 10,
        "num_eval_steps": 20,
        "use_tpu": False,
    }

    estimator = self._create_estimator(params)
    generator_kwargs = self._generator_kwargs()

    def _input_fn(params):
      return table_dataset_test_utils.create_random_dataset(
          num_examples=params["batch_size"],
          batch_size=params["batch_size"],
          repeat=True,
          generator_kwargs=generator_kwargs)

    estimator.train(_input_fn, max_steps=params["num_train_steps"])

    if load_checkpoint:
      params.update({"init_checkpoint": self.get_temp_dir()})
      estimator = self._create_estimator(params)
      estimator.train(_input_fn, max_steps=params["num_train_steps"])

    eval_metrics = estimator.evaluate(_input_fn, steps=params["num_eval_steps"])

    for metric_name in ("masked_lm_loss", "masked_lm_accuracy", "loss",
                        "next_sentence_accuracy", "next_sentence_loss"):
      self.assertIn(metric_name, eval_metrics)

    def _predict_input_fn(params):
      dataset = table_dataset_test_utils.create_random_dataset(
          num_examples=params["batch_size"],
          batch_size=params["batch_size"],
          repeat=True,
          generator_kwargs=generator_kwargs)
      return dataset.take(2)

    for predictions in estimator.predict(_predict_input_fn):
      self.assertIn("masked_lm_predictions", predictions)


if __name__ == "__main__":
  tf.test.main()
