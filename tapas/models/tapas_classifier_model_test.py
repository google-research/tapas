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

from typing import Iterator, Tuple
from absl.testing import parameterized
import numpy as np
from tapas.datasets import table_dataset
from tapas.datasets import table_dataset_test_utils
from tapas.models import tapas_classifier_model
from tapas.models.bert import modeling
import tensorflow.compat.v1 as tf



def bool_tuple(length, true_indexes):
  result = [False] * length
  for idx in true_indexes:
    result[idx] = True
  return tuple(result)


class TapasClassifierModelTest(parameterized.TestCase, tf.test.TestCase):

  def _generator_kwargs(self, add_aggregation_function_id,
                        add_classification_labels, add_answer, include_id):
    return dict(
        max_seq_length=10,
        max_predictions_per_seq=5,
        task_type=table_dataset.TableTask.CLASSIFICATION,
        add_aggregation_function_id=add_aggregation_function_id,
        add_classification_labels=add_classification_labels,
        add_answer=add_answer,
        include_id=include_id,
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
    tapas_config = tapas_classifier_model.TapasClassifierConfig(
        bert_config=bert_config,
        init_checkpoint=params["init_checkpoint"],
        learning_rate=params["learning_rate"],
        num_train_steps=params["num_train_steps"],
        num_warmup_steps=params["num_warmup_steps"],
        use_tpu=params["use_tpu"],
        positive_weight=10.0,
        num_aggregation_labels=params["num_aggregation_labels"],
        num_classification_labels=params["num_classification_labels"],
        aggregation_loss_importance=params["aggregation_loss_importance"],
        use_answer_as_supervision=params["use_answer_as_supervision"],
        answer_loss_importance=params["answer_loss_importance"],
        use_normalized_answer_loss=params["use_normalized_answer_loss"],
        huber_loss_delta=params["huber_loss_delta"],
        temperature=params["temperature"],
        agg_temperature=params["agg_temperature"],
        use_gumbel_for_cells=params["use_gumbel_for_cells"],
        use_gumbel_for_agg=params["use_gumbel_for_agg"],
        average_approximation_function=params["average_approximation_function"],
        cell_select_pref=params["cell_select_pref"],
        answer_loss_cutoff=params["answer_loss_cutoff"],
        grad_clipping=params["grad_clipping"],
        max_num_rows=params["max_num_rows"],
        max_num_columns=params["max_num_columns"],
        average_logits_per_cell=params["average_logits_per_cell"],
        select_one_column=params["select_one_column"],
    )
    model_fn = tapas_classifier_model.model_fn_builder(tapas_config)

    estimator = tf.estimator.tpu.TPUEstimator(
        params={
            "gradient_accumulation_steps":
                params.get("gradient_accumulation_steps", 1),
        },
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

  @parameterized.named_parameters(
      ("no_checkpoint_no_aggregation",) + bool_tuple(6, []),
      ("with_checkpoint_no_agg",) + bool_tuple(6, [0]),
      ("no_checkpoint_with_agg",) + bool_tuple(6, [1]),
      ("with_checkpoint_with_agg",) + bool_tuple(6, [0, 1]),
      ("agg_with_answer_as_supervision",) + bool_tuple(6, [0, 1, 2]),
      ("agg_with_answer_loss_normalization",) + bool_tuple(6, [0, 1, 2, 3]),
      ("no_checkpoint_no_agg_one_column",) + bool_tuple(6, [4]),
      ("gradient_accumulation",) + bool_tuple(6, [5]))
  def test_build_model_train_and_evaluate(
      self, load_checkpoint, do_model_aggregation, use_answer_as_supervision,
      use_normalized_answer_loss, select_one_column, do_gradient_accumulation):
    """Tests that we can train, save, load and evaluate the model."""
    params = dict(
        batch_size=2,
        init_checkpoint=None,
        learning_rate=5e-5,
        num_train_steps=50,
        num_warmup_steps=10,
        num_eval_steps=20,
        use_tpu=False,
        num_aggregation_labels=4 if do_model_aggregation else 0,
        num_classification_labels=0,
        aggregation_loss_importance=0.8,
        use_normalized_answer_loss=use_normalized_answer_loss,
        use_answer_as_supervision=use_answer_as_supervision,
        answer_loss_importance=0.001,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=\
          tapas_classifier_model.AverageApproximationFunction.RATIO,
        cell_select_pref=0.5,
        answer_loss_cutoff=100,
        grad_clipping=4.0,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=select_one_column,
        gradient_accumulation_steps=2 if do_gradient_accumulation else 1,
    )

    estimator = self._create_estimator(params)
    generator_kwargs = self._generator_kwargs(
        do_model_aggregation,
        add_classification_labels=False,
        add_answer=use_answer_as_supervision,
        include_id=False)

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

    for metric_name in ("eval_sequence_accuracy", "eval_recall",
                        "eval_mean_label", "eval_precision", "eval_auc",
                        "eval_accuracy", "eval_loss", "loss"):
      self.assertIn(metric_name, eval_metrics)

  @parameterized.named_parameters(("no_answer", False, False),
                                  ("with_answer", True, False),
                                  ("no_answer_one_column", False, True))
  def test_build_model_predict(self, use_answer_as_supervision,
                               select_one_column):
    """Tests that we predict using the model."""
    params = dict(
        batch_size=2,
        init_checkpoint=None,
        learning_rate=5e-5,
        num_train_steps=50,
        num_warmup_steps=10,
        num_eval_steps=20,
        use_tpu=False,
        num_aggregation_labels=4,
        num_classification_labels=6,
        aggregation_loss_importance=0.8,
        use_answer_as_supervision=use_answer_as_supervision,
        answer_loss_importance=0.001,
        use_normalized_answer_loss=False,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=\
          tapas_classifier_model.AverageApproximationFunction.RATIO,
        cell_select_pref=0.5,
        answer_loss_cutoff=100,
        grad_clipping=4.0,
        is_predict=True,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=select_one_column,
    )

    estimator = self._create_estimator(params)
    generator_kwargs = self._generator_kwargs(
        add_aggregation_function_id=True,
        add_classification_labels=True,
        add_answer=use_answer_as_supervision,
        include_id=True)

    def _input_fn(params):
      return table_dataset_test_utils.create_random_dataset(
          num_examples=params["batch_size"],
          batch_size=params["batch_size"],
          repeat=False,
          generator_kwargs=generator_kwargs)

    predictions = estimator.predict(_input_fn)
    for prediction in predictions:
      for field_name in ("probabilities", "column_ids", "row_ids",
                         "segment_ids", "question_id"):
        self.assertIn(field_name, prediction)
        print("prediction={}".format(prediction))


if __name__ == "__main__":
  tf.test.main()
