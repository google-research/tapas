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
import os
import tempfile
from typing import Iterator, Tuple

from absl.testing import parameterized
import numpy as np
from tapas.datasets import table_dataset
from tapas.datasets import table_dataset_test_utils
from tapas.models import tapas_classifier_model
from tapas.models.bert import modeling
from tapas.protos import table_pruning_pb2
from tapas.utils import attention_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from google.protobuf import text_format


_Loss = table_pruning_pb2.Loss
_TablePruningModel = table_pruning_pb2.TablePruningModel
_Tapas = table_pruning_pb2.TAPAS
_HardSelection = _Loss.HardSelection
_TOP_K = _HardSelection.SelectionFn.TOP_K
_MASK_TOP_K = _HardSelection.SelectionFn.MASK_TOP_K
_Unsupervised = _Loss.Unsupervised
_Regularization = _Unsupervised.Regularization


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
    tf.logging.info("Setting random seed to {}".format(420))
    np.random.seed(420)

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
        disable_per_token_loss=params.get("disable_per_token_loss", False),
        classification_label_weight=params.get("classification_label_weight",
                                               {}),
        table_pruning_config_file=params.get("table_pruning_config_file", None),
        restrict_attention_mode=params.get(
            "restrict_attention_mode",
            attention_utils.RestrictAttentionMode.FULL),
        restrict_attention_bucket_size=2,
        restrict_attention_header_size=4,
        cell_cross_entropy=params.get("cell_cross_entropy", False),
        cell_cross_entropy_hard_em=params.get("cell_cross_entropy_hard_em",
                                              False),
    )
    model_fn = tapas_classifier_model.model_fn_builder(tapas_config)

    estimator = tf_estimator.tpu.TPUEstimator(
        params={
            "gradient_accumulation_steps":
                params.get("gradient_accumulation_steps", 1),
            "fail_if_missing_variables_in_checkpoint":
                params.get("fail_if_missing_variables_in_checkpoint", False)
        },
        use_tpu=params["use_tpu"],
        model_fn=model_fn,
        config=tf_estimator.tpu.RunConfig(
            model_dir=tempfile.mkdtemp(dir=self.get_temp_dir()),
            save_summary_steps=params["num_train_steps"],
            save_checkpoints_steps=params["num_train_steps"]),
        train_batch_size=params["batch_size"],
        predict_batch_size=params["batch_size"],
        eval_batch_size=params["batch_size"])

    return estimator

  @parameterized.parameters(
      {},
      dict(load_checkpoint=True),
      dict(do_model_aggregation=True),
      dict(load_checkpoint=True, do_model_aggregation=True),
      dict(
          load_checkpoint=True,
          do_model_aggregation=True,
          use_answer_as_supervision=True),
      dict(
          load_checkpoint=True,
          do_model_aggregation=True,
          use_answer_as_supervision=True,
          use_normalized_answer_loss=True),
      dict(select_one_column=True),
      dict(do_gradient_accumulation=True),
      dict(attention=attention_utils.RestrictAttentionMode.HEADWISE_EFFICIENT),
      dict(attention=attention_utils.RestrictAttentionMode.TABLE_ATTENTION),
      dict(cell_cross_entropy=True),
      dict(cell_cross_entropy_hard_em=True),
  )
  def test_build_model_train_and_evaluate(
      self,
      load_checkpoint=False,
      do_model_aggregation=False,
      use_answer_as_supervision=False,
      use_normalized_answer_loss=False,
      select_one_column=False,
      do_gradient_accumulation=False,
      attention=None,
      cell_cross_entropy=False,
      cell_cross_entropy_hard_em=False,
  ):
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
        average_approximation_function=tapas_classifier_model
        .AverageApproximationFunction.RATIO,
        cell_select_pref=0.5,
        answer_loss_cutoff=100,
        grad_clipping=4.0,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=select_one_column,
        gradient_accumulation_steps=2 if do_gradient_accumulation else 1,
        cell_cross_entropy=cell_cross_entropy,
        cell_cross_entropy_hard_em=cell_cross_entropy_hard_em,
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

    if attention is not None:
      # Test that a normal model can get loaded into a sparse model
      load_checkpoint = True
      params["restrict_attention_mode"] = attention

    if load_checkpoint:
      params["init_checkpoint"] = estimator.model_dir
      params["fail_if_missing_variables_in_checkpoint"] = True

      # This mode adds extra parameters that will not be present.
      if attention in [
          attention_utils.RestrictAttentionMode.TABLE_ATTENTION
      ]:
        params["fail_if_missing_variables_in_checkpoint"] = False

      estimator = self._create_estimator(params)
      estimator.train(_input_fn, max_steps=params["num_train_steps"])

    eval_metrics = estimator.evaluate(_input_fn, steps=params["num_eval_steps"])

    for metric_name in ("eval_sequence_accuracy", "eval_mean_label",
                        "eval_loss", "loss"):
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
        classification_label_weight={2: 10},
        aggregation_loss_importance=0.8,
        use_answer_as_supervision=use_answer_as_supervision,
        answer_loss_importance=0.001,
        use_normalized_answer_loss=False,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=tapas_classifier_model
        .AverageApproximationFunction.RATIO,
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
    self.assertNotIn(field_name, "column_scores")
    self.assertNotIn(field_name, "token_scores")

  def _build_pruning_unsupervised_config(self, use_columns, bert_config_file,
                                         add_classification_loss):
    if use_columns:
      selection = _Tapas.Selection.COLUMNS
    else:
      selection = _Tapas.Selection.TOKENS
    return _TablePruningModel(
        max_num_tokens=6,
        tapas=_Tapas(
            bert_config_file=bert_config_file,
            selection=selection,
            loss=_Loss(
                train=_HardSelection(selection_fn=_TOP_K),
                eval=_HardSelection(selection_fn=_TOP_K),
                add_classification_loss=add_classification_loss,
                unsupervised=_Unsupervised(
                    regularization=_Regularization.NONE))))


  def _get_default_params_for_table_pruning(self, table_pruning_config_file):
    return dict(
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
        use_answer_as_supervision=False,
        answer_loss_importance=0.001,
        use_normalized_answer_loss=False,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=tapas_classifier_model
        .AverageApproximationFunction.RATIO,
        cell_select_pref=0.5,
        answer_loss_cutoff=100,
        grad_clipping=4.0,
        is_predict=True,
        max_num_rows=64,
        max_num_columns=10,
        average_logits_per_cell=True,
        select_one_column=False,
        disable_per_token_loss=False,
        table_pruning_config_file=table_pruning_config_file)

  def _print_pruning_metrics(self,
                             eval_metrics,
                             display_columns=None,
                             display_pruning_loss=True):
    metrics = [
        "eval_sequence_accuracy",
        "eval_mean_label",
        "eval_loss",
        "loss",
    ]
    if display_pruning_loss:
      metrics.append("pruning_loss",)
    for metric_name in metrics:
      self.assertIn(metric_name, eval_metrics)
      tf.logging.info("----------------------------" + metric_name)
      tf.logging.info(eval_metrics[metric_name])

  @parameterized.named_parameters(
      ("unsupervised_columns_without_pruning_loss", "unsupervised_columns",
       False),
      ("unsupervised_columns_with_pruning_loss", "unsupervised_columns", True),
      ("unsupervised_tokens_without_pruning_loss", "unsupervised_tokens",
       False),
      ("unsupervised_tokens_with_pruning_loss", "unsupervised_tokens", True),
  )
  def test_build_model_train_and_evaluate_pruning(self, test_file,
                                                  add_classification_loss):
    """Tests that we can train, save, load and evaluate the model."""
    # Adding the path according to the input file name.
    bert_config_file = tempfile.mktemp()
    bert_config = modeling.BertConfig.from_dict({
        "vocab_size": 10,
        "type_vocab_size": [3, 256, 256, 2, 256, 256, 10],
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 128,
        "intermediate_size": 512,
    })
    bert_config.to_json_file(bert_config_file)
    use_pruning_loss = True
    if test_file == "unsupervised_columns":
      table_pruning_config = self._build_pruning_unsupervised_config(
          use_columns=True,
          bert_config_file=bert_config_file,
          add_classification_loss=add_classification_loss)
      use_pruning_loss = add_classification_loss
    elif test_file == "unsupervised_tokens":
      table_pruning_config = self._build_pruning_unsupervised_config(
          use_columns=False,
          bert_config_file=bert_config_file,
          add_classification_loss=add_classification_loss)
      use_pruning_loss = add_classification_loss
    else:
      raise ValueError(f"Not supported test_file {test_file}")
    with tempfile.TemporaryDirectory() as dir_path:
      table_pruning_config_path = os.path.join(dir_path, "test.textproto")
      with tf.io.gfile.GFile(table_pruning_config_path, "w") as writer:
        writer.write(text_format.MessageToString(table_pruning_config))
      params = self._get_default_params_for_table_pruning(
          table_pruning_config_file=table_pruning_config_path)

      # For tests max_seq_length = 10, we need max_num_columns = max_seq_length,
      # as the data generator uses max_seq_length to generate all the data,
      # including the max_num_columns.
      params["max_num_columns"] = 10

      # To learn jointly table pruning and tapas,
      # disable_per_token_loss is activated.
      params["disable_per_token_loss"] = False

      estimator = self._create_estimator(params)
      generator_kwargs = self._generator_kwargs(
          add_aggregation_function_id=True,
          add_classification_labels=True,
          add_answer=False,
          include_id=False)

      def _input_fn(params):
        return table_dataset_test_utils.create_random_dataset(
            num_examples=params["batch_size"],
            batch_size=params["batch_size"],
            repeat=False,
            generator_kwargs=generator_kwargs)

      eval_metrics = estimator.evaluate(
          _input_fn, steps=params["num_eval_steps"])
      display_columns = "_tokens" not in test_file
      if use_pruning_loss:
        self._print_pruning_metrics(
            eval_metrics, display_columns=display_columns)
        inital_loss = eval_metrics["pruning_loss"]

        estimator.train(_input_fn, max_steps=params["num_train_steps"])

        eval_metrics = estimator.evaluate(
            _input_fn, steps=params["num_eval_steps"])
        self._print_pruning_metrics(
            eval_metrics, display_columns=display_columns)

        self.assertNotEqual(eval_metrics["pruning_loss"], inital_loss)
      else:
        self._print_pruning_metrics(
            eval_metrics,
            display_columns=display_columns,
            display_pruning_loss=False)
        self.assertNotIn("pruning_loss", eval_metrics)
        estimator.train(_input_fn, max_steps=params["num_train_steps"])

        eval_metrics = estimator.evaluate(
            _input_fn, steps=params["num_eval_steps"])
        self._print_pruning_metrics(
            eval_metrics,
            display_columns=display_columns,
            display_pruning_loss=False)
        self.assertNotIn("pruning_loss", eval_metrics)

      predictions = estimator.predict(_input_fn)
      for prediction in predictions:
        self.assertIn("token_scores", prediction)
        self.assertIn("column_scores", prediction)
        tf.logging.info("---------------------------- token_scores")
        tf.logging.info(prediction["token_scores"])
        break


if __name__ == "__main__":
  tf.test.main()
