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

from absl.testing import parameterized
import numpy as np
from tapas.datasets import table_dataset
from tapas.datasets import table_dataset_test_utils
from tapas.models import table_retriever_model
from tapas.models.bert import modeling
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


class TableRetrieverModelTest(parameterized.TestCase, tf.test.TestCase):

  def _generator_kwargs(self, include_id, use_mined_negatives):
    task_type = (
        table_dataset.TableTask.RETRIEVAL_NEGATIVES
        if use_mined_negatives else table_dataset.TableTask.RETRIEVAL)
    return dict(
        max_seq_length=10,
        max_predictions_per_seq=None,
        task_type=task_type,
        add_aggregation_function_id=False,
        add_classification_labels=False,
        add_answer=False,
        include_id=include_id,
        vocab_size=10,
        segment_vocab_size=3,
        num_columns=3,
        num_rows=2,
        add_candidate_answers=False,
        max_num_candidates=10,
    )

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
    tapas_config = table_retriever_model.RetrieverConfig(
        bert_config=bert_config,
        init_checkpoint=params["init_checkpoint"],
        learning_rate=params["learning_rate"],
        num_train_steps=params["num_train_steps"],
        num_warmup_steps=params["num_warmup_steps"],
        use_tpu=params["use_tpu"],
        use_out_of_core_negatives=params["use_out_of_core_negatives"],
        mask_repeated_tables=params["mask_repeated_tables"],
        mask_repeated_questions=params["mask_repeated_questions"],
        use_mined_negatives=params["use_mined_negatives"],
        grad_clipping=params["grad_clipping"],
        down_projection_dim=params["down_projection_dim"],
        init_from_single_encoder=params["init_from_single_encoder"],
        ignore_table_content=params["ignore_table_content"],
        max_query_length=params["max_query_length"],
        disabled_features=[],
    )
    model_fn = table_retriever_model.model_fn_builder(tapas_config)

    estimator = tf_estimator.tpu.TPUEstimator(
        use_tpu=params["use_tpu"],
        model_fn=model_fn,
        config=tf_estimator.tpu.RunConfig(
            model_dir=self.get_temp_dir(),
            save_summary_steps=params["num_train_steps"],
            save_checkpoints_steps=params["num_train_steps"]),
        train_batch_size=params["batch_size"],
        predict_batch_size=params["batch_size"],
        eval_batch_size=params["batch_size"])

    return estimator

  @parameterized.named_parameters(
      ("encoder", False, 0, True, False, False, False, False),
      ("proj_encoder", False, 2, True, False, False, False, False),
      ("checkpt_encoder", True, 0, True, False, False, False, False),
      ("checkpt_proj_encoder", True, 2, True, False, False, False, False),
      ("checkpt_proj", True, 2, False, False, False, False, False),
      ("checkpt_ignore_table", True, 2, False, True, False, False, False),
      ("checkpt_proj_negatives", True, 2, False, False, True, False, False),
      ("checkpt_negatives", True, 0, False, False, True, False, False),
      ("repeated_tables", False, 0, False, False, False, True, False),
      ("repeated_tables_negatives", False, 0, False, False, True, True, False),
      ("repeated_questions", False, 0, False, False, False, False, True),
      ("repeated_questions_tables", False, 0, False, False, False, True, True),
      ("repeated_questions_neg", False, 0, False, False, True, False, True),
  )
  def test_build_model_train_and_evaluate(
      self,
      load_checkpoint,
      down_projection_dim,
      init_from_single_encoder,
      ignore_table_content,
      use_mined_negatives,
      mask_repeated_tables,
      mask_repeated_questions,
  ):
    """Tests that we can train, save, load and evaluate the model."""
    params = dict(
        batch_size=3,
        init_checkpoint=None,
        learning_rate=5e-5,
        num_train_steps=50,
        num_warmup_steps=10,
        num_eval_steps=20,
        use_tpu=False,
        use_out_of_core_negatives=False,
        mask_repeated_tables=mask_repeated_tables,
        mask_repeated_questions=mask_repeated_questions,
        use_mined_negatives=use_mined_negatives,
        grad_clipping=4.0,
        max_query_length=10,
        down_projection_dim=down_projection_dim,
        init_from_single_encoder=init_from_single_encoder,
        ignore_table_content=ignore_table_content,
    )

    estimator = self._create_estimator(params)
    generator_kwargs = self._generator_kwargs(
        include_id=False,
        use_mined_negatives=use_mined_negatives,
    )

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

    for metric_name in ("eval_accuracy", "eval_loss", "loss"):
      self.assertIn(metric_name, eval_metrics)

  @parameterized.named_parameters(
      ("base", False, False, False),
      ("negatives", True, False, False),
      ("repeated_tables_negatives", True, True, False),
      ("repeated_tables", False, True, False),
      ("repeated_questions", False, False, True),
      ("repeated_questions_tables", False, True, True),
  )
  def test_build_model_predict(
      self,
      use_mined_negatives,
      mask_repeated_tables,
      mask_repeated_questions,
  ):
    """Tests that we predict using the model."""
    params = dict(
        batch_size=2,
        init_checkpoint=None,
        learning_rate=5e-5,
        num_train_steps=50,
        num_warmup_steps=10,
        num_eval_steps=20,
        use_tpu=False,
        use_out_of_core_negatives=False,
        mask_repeated_tables=mask_repeated_tables,
        mask_repeated_questions=mask_repeated_questions,
        use_mined_negatives=use_mined_negatives,
        grad_clipping=4.0,
        max_query_length=10,
        down_projection_dim=2,
        init_from_single_encoder=False,
        ignore_table_content=False,
    )

    estimator = self._create_estimator(params)
    generator_kwargs = self._generator_kwargs(
        include_id=True,
        use_mined_negatives=use_mined_negatives,
    )

    def _input_fn(params):
      return table_dataset_test_utils.create_random_dataset(
          num_examples=params["batch_size"],
          batch_size=params["batch_size"],
          repeat=False,
          generator_kwargs=generator_kwargs)

    predictions = estimator.predict(_input_fn)
    for prediction in predictions:
      for field_name in ("table_id", "query_rep", "table_rep"):
        self.assertIn(field_name, prediction)
      print("prediction={}".format(prediction))


if __name__ == "__main__":
  tf.test.main()
