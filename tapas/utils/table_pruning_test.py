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
"""Tests for third_party.py.tapas.utils.table_pruning."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tapas.models.bert import modeling
from tapas.protos import table_pruning_pb2
from tapas.utils import table_pruning
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

tf.disable_v2_behavior()

_Tapas = table_pruning_pb2.TAPAS
_Loss = table_pruning_pb2.Loss
_HardSelection = _Loss.HardSelection
_ALL = _HardSelection.SelectionFn.ALL
_TOP_K = _HardSelection.SelectionFn.TOP_K
_MASK_TOP_K = _HardSelection.SelectionFn.MASK_TOP_K
_Unsupervised = _Loss.Unsupervised
_Regularization = _Unsupervised.Regularization


def _create_features():
  tf.random.set_random_seed(42)
  # First example verifies columns extraction:
  # Expected score for the 3rd column is 0.0.
  # Second example verifies the cosine similarity:
  # The question contains the same word as the 1st column.
  # Expected score of the 1st column is close to 1.0.
  column_ids = tf.constant([[0, 0, 0, 1, 1, 2, 2, 2, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
  input_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
  input_ids = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
  default_tensor = tf.zeros_like(input_ids)
  features = {
      "column_ids": column_ids,
      "input_mask": input_mask,
      "input_ids": input_ids,
      "segment_ids": default_tensor,
      "row_ids": default_tensor,
      "prev_label_ids": default_tensor,
      "column_ranks": default_tensor,
      "inv_column_ranks": default_tensor,
      "numeric_relations": default_tensor
  }
  return features


class TablePruningTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_mask_columns_scores(self,):
    with self.cached_session() as sess:
      column_scores = tf.constant([[0.3, 0.2, -0.4, 0.0, 0.0],
                                   [0., 1., 0., 0.0, 0.0]])
      column_ids = tf.constant([[0, 0, 1, 1, 1, 2, 3], [0, 1, 2, 2, 2, 1, 0]])
      columns_mask = table_pruning.get_mask_columns_scores(
          max_num_columns=5, column_ids=column_ids, scores=column_scores)
      sess.run(tf.global_variables_initializer())
      # In column_ids in the first example we have only 3 columns (1,2,3).
      # The 0 value is not counted as it represents the question and padding.
      # In this example the columns (4, 5) are not represented.
      # The expected result is 1 for the first 3 columns and 0 for the others.
      expected_columns_mask = [[1., 1., 1., 0., 0.], [1., 1., 0., 0., 0.]]
      columns_mask = sess.run(columns_mask)
      self.assertAllClose(columns_mask, expected_columns_mask, atol=1e-5)

  def test_get_token_scores_from_column_scores(self,):
    with self.cached_session() as sess:
      column_probs = tf.constant([[0.3, 0.2, 0.8], [0.3, 0.2, 0.8]])
      column_ids = tf.constant([[0, 0, 1, 1, 1, 2, 3], [0, 1, 2, 2, 2, 1, 0]])
      input_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
      token_scores = table_pruning.get_token_scores_from_column_scores(
          column_ids=column_ids,
          column_probs=column_probs,
          input_mask=input_mask,
          max_num_columns=3)
      sess.run(tf.global_variables_initializer())
      # The expected columns scores are related to the column_ids, column_probs
      # and input_mask.
      # The question indexed by 0 in column_ids should be scored 1.0.
      # The padding indexed by 0 in column_ids and by 0 in input_mask should be
      # scored 0.0.
      # The columns indexed by 1,2,3 in column_ids should be scored by their
      # probability represented in column_probs where the probability of the
      # column i is located in column_probs[i-1].
      expected_token_scores = [[1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8],
                               [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]]
      token_scores = sess.run(token_scores)
      self.assertAllClose(token_scores, expected_token_scores, atol=1e-5)

  @parameterized.named_parameters(
      ("no_hard_selection", _HardSelection(selection_fn=_ALL), [[
          1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8
      ], [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]]),
      ("top_k", _HardSelection(selection_fn=_TOP_K), [[
          1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.8
      ], [1.0, 0.3, 0.2, 0.0, 0.0, 0.3, 0.0]]),
      ("mask_top_k", _HardSelection(selection_fn=_MASK_TOP_K),
       [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 1., 0.]]),
  )
  def test_hard_selection(self, config, expected_res):
    with self.cached_session() as sess:
      scores = tf.constant([[1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8],
                            [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]])
      hard_selection_selector = table_pruning.create_hard_selection_selector(
          config=config, max_num_tokens=4)
      hard_selection_mask = hard_selection_selector.apply_hard_selection(
          scores=scores)
      sess.run(tf.global_variables_initializer())
      hard_selection_mask = sess.run(hard_selection_mask)
      tf.logging.info("-------------hard_selection_mask---------------")
      tf.logging.info(hard_selection_mask)
      self.assertAllClose(hard_selection_mask, expected_res, atol=1e-5)

  @parameterized.named_parameters(
      ("no_hard_selection", _HardSelection(selection_fn=_ALL), [[
          1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8
      ], [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]]),
      ("top_k", _HardSelection(selection_fn=_TOP_K), [[
          1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.8
      ], [1.0, 0.3, 0.2, 0.0, 0.0, 0.3, 0.0]]),
      ("mask_top_k", _HardSelection(selection_fn=_MASK_TOP_K), [[
          1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0
      ], [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
  )
  def test_loss_selector_with_hard_selection(self, config, expected_res):
    with self.cached_session() as sess:
      scores = tf.constant([[1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8],
                            [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]])
      loss_selector = table_pruning.LossSelector(
          config=_Loss(train=config, eval=config), max_num_tokens=4)
      hard_selection_mask = loss_selector.apply_hard_selection(
          scores=scores, mode=tf_estimator.ModeKeys.TRAIN)
      sess.run(tf.global_variables_initializer())
      hard_selection_mask = sess.run(hard_selection_mask)
      tf.logging.info("-------------hard_selection_mask---------------")
      tf.logging.info(hard_selection_mask)
      self.assertAllClose(hard_selection_mask, expected_res, atol=1e-5)

  @parameterized.named_parameters(
      ("no_hard_selection", _HardSelection(selection_fn=_ALL), [[
          1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8
      ], [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]]),
      ("top_k", _HardSelection(selection_fn=_TOP_K), [[
          1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.8
      ], [1.0, 0.3, 0.2, 0.0, 0.0, 0.3, 0.0]]),
      ("mask_top_k", _HardSelection(selection_fn=_MASK_TOP_K), [[
          1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0
      ], [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
  )
  def test_tapas_with_hard_selection(self, config, expected_res):
    with self.cached_session() as sess:
      scores = tf.constant([[1.0, 1.0, 0.3, 0.3, 0.3, 0.2, 0.8],
                            [1.0, 0.3, 0.2, 0.2, 0.2, 0.3, 0.0]])
      bert_config_file = tempfile.mktemp()
      bert_config = modeling.BertConfig(
          type_vocab_size=[3, 256, 256, 2, 256, 256, 10],
          vocab_size=64,
          hidden_size=16,
          num_hidden_layers=4,
          num_attention_heads=2,
          intermediate_size=32,
          max_position_embeddings=64)
      bert_config.to_json_file(bert_config_file)
      config = _Tapas(
          bert_config_file=bert_config_file,
          selection=_Tapas.Selection.COLUMNS,
          loss=_Loss(
              train=config,
              eval=config,
              unsupervised=_Unsupervised(regularization=_Regularization.NONE)))
      selector = table_pruning.TapasPruningSelector(
          config=config, max_num_columns=4, max_num_rows=4, max_num_tokens=4)
      hard_selection_mask = selector.apply_hard_selection(
          scores=scores, mode=tf_estimator.ModeKeys.TRAIN)
      sess.run(tf.global_variables_initializer())
      hard_selection_mask = sess.run(hard_selection_mask)
      tf.logging.info("-------------hard_selection_mask---------------")
      tf.logging.info(hard_selection_mask)
      self.assertAllClose(hard_selection_mask, expected_res, atol=1e-5)

  @parameterized.named_parameters(
      ("none_loss", _Regularization.NONE, None),
      ("l1_loss", _Regularization.L1, 0.35),
      ("l2_loss", _Regularization.L2, 0.28214285),
      ("l1l2_loss", _Regularization.L1_L2, 0.13112244),
  )
  def test_unsupervised_loss(self, regularization, expected_loss):
    with self.cached_session() as sess:
      token_scores = tf.constant([[1.00, 1.00, 0.30, 0.00, 0.00, 0.00, 0.80],
                                  [1.00, 0.30, 0.20, 0.00, 0.00, 0.30, 0.00]])
      max_num_tokens = 4
      config = _Loss(
          unsupervised=_Unsupervised(regularization=regularization),
          train=_HardSelection(selection_fn=_TOP_K),
          eval=_HardSelection(selection_fn=_MASK_TOP_K))
      sess.run(tf.global_variables_initializer())
      loss = table_pruning.Unsupervised(
          config=config, max_num_tokens=max_num_tokens)
      computed_loss = loss.compute_loss(
          required_columns=None,
          column_scores=None,
          column_score_mask=None,
          token_scores=token_scores)
      if expected_loss is None:
        self.assertIsNone(computed_loss)
      else:
        computed_loss = sess.run(computed_loss)
        tf.logging.info("-------------computed_loss---------------")
        tf.logging.info(computed_loss)
        self.assertAlmostEqual(computed_loss, expected_loss, 5)


  def test_tapas_column_logits(self):
    # Validate that all the transformer parameters are used and that the
    # output has the right shape and is properly masked.
    bert_config_file = tempfile.mktemp()
    num_hidden_layers = 4
    type_vocab_size = [3, 256, 256, 2, 256, 256, 10]
    bert_config = modeling.BertConfig(
        type_vocab_size=type_vocab_size,
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64)
    bert_config.to_json_file(bert_config_file)

    with self.cached_session() as sess:
      features = _create_features()
      max_num_columns = 6
      config = _Tapas(
          bert_config_file=bert_config_file,
          selection=_Tapas.Selection.COLUMNS,
          loss=_Loss(
              unsupervised=_Unsupervised(regularization=_Regularization.L1)))
      max_num_tokens = 6
      model = table_pruning.TapasPruningSelector(
          config=config,
          max_num_columns=max_num_columns,
          max_num_rows=max_num_columns,
          max_num_tokens=max_num_tokens)

      column_scores = model.select_columns(tf_estimator.ModeKeys.TRAIN,
                                           features)

      tf.logging.info("*** Features ***")
      for name in sorted(features):
        tf.logging.info("  name = %s, shape = %s", name, features[name].shape)
      tf.logging.info("**** Trainable Variables ****")
      tvars = tf.trainable_variables()
      self.assertLen(tvars, 16 * num_hidden_layers + len(type_vocab_size) + 8)
      for var in tvars:
        tf.logging.info("  name = %s, shape = %s", var.name, var.shape)
      sess.run(tf.global_variables_initializer())
      column_scores = sess.run(column_scores)
      tf.logging.info("*** Scores ***")
      self.assertLen(column_scores, 2)
      tf.logging.info("*** Scores example 0 ***")
      scores_example_0 = column_scores[0]
      tf.logging.info(scores_example_0)
      # At this stage we don't apply yet the sigmoid so the column_scores for
      # non existing columns in column_ids are scored -10000.0.
      for i in range(2, max_num_columns):
        self.assertEqual(scores_example_0[i], -10000.0)
      tf.logging.info("*** Scores example 1 ***")
      scores_example_1 = column_scores[1]
      tf.logging.info(scores_example_1)
      for i in range(1, max_num_columns):
        self.assertEqual(scores_example_1[i], -10000.0)

  def test_disable_table_pruning(self,):
    features = _create_features()
    model = table_pruning.NoTablePruning()
    column_scores = model.select_columns(tf_estimator.ModeKeys.TRAIN, features)
    self.assertIsNone(column_scores)
    loss = model.compute_loss(column_scores, column_scores, column_scores, None)
    self.assertIsNone(loss)

  def test_first_tokens(self,):
    with self.cached_session() as sess:
      features = _create_features()
      max_num_columns = 6
      column_ids = features["column_ids"]
      max_num_tokens = 4
      model = table_pruning.OnesTablePruning(
          config=table_pruning_pb2.FirstTokens(),
          max_num_columns=max_num_columns,
          max_num_tokens=max_num_tokens)
      column_scores = model.select_columns(tf_estimator.ModeKeys.TRAIN,
                                           features)
      loss = model.compute_loss(column_scores, column_scores, column_scores,
                                None)
      token_scores = tf.cast(tf.ones_like(column_ids), dtype=tf.float32)
      token_scores = model.apply_hard_selection(
          mode=tf_estimator.ModeKeys.TRAIN, scores=token_scores)
      sess.run(tf.global_variables_initializer())
      column_ids, column_scores = sess.run([column_ids, column_scores])
      tf.logging.info("-------------column_ids---------------")
      tf.logging.info(column_ids)
      tf.logging.info("-------------column_scores---------------")
      tf.logging.info(column_scores)
      # The expected result is ones for all the column scores.
      self.assertAllEqual(column_scores, np.ones_like(column_scores))
      # The loss should be None.
      self.assertIsNone(loss)
      token_scores = sess.run(token_scores)
      tf.logging.info("-------------token_scores---------------")
      tf.logging.info(token_scores)
      expected_scores = [[1 for _ in range(10)], [1 for _ in range(10)]]
      for i in range(max_num_tokens, 10):
        expected_scores[0][i] = 0.0
        expected_scores[1][i] = 0.0
      for i, example_scores in enumerate(token_scores):
        self.assertListEqual(list(example_scores), expected_scores[i])

  @parameterized.named_parameters(
      ("unsupevised_l1", "unsupervised", 0.6459053),
  )
  def test_tapas_loss(self, test_name, expected_loss):
    bert_config_file = tempfile.mktemp()
    num_hidden_layers = 4
    type_vocab_size = [3, 256, 256, 2, 256, 256, 10]
    bert_config = modeling.BertConfig(
        type_vocab_size=type_vocab_size,
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64)
    bert_config.to_json_file(bert_config_file)
    with self.cached_session() as sess:
      features = _create_features()
      max_num_columns = 4
      loss_config = _Loss(
          train=_HardSelection(selection_fn=_TOP_K),
          eval=_HardSelection(selection_fn=_MASK_TOP_K),
          unsupervised=_Unsupervised(regularization=_Regularization.L1))
      if test_name == "supervised":
        loss_config = _Loss(
            train=_HardSelection(selection_fn=_TOP_K),
            eval=_HardSelection(selection_fn=_MASK_TOP_K),
            supervised=_Supervised(back_propagation=True))
      config = _Tapas(
          bert_config_file=bert_config_file,
          selection=_Tapas.Selection.COLUMNS,
          loss=loss_config)
      max_num_tokens = 6
      required_columns = tf.constant([[1., 0., 0., 0.], [1., 0., 0., 0.]])
      model = table_pruning.TapasPruningSelector(
          config=config,
          max_num_columns=max_num_columns,
          max_num_rows=max_num_columns,
          max_num_tokens=max_num_tokens)
      column_scores = model.select_columns(tf_estimator.ModeKeys.TRAIN,
                                           features)
      column_score_mask = table_pruning.get_mask_columns_scores(
          max_num_columns=max_num_columns,
          scores=column_scores,
          column_ids=features["column_ids"])
      token_scores = table_pruning.get_token_scores_from_column_scores(
          column_ids=features["column_ids"],
          column_probs=tf.sigmoid(column_scores),
          input_mask=features["input_mask"],
          max_num_columns=max_num_columns)
      loss = model.compute_loss(
          required_columns=required_columns,
          column_scores=column_scores,
          column_score_mask=column_score_mask,
          token_scores=token_scores)
      sess.run(tf.global_variables_initializer())
      required_columns = sess.run(required_columns)
      tf.logging.info("-------------required_columns---------------")
      tf.logging.info(required_columns)
      column_scores = sess.run(column_scores)
      tf.logging.info("-------------column_scores---------------")
      tf.logging.info(column_scores)
      column_score_mask = sess.run(column_score_mask)
      tf.logging.info("-------------column_score_mask---------------")
      tf.logging.info(column_score_mask)
      loss = sess.run(loss)
      tf.logging.info("-------------pruning_loss---------------")
      tf.logging.info(loss)
      self.assertAlmostEqual(loss, expected_loss, 5)

  @parameterized.named_parameters(
      ("column_selection_top_k", _Tapas.Selection.COLUMNS, _TOP_K, [[
          1., 1., 1., 0., 0., 0.50967264, 0.50967264, 0.50967264, 0., 0.
      ], [1., 1., 1., 1., 1., 0.48796302, 0., 0., 0., 0.]]),
      ("token_selection_top_k", _Tapas.Selection.TOKENS, _TOP_K, [[
          1., 1., 1., 0., 0., 0.5039804, 0.5115916, 0.5134441, 0., 0.
      ], [1., 1., 1., 1., 1., 0., 0.49576515, 0., 0., 0.]]),
      ("column_selection_mask_top_k", _Tapas.Selection.COLUMNS, _MASK_TOP_K, [[
          1., 1., 1., 0., 0., 1., 1., 1., 0., 0.
      ], [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.]]),
      ("token_selection_mask_top_k", _Tapas.Selection.TOKENS, _MASK_TOP_K, [[
          1., 1., 1., 0., 0., 1., 1., 1., 0., 0.
      ], [1., 1., 1., 1., 1., 0., 1., 0., 0., 0.]]),
  )
  def test_compute_scores(self, selection, train_selection_fn, expected_scores):
    bert_config_file = tempfile.mktemp()
    num_hidden_layers = 4
    type_vocab_size = [3, 256, 256, 2, 256, 256, 10]
    bert_config = modeling.BertConfig(
        type_vocab_size=type_vocab_size,
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64)
    bert_config.to_json_file(bert_config_file)
    with self.cached_session() as sess:
      features = _create_features()
      max_num_columns = 4
      max_num_tokens = 6
      config = _Tapas(
          bert_config_file=bert_config_file,
          selection=selection,
          loss=_Loss(
              train=_HardSelection(selection_fn=train_selection_fn),
              eval=_HardSelection(selection_fn=train_selection_fn),
              unsupervised=_Unsupervised(regularization=_Regularization.NONE)))
      selector = table_pruning.TapasPruningSelector(
          config=config,
          max_num_columns=max_num_columns,
          max_num_rows=max_num_columns,
          max_num_tokens=max_num_tokens)
      stats = selector.compute_scores(
          mode=tf_estimator.ModeKeys.TRAIN, features=features)
      token_scores = stats.token_scores
      sess.run(tf.global_variables_initializer())
      token_scores = sess.run(token_scores)
      tf.logging.info("-------------token_scores---------------")
      tf.logging.info(token_scores)
      self.assertAllClose(token_scores, expected_scores, atol=1e-5)

  def test_gather_nd(self):
    bert_config_file = tempfile.mktemp()
    num_hidden_layers = 4
    type_vocab_size = [3, 256, 256, 2, 256, 256, 10]
    bert_config = modeling.BertConfig(
        type_vocab_size=type_vocab_size,
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64)
    bert_config.to_json_file(bert_config_file)
    with self.cached_session() as sess:
      features = _create_features()
      max_num_columns = 4
      max_num_tokens = 4
      config = _Tapas(
          bert_config_file=bert_config_file,
          selection=_Tapas.Selection.COLUMNS,
          loss=_Loss(
              train=_HardSelection(selection_fn=_TOP_K),
              eval=_HardSelection(selection_fn=_TOP_K),
              add_classification_loss=True,
              unsupervised=_Unsupervised(regularization=_Regularization.NONE)))
      selector = table_pruning.TapasPruningSelector(
          config=config,
          max_num_columns=max_num_columns,
          max_num_rows=max_num_columns,
          max_num_tokens=max_num_tokens)
      stats = selector.compute_scores(
          mode=tf_estimator.ModeKeys.TRAIN, features=features)
      token_scores = stats.token_scores
      gather_op = stats.gather_op
      new_features = gather_op(features=features, scores=token_scores)
      sess.run(tf.global_variables_initializer())
      new_features = sess.run(new_features)
      expected_column_ids = [[0, 0, 0, 1], [0, 0, 0, 0]]
      expected_input_mask = [[1, 1, 1, 1], [1, 1, 1, 1]]
      expected_input_ids = [[0, 1, 2, 3], [1, 1, 1, 1]]
      tf.logging.info("-------------features---------------")
      for k, v in new_features.items():
        tf.logging.info(f"-------------{k}---------------")
        tf.logging.info(v)
      self.assertAllClose(new_features["column_ids"], expected_column_ids, 5)
      self.assertAllClose(new_features["input_mask"], expected_input_mask, 5)
      self.assertAllClose(new_features["input_ids"], expected_input_ids, 5)


if __name__ == "__main__":
  absltest.main()
