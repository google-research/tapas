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
"""Defines the methods to use for table pruning."""

from typing import Any, Callable, Dict, List, Optional, Text

import dataclasses

from tapas.models import segmented_tensor
from tapas.models import tapas_classifier_model_utils as utils
from tapas.models.bert import modeling
from tapas.models.bert import table_bert
from tapas.protos import table_pruning_pb2
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from google.protobuf import text_format

PRUNING_SCOPE = "pruning"
_SEQUENCE_OUTPUT_KEEP_PROB = 0.9

_Loss = table_pruning_pb2.Loss
_TablePruningModel = table_pruning_pb2.TablePruningModel
_Tapas = table_pruning_pb2.TAPAS
_AvgCosSimilarity = table_pruning_pb2.AvgCosSimilarity
_FirstTokens = table_pruning_pb2.FirstTokens
_HardSelection = _Loss.HardSelection
_Unsupervised = _Loss.Unsupervised
_Regularization = _Unsupervised.Regularization


@dataclasses.dataclass(frozen=True)
class Scores:
  column_scores: Optional[tf.Tensor] = None
  column_probs: Optional[tf.Tensor] = None
  column_score_mask: Optional[tf.Tensor] = None
  token_scores: Optional[tf.Tensor] = None
  reduced_token_scores: Optional[tf.Tensor] = None
  gather_op: Optional[Callable[[Dict[Text, tf.Tensor], tf.Tensor],
                               Dict[Text, tf.Tensor]]] = None


class TablePruningSelector:
  """Base class for table pruning."""

  def __init__(self, config, max_num_tokens, max_num_columns):
    self._config = config
    self._max_num_columns = max_num_columns
    self._max_num_tokens = max_num_tokens
    # The default value of bert_init_checkpoint is None.
    # It is called in tapas_classifier_model.
    self.bert_init_checkpoint = None

  def select_columns(self, mode, features):
    return self._select_columns(mode, features)

  def _select_columns(self, mode, features):
    raise NotImplementedError()

  def compute_loss(self, required_columns, column_scores,
                   column_score_mask,
                   token_scores):
    del required_columns, column_scores, column_score_mask, token_scores
    return None

  def apply_hard_selection(self, mode, scores):
    del mode
    return scores

  def compute_scores(self, mode, features):
    del mode, features
    return Scores()

  def get_sequence_output(self):
    return None

  def get_pooled_output(self):
    return None

  def should_add_classification_loss(self):
    return False


class NoTablePruning(TablePruningSelector):
  """Disables the use of table pruning."""

  def __init__(self):
    super(NoTablePruning, self).__init__(
        config=None, max_num_columns=-1, max_num_tokens=-1)
    self._model = None

  def _select_columns(self, mode, features):
    return None


class ModelPruningSelector(TablePruningSelector):
  """Runs an independent model to select columns."""

  def _init_create_model_selector(self, config, max_num_tokens):
    return create_loss_selector(config, max_num_tokens)

  def __init__(self, config, max_num_columns, max_num_tokens):
    super(ModelPruningSelector, self).__init__(
        config=config,
        max_num_columns=max_num_columns,
        max_num_tokens=max_num_tokens)
    # create the model selector that precise the type of learning:
    # supervised or unsupervised.
    self._model = self._init_create_model_selector(config, max_num_tokens)
    self._mask_top_k = MaskTopK(max_num_tokens=self._max_num_tokens)

  def compute_loss(self, required_columns, column_scores,
                   column_score_mask,
                   token_scores):
    return self._model.compute_loss(required_columns, column_scores,
                                    column_score_mask, token_scores)

  def apply_hard_selection(self, mode, scores):
    return self._model.apply_hard_selection(mode, scores)

  def _gather_op(
      self,
  ):
    """function used by the bert model to gather half of the seqence."""

    def gather_top_k(features, scores):
      input_mask = features["input_mask"]
      # <int32>[batch_size, max_num_tokens, 1]
      indexes = self._get_index_gather(scores=scores, input_mask=input_mask)
      input_shape = modeling.get_shape_list(input_mask, expected_rank=2)
      seq_len = input_shape[1]
      reduced_features = {}
      for k, v in features.items():
        if v is not None:
          v_shape = v.shape.as_list()
          if len(v_shape) == 2 and v_shape[1] == seq_len:
            # <float32>[batch_size, max_num_tokens]
            reduced_features[k] = tf.gather_nd(
                params=v, indices=indexes, batch_dims=1)
          else:
            reduced_features[k] = v
      return reduced_features

    return gather_top_k

  def _get_index_gather(self, scores,
                        input_mask):
    """Computes the indexes of gather_op given a scores tensor."""
    # <float32>[batch_size, seq_len]
    scores_mask = self._mask_top_k.apply_hard_selection(
        scores=scores) * tf.cast(
            input_mask, dtype=tf.float32)
    # <int32>[batch_size, max_num_tokens]
    sorted_indexes = tf.math.top_k(scores_mask, self._max_num_tokens)[1]
    # <int32>[batch_size, max_num_tokens, 1]
    return tf.expand_dims(sorted_indexes, -1)

  def _gather_scores(self, scores,
                     input_mask):
    """Gather the smaller tensor of scores."""
    # <int32>[batch_size, max_num_tokens, 1]
    indexes = self._get_index_gather(scores=scores, input_mask=input_mask)
    # <float32>[batch_size, max_num_tokens]
    return tf.gather_nd(indices=indexes, params=scores, batch_dims=1)

  def compute_scores(self, mode, features):
    """Computes column_scores, column_probs, column_score_mask, token_scores."""
    stats = self._computes_column_and_token_scores(mode, features)
    hard_selection = self.apply_hard_selection(
        mode=mode, scores=stats.token_scores)
    input_mask = features["input_mask"]
    token_scores = hard_selection * tf.cast(input_mask, dtype=tf.float32)
    reduced_token_scores = self._gather_scores(
        scores=token_scores, input_mask=input_mask)
    return Scores(
        column_scores=stats.column_scores,
        column_probs=stats.column_probs,
        column_score_mask=stats.column_score_mask,
        token_scores=token_scores,
        reduced_token_scores=reduced_token_scores,
        gather_op=self._gather_op())

  def _compute_columns_stats(self, column_scores,
                             column_ids):
    """Computes the scores probability and mask scores."""
    # <float32>[batch_size, max_num_columns]
    column_probs = tf.sigmoid(column_scores)
    # <float32>[batch_size, max_num_columns]
    column_score_mask = get_mask_columns_scores(
        max_num_columns=self._max_num_columns,
        column_ids=column_ids,
        scores=column_probs)
    column_scores = column_scores * column_score_mask
    column_probs = column_probs * column_score_mask
    return Scores(
        column_scores=column_scores,
        column_probs=column_probs,
        column_score_mask=column_score_mask)

  def _computes_column_and_token_scores(
      self, mode, features):
    """Computes column_scores, column_probs, column_score_mask, token_scores."""
    column_ids = features["column_ids"]
    input_mask = features["input_mask"]
    # <float32>[batch_size, max_num_columns]
    column_scores = self.select_columns(mode, features)
    column_stats = self._compute_columns_stats(column_scores, column_ids)
    column_probs = column_stats.column_probs
    # <float32>[batch_size, seq_length]
    token_scores = get_token_scores_from_column_scores(
        column_ids=column_ids,
        column_probs=column_probs,
        input_mask=input_mask,
        max_num_columns=self._max_num_columns)
    return Scores(
        column_scores=column_stats.column_scores,
        column_probs=column_probs,
        column_score_mask=column_stats.column_score_mask,
        token_scores=token_scores)

  def should_add_classification_loss(self):
    raise NotImplementedError()


class OnesTablePruning(ModelPruningSelector):
  """Empty class for selecting all columns."""

  def _init_create_model_selector(self, config, max_num_tokens):
    return MaskTopK(max_num_tokens=max_num_tokens)

  def __init__(self, config, max_num_columns,
               max_num_tokens):
    super(OnesTablePruning, self).__init__(
        config=config,
        max_num_columns=max_num_columns,
        max_num_tokens=max_num_tokens)

  def _select_columns(self, mode, features):
    del mode
    input_shape = modeling.get_shape_list(
        features["input_ids"], expected_rank=2)
    batch_size = input_shape[0]
    return tf.ones([batch_size, self._max_num_columns])

  def apply_hard_selection(self, mode, scores):
    # Selects only the first tokens.
    return self._model.apply_hard_selection(scores)

  def compute_loss(self, required_columns, column_scores,
                   column_score_mask,
                   token_scores):
    return None

  def should_add_classification_loss(self):
    return False


class TapasPruningSelector(ModelPruningSelector):
  """Runs an independent Tapas model to select columns."""

  def __init__(self, config, max_num_columns, max_num_rows,
               max_num_tokens):
    super(TapasPruningSelector, self).__init__(
        config=config,
        max_num_columns=max_num_columns,
        max_num_tokens=max_num_tokens)
    self._max_num_rows = max_num_rows
    self._bert_config = modeling.BertConfig.from_json_file(
        config.bert_config_file)
    self.bert_init_checkpoint = config.bert_init_checkpoint
    self._reset_position_index_per_cell = config.reset_position_index_per_cell

  def _compute_column_scores_from_token_scores(
      self, mode, output_layer,
      features):
    """Gets the columns scores by avereging the tokens scores."""
    with tf.variable_scope(PRUNING_SCOPE, reuse=tf.AUTO_REUSE):
      if mode == tf_estimator.ModeKeys.TRAIN:
        output_layer = tf.nn.dropout(
            output_layer, keep_prob=_SEQUENCE_OUTPUT_KEEP_PROB)
      input_mask = features["input_mask"]
      row_ids = features["row_ids"]
      column_ids = features["column_ids"]

      # Construct indices for the table.
      row_index = segmented_tensor.IndexMap(
          indices=tf.minimum(row_ids, self._max_num_rows - 1),
          num_segments=self._max_num_rows,
          batch_dims=1)
      col_index = segmented_tensor.IndexMap(
          indices=tf.minimum(column_ids, self._max_num_columns),
          num_segments=self._max_num_columns + 1,
          batch_dims=1)
      cell_index = segmented_tensor.ProductIndexMap(row_index, col_index)

      # Masks.
      # <float32>[batch_size, seq_length]
      input_mask_float = tf.cast(input_mask, tf.float32)
      # Mask for cells that exist in the table (i.e. that are not padding).
      cell_mask, _ = segmented_tensor.reduce_mean(input_mask_float, cell_index)

      # Compute logits per column which can be used to select a column.
      # <float32>[batch_size, max_num_columns]
      column_scores = utils.compute_column_logits(
          output_layer=output_layer,
          cell_index=cell_index,
          cell_mask=cell_mask,
          init_cell_selection_weights_to_zero=False,
          allow_empty_column_selection=False)[:, 1:]
      column_scores = tf.debugging.assert_all_finite(
          column_scores, "column_scores contains nan values.")
      return column_scores

  def _select_columns(self, mode, features):
    with tf.variable_scope(PRUNING_SCOPE, reuse=tf.AUTO_REUSE):
      model = table_bert.create_model(
          features=features,
          mode=mode,
          bert_config=self._bert_config,
          reset_position_index_per_cell=self._reset_position_index_per_cell)
      output_layer = model.get_sequence_output()
      self._output_layer = model.get_sequence_output()
      self._pooled_output = model.get_pooled_output()
    column_scores = self._compute_column_scores_from_token_scores(
        mode=mode, output_layer=output_layer, features=features)
    return column_scores

  def _compute_token_scores(self, mode,
                            features):
    """Computes the token probabilities using the pruning tapas outputlayer."""
    with tf.variable_scope(PRUNING_SCOPE, reuse=tf.AUTO_REUSE):
      model = table_bert.create_model(
          features=features,
          mode=mode,
          bert_config=self._bert_config,
          reset_position_index_per_cell=self._reset_position_index_per_cell)
      output_layer = model.get_sequence_output()
      self._output_layer = model.get_sequence_output()
      self._pooled_output = model.get_pooled_output()
      if mode == tf_estimator.ModeKeys.TRAIN:
        output_layer = tf.nn.dropout(
            output_layer, keep_prob=_SEQUENCE_OUTPUT_KEEP_PROB)
      # No temperature is used.
      token_logits = utils.compute_token_logits(
          output_layer=output_layer,
          temperature=1.0,
          init_cell_selection_weights_to_zero=False)
      token_logits = tf.debugging.assert_all_finite(
          token_logits, "token_logits contains nan values.")
      proba_tokens = tf.sigmoid(token_logits)
      input_mask = features["input_mask"]
      column_ids = features["column_ids"]
      question_mask_proba_tokens = tf.where(
          column_ids <= tf.zeros_like(column_ids), tf.ones_like(proba_tokens),
          proba_tokens)
      input_mask_proba_tokens = question_mask_proba_tokens * tf.cast(
          input_mask, dtype=tf.float32)
      return input_mask_proba_tokens

  def _computes_column_and_token_scores(
      self, mode, features):
    if self._config.selection == _Tapas.Selection.COLUMNS:
      return super(TapasPruningSelector,
                   self)._computes_column_and_token_scores(mode, features)
    elif self._config.selection == _Tapas.Selection.TOKENS:
      # <float32>[batch_size, seq_length]
      token_scores = self._compute_token_scores(mode, features)
      # <float32>[batch_size, max_num_columns]
      column_scores = self._compute_column_scores_from_token_scores(
          mode=mode, output_layer=self._output_layer, features=features)
      column_ids = features["column_ids"]
      columns_stats = self._compute_columns_stats(column_scores, column_ids)
      return Scores(
          column_scores=columns_stats.column_scores,
          column_probs=columns_stats.column_probs,
          column_score_mask=columns_stats.column_score_mask,
          token_scores=token_scores)
    else:
      raise NotImplementedError(
          f"Tapas.Selection not implemented {self._config.selection}")

  def get_sequence_output(self):
    return self._output_layer

  def get_pooled_output(self):
    return self._pooled_output

  def should_add_classification_loss(self):
    return self._model.should_add_classification_loss()


class AverageCosineSimilaritySelector(ModelPruningSelector):
  """Computes the cosine similarity between the average question and column token."""

  def __init__(self, vocab_size, hidden_size,
               initializer_range, max_num_columns,
               type_vocab_size, disabled_features,
               disable_position_embeddings, max_position_embeddings,
               config, max_num_tokens):
    super(AverageCosineSimilaritySelector, self).__init__(
        config=config,
        max_num_columns=max_num_columns,
        max_num_tokens=max_num_tokens)
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._initializer_range = initializer_range
    self._max_num_columns = max_num_columns
    self._type_vocab_size = type_vocab_size
    self._use_position_embeddings = not disable_position_embeddings
    self._disabled_features = disabled_features
    self._max_position_embeddings = max_position_embeddings
    self._use_positional_embeddings = config.use_positional_embeddings

  def _select_columns(self, mode, features):
    input_mask = features["input_mask"]
    column_ids = features["column_ids"]
    with tf.variable_scope("bert"):
      with tf.variable_scope("embeddings", reuse=tf.compat.v1.AUTO_REUSE):
        input_embeddings, _ = modeling.embedding_lookup(
            input_ids=features["input_ids"],
            vocab_size=self._vocab_size,
            embedding_size=self._hidden_size,
            initializer_range=self._initializer_range,
            word_embedding_name="word_embeddings")
        if self._use_positional_embeddings:
          token_type_ids = []
          token_type_features = [
              "segment_ids", "column_ids", "row_ids", "prev_label_ids",
              "column_ranks", "inv_column_ranks", "numeric_relations"
          ]
          for key in token_type_features:
            if self._disabled_features is not None and key in self._disabled_features:
              token_type_ids.append(tf.zeros_like(features[key]))
            else:
              token_type_ids.append(features[key])
          input_embeddings = modeling.embedding_postprocessor(
              input_tensor=input_embeddings,
              use_token_type=True,
              token_type_ids=token_type_ids,
              token_type_vocab_size=self._type_vocab_size,
              token_type_embedding_name="token_type_embeddings",
              use_position_embeddings=self._use_position_embeddings,
              position_embedding_name="position_embeddings",
              initializer_range=self._initializer_range,
              max_position_embeddings=self._max_position_embeddings,
              extra_embeddings=None,
              dropout_prob=0.0)
        # Indexes all the zero values from the input_mask by (max_num_columns+1)
        # The index 0 is for the question and from 1 to max_num_columns included
        # is for the columns.
        masked_col_ids = column_ids * input_mask + (1 - input_mask) * (
            self._max_num_columns + 1)
        col_index = segmented_tensor.IndexMap(
            indices=masked_col_ids,
            num_segments=self._max_num_columns + 2,
            batch_dims=1)
        average_embeddings, _ = segmented_tensor.reduce_mean(
            input_embeddings, col_index)
        # Removes the last index as it contains the avg of non selected values
        average_embeddings = average_embeddings[:, :-1]
        normalize_average_embeddings = tf.math.l2_normalize(
            average_embeddings, axis=2)
        questions_embeddings = normalize_average_embeddings[:, :1]
        columns_embeddings = normalize_average_embeddings[:, 1:]
        multiply = columns_embeddings * questions_embeddings
        multiply = tf.where(
            tf.is_nan(multiply), tf.zeros_like(multiply), multiply)
        column_scores = tf.math.reduce_sum(
            multiply, axis=-1, name="column_scores")
        return column_scores

  def should_add_classification_loss(self):
    return False


def create_selector(
    table_pruning_config_file,
    vocab_size,
    hidden_size,
    initializer_range,
    max_num_columns,
    max_num_rows,
    type_vocab_size,
    disabled_features,
    disable_position_embeddings,
    max_position_embeddings,
):
  """Activates the scoring model according to table pruning config."""
  if not table_pruning_config_file:
    return NoTablePruning()
  config = table_pruning_pb2.TablePruningModel()
  with tf.gfile.Open(table_pruning_config_file) as input_file:
    # ParseLines
    config = text_format.ParseLines(input_file,
                                    table_pruning_pb2.TablePruningModel())
  model = config.WhichOneof("table_pruning_model")
  max_num_tokens = config.max_num_tokens
  if model == "avg_cos_similarity":
    return AverageCosineSimilaritySelector(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        initializer_range=initializer_range,
        max_num_columns=max_num_columns,
        type_vocab_size=type_vocab_size,
        disabled_features=disabled_features,
        disable_position_embeddings=disable_position_embeddings,
        max_position_embeddings=max_position_embeddings,
        config=config.avg_cos_similarity,
        max_num_tokens=max_num_tokens)
  elif model == "tapas":
    return TapasPruningSelector(
        config=config.tapas,
        max_num_columns=max_num_columns,
        max_num_tokens=max_num_tokens,
        max_num_rows=max_num_rows)
  elif model == "first_tokens":
    return OnesTablePruning(
        config=config.first_tokens,
        max_num_columns=max_num_columns,
        max_num_tokens=max_num_tokens)
  else:
    raise NotImplementedError(f"TablePruningModel not implemented {model}")


class LossSelector:
  """Base class for model selection supervised and unsupervised."""

  def __init__(self, config, max_num_tokens):
    self._config = config
    self._train_hard_selection = create_hard_selection_selector(
        config.train, max_num_tokens=max_num_tokens)
    self._eval_hard_selection = create_hard_selection_selector(
        config.eval, max_num_tokens=max_num_tokens)
    self._should_add_classification_loss = config.add_classification_loss

  def compute_loss(self, required_columns, column_scores,
                   column_score_mask,
                   token_scores):
    raise NotImplementedError()

  def apply_hard_selection(self, mode, scores):
    if mode == tf_estimator.ModeKeys.TRAIN:
      return self._train_hard_selection.apply_hard_selection(scores)
    return self._eval_hard_selection.apply_hard_selection(scores)

  def should_add_classification_loss(self):
    return self._should_add_classification_loss




class Unsupervised(LossSelector):
  """Computes the unsupervised loss acconrding to the config."""

  def __init__(self, config, max_num_tokens):
    super(Unsupervised, self).__init__(
        config=config, max_num_tokens=max_num_tokens)
    self._regularization = config.unsupervised.regularization

  def compute_loss(self, required_columns, column_scores,
                   column_score_mask,
                   token_scores):
    if self._regularization == _Regularization.NONE:
      return None
    elif self._regularization == _Regularization.L1:
      # token_scores: <float32>[batch_size, seq_length]
      return tf.reduce_mean(token_scores)
    elif self._regularization == _Regularization.L2:
      # token_scores: <float32>[batch_size, seq_length]
      return tf.reduce_mean(token_scores**2)
    elif self._regularization == _Regularization.L1_L2:
      # token_scores: <float32>[batch_size, seq_length]
      batch_l1 = tf.reduce_mean(token_scores, axis=1)
      return tf.reduce_mean(batch_l1 * batch_l1)
    else:
      raise NotImplementedError(
          f"Unsupervised loss is not implemented {self._regularization}")


def create_loss_selector(config, max_num_tokens):
  """Creates the loss selector according to the config."""
  config_loss = config.loss
  loss = config_loss.WhichOneof("loss")
  if loss == "unsupervised":
    return Unsupervised(config_loss, max_num_tokens)
  else:
    raise NotImplementedError(f"LossSelector not implemented {loss}")


class HardSelection:
  """Defines the hard selection strategy used for train or for evaluation."""

  def apply_hard_selection(self, scores):
    raise NotImplementedError()


class NoHardSelection(HardSelection):

  def apply_hard_selection(self, scores):
    return scores


class MaskTopK(HardSelection):
  """Selects the topk tokens. Returns the mask when aplying selection."""

  def __init__(self, max_num_tokens):
    self._max_num_tokens = max_num_tokens

  def apply_hard_selection(self, scores):
    # <int32>[batch_size, seq_length]
    # Using argsort(argsort(.)) gives the rank of the score.
    sorted_indexes = tf.argsort(
        tf.argsort(scores, direction="DESCENDING", axis=-1, stable=True))
    # <float32>[batch_size, seq_length]
    mask = tf.cast(
        sorted_indexes < tf.ones_like(sorted_indexes) * self._max_num_tokens,
        dtype=tf.float32)
    return mask


class TopK(MaskTopK):
  """Selects the topk tokens scores.

  Returns the scores when aplying selection.
  """

  def apply_hard_selection(self, scores):
    mask = super(TopK, self).apply_hard_selection(scores)
    return mask * scores


def create_hard_selection_selector(config,
                                   max_num_tokens):
  if config.selection_fn == _HardSelection.SelectionFn.ALL:
    return NoHardSelection()
  elif config.selection_fn == _HardSelection.SelectionFn.TOP_K:
    return TopK(max_num_tokens=max_num_tokens)
  elif config.selection_fn == _HardSelection.SelectionFn.MASK_TOP_K:
    return MaskTopK(max_num_tokens=max_num_tokens)
  else:
    raise NotImplementedError(
        f"HardSelection not implemented {config.selection_fn}")


def get_mask_columns_scores(max_num_columns, column_ids,
                            scores):
  """Extracts the columns mask.

  Contains 1 for the scored columns and 0 the other columns.
  Args:
    max_num_columns: float contains the maximum number of columns.
    column_ids: <int32>[batch_size, seq_length] additional to the columns' ids
      (1,to max_number_columns), the value 0 refers to question tokens and
      padding.
    scores: <float32>[batch_size, max_num_columns] contains the column scores.

  Returns:
    <float32>[batch_size, max_num_columns]: Extracts the columns mask scores.
    Using the input mask and the column_ids recovers the columns mask scores.
  """
  # <float32>[batch_size, max_num_columns]
  ranges = tf.ones_like(scores) * tf.cast(
      tf.range(max_num_columns), dtype=tf.float32)
  # <float32>[batch_size, max_num_columns]
  max_num_columns_ids = tf.expand_dims(
      tf.cast(tf.reduce_max(column_ids, axis=1), dtype=tf.float32), -1)
  # <bool>[batch_size, max_num_columns]
  # max_num_columns_ids = max_num_columns + 1 as the value 0 refers to the
  # padding and to the questions tokens.
  # That follows selecting ranges < max_num_columns_ids.
  condition = tf.math.less(ranges, max_num_columns_ids)
  # <float32>[batch_size, max_num_columns]
  column_score_mask = tf.cast(condition, dtype=tf.float32)
  return column_score_mask


def get_token_scores_from_column_scores(
    column_ids,
    column_probs,
    input_mask,
    max_num_columns,
):
  """Given the columns scores in [0,1] extracts the tokens scores.

  It also gives a score of 1.0 for the question's tokens and padding.

  Args:
    column_ids: <int32>[batch_size, seq_length] additional to the columns' ids
      [1, max_num_columns] the value 0 refers to question tokens and padding.
    column_probs: <float32>[batch_size, max_column_id]: contains only the
      columns' scores: question score or padding not included. The expected
        values are in [0,1].
    input_mask: <float32>[batch_size, seq_length] used to zero-out the padding.
    max_num_columns: the maximum number of columns.

  Returns:
    <float32>[batch_size, seq_length]: The tokens' scores.
  """
  col_index = segmented_tensor.IndexMap(
      indices=column_ids, num_segments=max_num_columns + 1, batch_dims=1)
  # <float32>[batch size, max_num_columns+1]: it contains the question at pos 0.
  # The scores for the question and padding is 1.
  padded_column_scores = tf.pad(
      column_probs, paddings=[[0, 0], [1, 0]], constant_values=1.0)
  # <float32>[batch_size, seq_length]
  return segmented_tensor.gather(
      index=col_index, values=padded_column_scores) * tf.cast(
          input_mask, dtype=tf.float32)


def get_table_pruning_loss(
    table_selector,
    table_selector_output,
    do_model_aggregation,
    do_model_classification,
    initial_features,
    is_training,
    config,
    classification_fun):
  """Returns the table pruning loss if applicable."""
  table_pruning_loss = None
  if (table_selector.should_add_classification_loss() and
      table_selector.get_sequence_output() is not None):
    init_row_ids = initial_features["row_ids"]
    init_table_mask = tf.cast(init_row_ids > 0, dtype=tf.int32)
    init_aggregation_function_id = (
        tf.squeeze(initial_features["aggregation_function_id"], axis=[1])
        if do_model_aggregation else None)
    init_classification_class_index = (
        tf.squeeze(initial_features["classification_class_index"], axis=[1])
        if do_model_classification else None)
    init_answer, init_numeric_values, init_numeric_values_scale = (
        utils.extract_answer_from_features(
            features=initial_features,
            use_answer_as_supervision=config.use_answer_as_supervision))
    with tf.variable_scope(PRUNING_SCOPE, reuse=tf.AUTO_REUSE):
      outputs_pruning = classification_fun(
          config=config,
          output_layer=table_selector.get_sequence_output(),
          output_layer_aggregation=table_selector.get_pooled_output(),
          label_ids=initial_features["label_ids"],
          input_mask=initial_features["input_mask"],
          table_mask=init_table_mask,
          aggregation_function_id=init_aggregation_function_id,
          answer=init_answer,
          numeric_values=init_numeric_values,
          numeric_values_scale=init_numeric_values_scale,
          is_training=is_training,
          row_ids=init_row_ids,
          column_ids=initial_features["column_ids"],
          classification_class_index=init_classification_class_index)
      if table_pruning_loss is not None:
        table_pruning_loss += outputs_pruning.total_loss
      else:
        table_pruning_loss = outputs_pruning.total_loss
  return table_pruning_loss
