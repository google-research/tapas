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
"""TABLE BERT utility functions."""
import dataclasses
import functools
from typing import Any, Optional

from tapas.models.bert import modeling
from tapas.utils import attention_utils
from tapas.utils import tableformer_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

_AttentionMode = attention_utils.RestrictAttentionMode


@dataclasses.dataclass
class CustomAttention:
  attention_mask: Optional[tf.Tensor] = None
  custom_attention_layer: Optional[Any] = None


@dataclasses.dataclass(frozen=True)
class CustomAttentionConfig:
  """Helper config class to generate custom attention."""
  restrict_attention_mode: _AttentionMode = _AttentionMode.FULL
  restrict_attention_bucket_size: int = 0
  restrict_attention_header_size: int = 0
  restrict_attention_row_heads_ratio: float = 0.5
  restrict_attention_sort_after_projection: bool = True
  attention_bias_disabled: int = 0
  attention_bias_use_relative_scalar_only: bool = True


def get_token_type_features():
  return [
      "segment_ids", "column_ids", "row_ids", "prev_label_ids", "column_ranks",
      "inv_column_ranks", "numeric_relations"
  ]


def get_token_type_ids(features, token_type_features, disabled_features):
  token_type_ids = []
  disabled_ids = []
  for i, key in enumerate(token_type_features):
    if disabled_features is not None and key in disabled_features:
      tf.logging.info("Disable %s", key)
      disabled_ids.append(i)
    token_type_ids.append(features[key])
  return token_type_ids, disabled_ids


def run_custom_attention(
    is_training,
    num_attention_heads,
    config,
    features,
):
  """Extracts the custumized attention."""
  attention_mask = None
  custom_attention_layer = None
  num_row_heads = int(num_attention_heads *
                      config.restrict_attention_row_heads_ratio)
  num_column_heads = num_attention_heads - num_row_heads
  restrict_attention_mode = config.restrict_attention_mode
  bucket_size = config.restrict_attention_bucket_size
  header_size = config.restrict_attention_header_size
  if restrict_attention_mode == _AttentionMode.HEADWISE_SAME_COLUMN_OR_ROW:
    attention_mask = attention_utils.compute_headwise_sparse_attention_mask(
        num_row_heads=num_row_heads,
        num_column_heads=num_column_heads,
        bucket_size=bucket_size,
        header_size=header_size,
        **features)
  elif restrict_attention_mode == _AttentionMode.SAME_COLUMN_OR_ROW:
    attention_mask = attention_utils.compute_sparse_attention_mask(**features)
  elif restrict_attention_mode == _AttentionMode.HEADWISE_EFFICIENT:
    custom_attention_layer = attention_utils.create_bucketed_attention_layer(
        input_mask=features["input_mask"],
        input_header=tf.math.equal(features["segment_ids"], 0),
        bucket_size=bucket_size,
        header_size=header_size,
        sort_after_projection=config.restrict_attention_sort_after_projection,
        token_type_ids=[(num_row_heads, True, features["row_ids"]),
                        (num_column_heads, False, features["column_ids"])])

  elif restrict_attention_mode == _AttentionMode.TABLE_ATTENTION:
    custom_attention_layer = functools.partial(
        modeling.attention_layer,
        relative_relation_ids=tableformer_utils.get_relative_relation_ids(
            features, disabled_attention_bias=config.attention_bias_disabled),
        use_relative_scalar_only=config.attention_bias_use_relative_scalar_only,
        relative_relation_ids_vocab_size=tableformer_utils
        .RELATIVE_RELATION_IDS_VOCAB_SIZE,
    )
  elif restrict_attention_mode == _AttentionMode.FULL:
    pass
  else:
    raise ValueError(f"Unknown attention mode: {restrict_attention_mode}")
  return CustomAttention(
      attention_mask=attention_mask,
      custom_attention_layer=custom_attention_layer,
  )


def create_model(
    features,
    mode,
    bert_config,
    restrict_attention_mode=_AttentionMode.FULL,
    restrict_attention_bucket_size=0,
    restrict_attention_header_size=None,
    restrict_attention_row_heads_ratio=0.5,
    restrict_attention_sort_after_projection=True,
    token_weights=None,
    disabled_features=None,
    disable_position_embeddings=False,
    reset_position_index_per_cell=False,
    proj_value_length=None,
    attention_bias_disabled=0,
    attention_bias_use_relative_scalar_only=True,
):
  """Creates a TABLE BERT model."""
  is_training = (mode == tf_estimator.ModeKeys.TRAIN)
  token_type_features = get_token_type_features()
  token_type_ids, disabled_ids = get_token_type_ids(
      features=features,
      token_type_features=token_type_features,
      disabled_features=disabled_features)
  custom_attention = run_custom_attention(
      is_training=is_training,
      num_attention_heads=bert_config.num_attention_heads,
      config=CustomAttentionConfig(
          restrict_attention_mode=restrict_attention_mode,
          restrict_attention_row_heads_ratio=restrict_attention_row_heads_ratio,
          restrict_attention_bucket_size=restrict_attention_bucket_size,
          restrict_attention_header_size=restrict_attention_header_size,
          restrict_attention_sort_after_projection=restrict_attention_sort_after_projection,
          attention_bias_disabled=attention_bias_disabled,
          attention_bias_use_relative_scalar_only=attention_bias_use_relative_scalar_only,
      ),
      features=features,
  )
  return modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=features["input_ids"],
      input_mask=features["input_mask"],
      attention_mask=custom_attention.attention_mask,
      custom_attention_layer=custom_attention.custom_attention_layer,
      token_weights=token_weights,
      token_type_ids=token_type_ids,
      disabled_ids=disabled_ids,
      use_position_embeddings=not disable_position_embeddings,
      reset_position_index_per_cell=reset_position_index_per_cell,
      proj_value_length=proj_value_length,
  )
