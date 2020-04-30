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
"""TABLE BERT utility functions."""

from tapas.models.bert import modeling

import tensorflow.compat.v1 as tf




def create_model(
    features,
    mode,
    bert_config,
    disabled_features=None,
    disable_position_embeddings=False):
  """Creates a TABLE BERT model."""
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  token_type_features = [
      "segment_ids", "column_ids", "row_ids", "prev_label_ids", "column_ranks",
      "inv_column_ranks", "numeric_relations"
  ]
  token_type_ids = []
  for key in token_type_features:
    if disabled_features is not None and key in disabled_features:
      token_type_ids.append(tf.zeros_like(features[key]))
    else:
      token_type_ids.append(features[key])


  return modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=features["input_ids"],
      input_mask=features["input_mask"],
      token_type_ids=token_type_ids,
      use_position_embeddings=not disable_position_embeddings)
