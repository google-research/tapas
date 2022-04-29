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
"""Helper methods to build TableFormer layers."""

from typing import Mapping

import tensorflow.compat.v1 as tf

# Size of the vocab for the relative relation ids used in TableFormer.
RELATIVE_RELATION_IDS_VOCAB_SIZE = 13


def get_relative_relation_ids(features,
                              disabled_attention_bias = 0):
  """Returns a relative (attention) relation id matrix.

    The method expects that for a given query-table structure, we have the
    following structural information in the `features`:
    * "row_ids": 0 for query, 0 for header, 1-n for other rows (where n is the
      number of rows excluding the header).
    * "column_ids": 0 for query, 1-m for the table columns (where m is the
      number of columns).
    * "segment_ids": 0 for query, 1 for table tokens.
    * "input_mask": 1 for query+table tokens.
    NB: All features are 0 for [PAD] tokens

    For example, given a query+table structure:
    Q1 Q2 A0 B0 C0
          A1 B1 C1
          A2 B2 C2 PAD1 PAD2 PAD3 PAD4
    The following features must exist:
    features['row_ids'] =
        <int32>[[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]]
    features['column_ids'] =
        <int32>[[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]]
    features['segment_ids'] =
        <int32>[[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    features['input_mask'] = tf.constant(
        <int32>[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

  Args:
    features: Mapping of all token type ids.
    disabled_attention_bias: Disabled attention bias id for ablation
      study only. This should by default be 0 for general purposes.

  Returns:
    Tensor of shape <int32>[batch_size, seq_length, seq_length] representing
    a relative (attention) relation id matrix.
  """
  cell_mask = tf.logical_not(
      tf.math.equal(features['row_ids'], 0)) & tf.math.equal(
          features['segment_ids'], 1) & tf.math.equal(features['input_mask'], 1)
  header_mask = tf.math.equal(features['row_ids'], 0) & tf.math.equal(
      features['segment_ids'], 1) & tf.math.equal(features['input_mask'], 1)
  sent_mask = tf.math.equal(features['input_mask'], 1) & tf.math.equal(
      features['segment_ids'], 0)
  relative_attention_ids = []
  tf.logging.info('Disable attention bias: %s', disabled_attention_bias)

  # "Same row" attention bias with type id = 1.
  if disabled_attention_bias != 1:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['row_ids'], axis=1),
                tf.expand_dims(features['row_ids'], axis=2))
            & tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * 1)
  else:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2)) & tf.math.equal(
                    tf.expand_dims(features['row_ids'], axis=1),
                    tf.expand_dims(features['row_ids'], axis=2))
            & tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * 1)

  # "Same column" attention bias matrix with type id = 2.
  if disabled_attention_bias != 2:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2))
            & tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * 2)
  else:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2)) & tf.math.equal(
                    tf.expand_dims(features['row_ids'], axis=1),
                    tf.expand_dims(features['row_ids'], axis=2))
            & tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * 2)

  # "Same cell" attention bias matrix with type id = 3.
  if disabled_attention_bias == 3:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2)) & tf.math.equal(
                    tf.expand_dims(features['row_ids'], axis=1),
                    tf.expand_dims(features['row_ids'], axis=2))
            & tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * -3)

  # "Cell to its header" bias matrix with type id = 4.
  if disabled_attention_bias != 4:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2))
            & tf.expand_dims(header_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * 4)

  # "Cell to sentence" bias matrix with type id = 5.
  if disabled_attention_bias != 5:
    relative_attention_ids.append(
        tf.cast(
            tf.expand_dims(sent_mask, axis=1)
            & tf.expand_dims(cell_mask, axis=2), tf.int32) * 5)

  # "Header to column cell" bias matrix with type id = 6.
  if disabled_attention_bias != 6:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2))
            & tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(header_mask, axis=2), tf.int32) * 6)

  # "Header to other header" bias matrix with type id = 7.
  if disabled_attention_bias != 7:
    relative_attention_ids.append(
        tf.cast(
            tf.expand_dims(header_mask, axis=1)
            & tf.expand_dims(header_mask, axis=2), tf.int32) * 7)

  # "Header to same header" bias matrix with type id = 8.
  if disabled_attention_bias != 8:
    if disabled_attention_bias != 7:
      relative_attention_ids.append(
          tf.cast(
              tf.math.equal(
                  tf.expand_dims(features['column_ids'], axis=1),
                  tf.expand_dims(features['column_ids'], axis=2))
              & tf.math.equal(
                  tf.expand_dims(features['row_ids'], axis=1),
                  tf.expand_dims(features['row_ids'], axis=2))
              & tf.expand_dims(header_mask, axis=1)
              & tf.expand_dims(header_mask, axis=2), tf.int32) * 1)
    else:
      relative_attention_ids.append(
          tf.cast(
              tf.math.equal(
                  tf.expand_dims(features['column_ids'], axis=1),
                  tf.expand_dims(features['column_ids'], axis=2))
              & tf.math.equal(
                  tf.expand_dims(features['row_ids'], axis=1),
                  tf.expand_dims(features['row_ids'], axis=2))
              & tf.expand_dims(header_mask, axis=1)
              & tf.expand_dims(header_mask, axis=2), tf.int32) * 8)
  else:
    relative_attention_ids.append(
        tf.cast(
            tf.math.equal(
                tf.expand_dims(features['column_ids'], axis=1),
                tf.expand_dims(features['column_ids'], axis=2))
            & tf.math.equal(
                tf.expand_dims(features['row_ids'], axis=1),
                tf.expand_dims(features['row_ids'], axis=2))
            & tf.expand_dims(header_mask, axis=1)
            & tf.expand_dims(header_mask, axis=2), tf.int32) * -7)

  # "Header to sentence" bias matrix with type id = 9.
  if disabled_attention_bias != 9:
    relative_attention_ids.append(
        tf.cast(
            tf.expand_dims(sent_mask, axis=1)
            & tf.expand_dims(header_mask, axis=2), tf.int32) * 9)

  # "Sentence to cell" bias matrix with type id = 10.
  if disabled_attention_bias != 10:
    relative_attention_ids.append(
        tf.cast(
            tf.expand_dims(cell_mask, axis=1)
            & tf.expand_dims(sent_mask, axis=2), tf.int32) * 10)

  # "Sentence to header" bias matrix with type id = 11.
  if disabled_attention_bias != 11:
    relative_attention_ids.append(
        tf.cast(
            tf.expand_dims(header_mask, axis=1)
            & tf.expand_dims(sent_mask, axis=2), tf.int32) * 11)

  # "Sentence to sentence" bias matrix with type id = 12.
  if disabled_attention_bias != 12:
    relative_attention_ids.append(
        tf.cast(
            tf.expand_dims(sent_mask, axis=1)
            & tf.expand_dims(sent_mask, axis=2), tf.int32) * 12)

  return tf.cast(tf.math.add_n(relative_attention_ids), tf.int32)
