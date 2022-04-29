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
"""Utilities around apache beams."""

from typing import Iterable, List, Tuple

from tapas.protos import interaction_pb2
from tapas.utils import pretrain_utils

to_numpy_seed = pretrain_utils.to_numpy_seed
split_by_table_id_and_write = pretrain_utils.split_by_table_id_and_write


def rekey(
    interaction):
  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(interaction)
  iid = interaction.table.table_id
  iid = hex(to_numpy_seed(iid))
  new_interaction.id = iid
  new_interaction.table.table_id = iid
  return new_interaction


def _get_sharded_ranges(
    begin,
    end,
    max_length,
):
  """Recursively cuts ranges in half to satisfy 'max_length'."""
  if max_length <= 0:
    raise ValueError("max_length <= 0.")
  length = end - begin
  if length <= max_length:
    return [(begin, end)]
  pivot = begin + length // 2
  return (_get_sharded_ranges(begin, pivot, max_length) +
          _get_sharded_ranges(pivot, end, max_length))


def get_row_sharded_interactions(
    interaction,
    max_num_cells,
):
  """Equally shards the interaction row-wise to satisfy 'max_num_cells'."""
  num_columns = len(interaction.table.columns)
  max_num_rows = max_num_cells // num_columns
  if max_num_rows == 0:
    return
  for begin, end in _get_sharded_ranges(
      begin=0,
      end=len(interaction.table.rows),
      max_length=max_num_rows,
  ):
    new_interaction = interaction_pb2.Interaction()
    new_interaction.CopyFrom(interaction)
    del new_interaction.table.rows[:]
    for row in interaction.table.rows[begin:end]:
      new_interaction.table.rows.add().CopyFrom(row)
    yield new_interaction
