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
"""Contains the best hyper-parameter configurations for different tasks."""

from typing import Mapping, Any, Text
from tapas.utils import tasks


def get_sqa_hparams():
  return get_hparams(tasks.Task.SQA)


def get_wtq_hparams():
  return get_hparams(tasks.Task.WTQ)


def get_wikisql_super_hparams():
  return get_hparams(tasks.Task.WIKISQL_SUPERVISED)


def get_wikisql_hparams():
  return get_hparams(tasks.Task.WIKISQL)


def get_hparams(task):
  """Gets hpraram dictionary for a given tasks."""
  if task == tasks.Task.SQA:
    return {
        'init_cell_selection_weights_to_zero': False,
        'learning_rate': 5e-5 * (128 / 512),
        'num_train_examples': 200000 * 128,
        'select_one_column': True,
        'allow_empty_column_selection': False,
        'train_batch_size': 128,
        'warmup_ratio': 0.01,
    }
  params = {
      'grad_clipping': 10.0,
      'num_train_examples': 50000 * 512,
      'train_batch_size': 512,
  }
  if task == tasks.Task.WTQ:
    params.update({
        'answer_loss_cutoff': 0.664694,
        'cell_select_pref': 0.207951,
        'huber_loss_delta': 0.121194,
        'init_cell_selection_weights_to_zero': True,
        'learning_rate': 0.0000193581,
        'select_one_column': True,
        'allow_empty_column_selection': False,
        'temperature': 0.0352513,
        'warmup_ratio': 0.128960,
    })
  elif task == tasks.Task.WIKISQL:
    params.update({
        'answer_loss_cutoff': 0.185567,
        'cell_select_pref': 0.611754,
        'huber_loss_delta': 1265.74,
        'init_cell_selection_weights_to_zero': False,
        'learning_rate': 0.0000617164,
        'select_one_column': False,
        'allow_empty_column_selection': False,
        'temperature': 0.107515,
        'warmup_ratio': 0.142400,
    })
  elif task == tasks.Task.WIKISQL_SUPERVISED:
    params.update({
        'answer_loss_cutoff': 36.4519,
        'cell_select_pref': 0.903421,
        'huber_loss_delta': 222.088,
        'init_cell_selection_weights_to_zero': True,
        'learning_rate': 0.0000412331,
        'select_one_column': True,
        'allow_empty_column_selection': True,
        'temperature': 0.763141,
        'warmup_ratio': 0.168479,
    })
  else:
    raise ValueError(f'Unknown task: {task.name}')
  return params
