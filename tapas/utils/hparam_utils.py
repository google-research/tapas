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


def get_hybridqa_hparams():
  return get_hparams(tasks.Task.HYBRIDQA)


def get_hybridqa_rc_hparams():
  return get_hparams(tasks.Task.HYBRIDQA_RC)


def get_tabfact_hparams():
  return get_hparams(tasks.Task.TABFACT)


def get_nq_e2e_hparams():
  return get_hparams(tasks.Task.NQ_RETRIEVAL)


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
  elif task == tasks.Task.TABFACT:
    params.update({
        'disable_per_token_loss': True,
        'num_classification_labels': 2,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.05,
        'num_aggregation_labels': 0,
        'num_train_examples': 80000 * 512,
        'bert_config_attention_probs_dropout_prob': 0.0,
        'bert_config_hidden_dropout_prob': 0.07,
        'compute_denotation_accuracy': False,
        'init_cell_selection_weights_to_zero': False,
        'select_one_column': False,
        'allow_empty_column_selection': False,
    })
  elif task == tasks.Task.NQ_RETRIEVAL:
    params.update({
        'disable_per_token_loss': False,
        'grad_clipping': 10.0,
        'num_classification_labels': 2,
        'num_aggregation_labels': 0,
        'init_cell_selection_weights_to_zero': False,
        'num_train_examples': 50000 * 512,
        'select_one_column': False,
        'allow_empty_column_selection': True,
        'train_batch_size': 512,
        'compute_e2e_retrieval_metrics': True,
        'compute_denotation_accuracy': False,
        'mask_examples_without_labels': True,
        'bert_config_attention_probs_dropout_prob': 0.034,
        'bert_config_hidden_dropout_prob': 0.2,
        'learning_rate': 1e-06,
        'warmup_ratio': 0.0,
        'span_prediction': 'span',
    })
  elif task == tasks.Task.HYBRIDQA:
    params.update({
        'num_train_examples': 80000 * 128,
        'train_batch_size': 128,
        'eval_batch_size': 8,
        'predict_batch_size': 8,
        'num_aggregation_labels': 0,
        'grad_clipping': 10.0,
        'learning_rate': 1e-5,
        'allow_empty_column_selection': False,
        'warmup_ratio': 0.05,
        'max_seq_length': 2048,
        'restrict_attention_bucket_size': 42,
        'restrict_attention_header_size': 116,
        'compute_denotation_accuracy': False,
        'output_token_probabilities': True,
        'compute_cell_selection_metrics': True,
        'cell_cross_entropy': True,
    })
  elif task == tasks.Task.HYBRIDQA_RC:
    params.update({
        'disable_per_token_loss': False,
        'grad_clipping': 10.0,
        'num_classification_labels': 0,
        'num_aggregation_labels': 0,
        'init_cell_selection_weights_to_zero': False,
        'learning_rate': 5e-5,
        'num_train_examples': 25000 * 512,
        'select_one_column': False,
        'train_batch_size': 512,
        'warmup_ratio': 0.01,
        'compute_e2e_retrieval_metrics': True,
        'compute_denotation_accuracy': False,
        'span_prediction': 'span',
    })
  else:
    raise ValueError(f'Unknown task: {task.name}')
  return params
