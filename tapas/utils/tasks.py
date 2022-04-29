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
"""Lists all the fine-tuning tasks supported by Tapas."""

import enum


class Task(enum.Enum):
  """Fine-tuning tasks supported by Tapas."""
  SQA = 0
  WTQ = 1
  WIKISQL = 2
  WIKISQL_SUPERVISED = 3
  TABFACT = 4
  NQ_RETRIEVAL = 7
  HYBRIDQA_RC = 8  # Reading comprehension (RC) of HybridQA.
  HYBRIDQA_E2E = 9  # HybridQA RC using preds of HybridQA Cell Selection.
  HYBRIDQA = 6
  SEM_TAB_FACT = 10  # https://sites.google.com/corp/view/sem-tab-facts
