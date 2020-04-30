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
"""Utilitity functions to deal with predictions."""

import ast
import csv
from typing import Any, Iterable, Set, Text, Tuple

import pandas as pd
from tapas.protos import interaction_pb2
import tensorflow.compat.v1 as tf


def parse_coordinates(raw_coordinates):
  """Parses cell coordinates from text."""
  return {ast.literal_eval(x) for x in ast.literal_eval(raw_coordinates)}


def iterate_predictions(prediction_file):
  with tf.io.gfile.GFile(prediction_file, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
      yield row


def iterate_interactions(
    interactions_file):
  for value in tf.python_io.tf_record_iterator(interactions_file):
    interaction = interaction_pb2.Interaction()
    interaction.ParseFromString(value)
    yield interaction


def parse_interaction_id(text):
  return text[:text.rindex('-')]


def table_to_panda_frame(table):
  contents = [[cell.text for cell in row.cells] for row in table.rows]
  headers = [
      f'{column.text}_{index}' for index, column in enumerate(table.columns)
  ]
  return pd.DataFrame(contents, columns=headers)
