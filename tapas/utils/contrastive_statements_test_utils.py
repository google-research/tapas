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
"""Helpers for contrastive statements tests."""

import re
from typing import List, Mapping

from tapas.protos import annotated_text_pb2
from tapas.protos import interaction_pb2

_AnnotatedText = annotated_text_pb2.AnnotatedText


def get_test_interaction():
  """Creates example interaction with annotations."""
  interaction = create_interaction(
      [
          ['Name', 'Age', 'Birthday'],
          ['Bob', '1.7', '24 April 1950'],
          ['Julia', '1.5', '24 April 1951'],
          ['Peter', '1.9', '24 March 1950'],
      ], 'Robert was born on 24 April 1950.', {
          'bob': 'http://en.wikipedia.org/wiki/Bob',
          'robert': 'http://en.wikipedia.org/wiki/Bob',
          'peter': 'http://en.wikipedia.org/wiki/Peter',
          'julia': 'http://en.wikipedia.org/wiki/Julia',
      })
  return interaction


def create_interaction(
    table,
    statement,
    mentions,
):
  """Creates interaction proto with annotations by matching entity mentions."""
  interaction = interaction_pb2.Interaction()
  for index, row in enumerate(table):
    new_row = interaction.table.columns
    if index > 0:
      new_row = interaction.table.rows.add().cells
    for cell in row:
      new_row.add().text = cell
  for row in interaction.table.rows:
    for cell in row.cells:
      cell_text = cell.text.lower()
      if cell_text not in mentions:
        continue
      annotated_text = cell.Extensions[_AnnotatedText.annotated_cell_ext]
      annotation = annotated_text.annotations.add()
      annotation.begin_byte_index = 0
      annotation.end_byte_index = len(cell_text)
      annotation.identifier = mentions[cell_text]

  question = interaction.questions.add()
  question.original_text = statement
  q_annotated_text = question.Extensions[_AnnotatedText.annotated_question_ext]
  question_text = question.original_text.lower()
  for phrase, identifier in mentions.items():
    for match in re.finditer(phrase, question_text):
      annotation = q_annotated_text.annotations.add()
      begin, end = match.span()
      annotation.begin_byte_index = begin
      annotation.end_byte_index = end
      annotation.identifier = identifier
  return interaction
