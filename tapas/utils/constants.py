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
"""Constants used by Dopa tables project."""

import enum

EMPTY_TEXT = 'EMPTY'

NUMBER_TYPE = 'number'
DATE_TYPE = 'date'


class Relation(enum.Enum):
  HEADER_TO_CELL = 1  # Connects header to cell.
  CELL_TO_HEADER = 2  # Connects cell to header.
  QUERY_TO_HEADER = 3  # Connects query to headers.
  QUERY_TO_CELL = 4  # Connects query to cells.
  ROW_TO_CELL = 5  # Connects row to cells.
  CELL_TO_ROW = 6  # Connects cells to row.
  EQ = 7  # Annotation value is same as cell value
  LT = 8  # Annotation value is less than cell value
  GT = 9  # Annotation value is greater than cell value
