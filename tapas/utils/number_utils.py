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
"""A function for annotating numbers and dates.

   These are meant to be used in an academic setup, for production use cases
   Gold mine or Aqua should be used.
"""

import collections
import datetime
import math
import re
from typing import Text, Tuple, List, Optional

from tapas.protos import interaction_pb2
from tapas.utils import text_utils


# Constants for parsing date expressions.
# Masks that specify (by a bool) which of (year, month, day) will be populated.
_DateMask = collections.namedtuple('_DateMask', ['year', 'month', 'day'])

_YEAR = _DateMask(True, False, False)
_YEAR_MONTH = _DateMask(True, True, False)
_YEAR_MONTH_DAY = _DateMask(True, True, True)
_MONTH = _DateMask(False, True, False)
_MONTH_DAY = _DateMask(False, True, True)

# Pairs of patterns to pass to 'datetime.strptime' and masks specifying which
# fields will be set by the corresponding pattern.
_DATE_PATTERNS = (('%B', _MONTH), ('%Y', _YEAR), ('%Ys', _YEAR),
                  ('%b %Y', _YEAR_MONTH), ('%B %Y', _YEAR_MONTH),
                  ('%B %d', _MONTH_DAY), ('%b %d', _MONTH_DAY), ('%d %b',
                                                                 _MONTH_DAY),
                  ('%d %B', _MONTH_DAY), ('%B %d, %Y', _YEAR_MONTH_DAY),
                  ('%d %B %Y', _YEAR_MONTH_DAY), ('%m-%d-%Y', _YEAR_MONTH_DAY),
                  ('%Y-%m-%d', _YEAR_MONTH_DAY), ('%Y-%m', _YEAR_MONTH),
                  ('%B %Y', _YEAR_MONTH), ('%d %b %Y', _YEAR_MONTH_DAY),
                  ('%Y-%m-%d', _YEAR_MONTH_DAY), ('%b %d, %Y', _YEAR_MONTH_DAY),
                  ('%d.%m.%Y', _YEAR_MONTH_DAY),
                  ('%A, %b %d', _MONTH_DAY), ('%A, %B %d', _MONTH_DAY))

# This mapping is used to convert date patterns to regex patterns.
_FIELD_TO_REGEX = (
    ('%A', r'\w+'),  # Weekday as locale’s full name.
    ('%B', r'\w+'),  # Month as locale’s full name.
    ('%Y', r'\d{4}'),  #  Year with century as a decimal number.
    ('%b', r'\w{3}'),  # Month as locale’s abbreviated name.
    ('%d', r'\d{1,2}'),  # Day of the month as a zero-padded decimal number.
    ('%m', r'\d{1,2}'),  # Month as a zero-padded decimal number.
)


def _process_date_pattern(dp):
  """Compute a regex for each date pattern to use as a prefilter."""
  pattern, mask = dp
  regex = pattern
  regex = regex.replace('.', re.escape('.'))
  regex = regex.replace('-', re.escape('-'))
  regex = regex.replace(' ', r'\s+')
  for field, field_regex in _FIELD_TO_REGEX:
    regex = regex.replace(field, field_regex)
  # Make sure we didn't miss any of the fields.
  assert '%' not in regex, regex
  return pattern, mask, re.compile('^' + regex + '$')


def _process_date_patterns():
  return tuple(_process_date_pattern(dp) for dp in _DATE_PATTERNS)


_PROCESSED_DATE_PATTERNS = _process_date_patterns()

_MAX_DATE_NGRAM_SIZE = 5

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L414.
_NUMBER_WORDS = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve'
]

_ORDINAL_WORDS = [
    'zeroth', 'first', 'second', 'third', 'fourth', 'fith', 'sixth', 'seventh',
    'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth'
]

_ORDINAL_SUFFIXES = ['st', 'nd', 'rd', 'th']

_NUMBER_PATTERN = re.compile(r'((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))')

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L293.
_MIN_YEAR = 1700
_MAX_YEAR = 2016

_INF = float('INF')


def _get_numeric_value_from_date(
    date, mask):
  """Converts date to a numeric value proto with a date value."""
  if date.year < _MIN_YEAR or date.year > _MAX_YEAR:
    raise ValueError('Invalid year: %d' % date.year)

  new_date = interaction_pb2.Date()
  if mask.year:
    new_date.year = date.year
  if mask.month:
    new_date.month = date.month
  if mask.day:
    new_date.day = date.day
  return interaction_pb2.NumericValue(date=new_date)


def _get_span_length_key(span):
  """Sorts span by decreasing length first and incresing first index second."""
  return span[1] - span[0], -span[0]


def _get_numeric_value_from_float(value):
  return interaction_pb2.NumericValue(float_value=value)


# Doesn't parse ordinal expressions such as '18th of february 1655'.
def _parse_date(text):
  """Attempts to format a text as a standard date string (yyyy-mm-dd)."""
  text = re.sub(r'Sept\b', 'Sep', text)
  for in_pattern, mask, regex in _PROCESSED_DATE_PATTERNS:
    if not regex.match(text):
      continue
    try:
      date = datetime.datetime.strptime(text, in_pattern).date()
    except ValueError:
      continue
    try:
      return _get_numeric_value_from_date(date, mask)
    except ValueError:
      continue
  return None


def _parse_number(text):
  """Parses simple cardinal and ordinals numbers."""
  for suffix in _ORDINAL_SUFFIXES:
    if text.endswith(suffix):
      text = text[:-len(suffix)]
      break
  text = text.replace(',', '')
  try:
    value = float(text)
  except ValueError:
    return None
  if math.isnan(value):
    return None
  if value == _INF:
    return None
  return value


def parse_text(text):
  """Extracts longest number and date spans.

  Args:
    text: text to annotate.

  Returns:
    List of longest numeric value spans.
  """
  span_dict = collections.defaultdict(list)
  for match in _NUMBER_PATTERN.finditer(text):
    span_text = text[match.start():match.end()]
    number = _parse_number(span_text)
    if number is not None:
      span_dict[match.span()].append(_get_numeric_value_from_float(number))

  for begin_index, end_index in text_utils.get_all_spans(
      text, max_ngram_length=1):
    if (begin_index, end_index) in span_dict:
      continue
    span_text = text[begin_index:end_index]

    number = _parse_number(span_text)
    if number is not None:
      span_dict[begin_index, end_index].append(
          _get_numeric_value_from_float(number))
    for number, word in enumerate(_NUMBER_WORDS):
      if span_text == word:
        span_dict[begin_index, end_index].append(
            _get_numeric_value_from_float(float(number)))
        break
    for number, word in enumerate(_ORDINAL_WORDS):
      if span_text == word:
        span_dict[begin_index, end_index].append(
            _get_numeric_value_from_float(float(number)))
        break

  for begin_index, end_index in text_utils.get_all_spans(
      text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
    span_text = text[begin_index:end_index]
    date = _parse_date(span_text)
    if date is not None:
      span_dict[begin_index, end_index].append(date)

  spans = sorted(
      span_dict.items(),
      key=lambda span_value: _get_span_length_key(span_value[0]),
      reverse=True)
  selected_spans = []
  for span, value in spans:
    for selected_span, _ in selected_spans:
      if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
        break
    else:
      selected_spans.append((span, value))

  selected_spans.sort(key=lambda span_value: span_value[0][0])

  numeric_value_spans = []
  for span, values in selected_spans:
    numeric_value_spans.append(
        interaction_pb2.NumericValueSpan(
            begin_index=span[0], end_index=span[1], values=values))
  return numeric_value_spans


def is_ordinal(text):
  if text in _ORDINAL_WORDS:
    return True
  if not _parse_number(text):
    return False
  for suffix in _ORDINAL_SUFFIXES:
    if text.endswith(suffix):
      return True
  return False
