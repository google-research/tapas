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
# coding=utf8
"""Includes utility functions to convert and normalize text."""

import math
import re
import struct
from typing import Iterable, List, Text, Tuple
import unicodedata

import six
from tapas.utils import constants


def wtq_normalize(x):
  """Returns the normalized version of x.

  This normalization function is taken from WikiTableQuestions github, hence the
  wtq prefix. For more information, see
  https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py

  Args:
    x: the object (integer type or string) to normalize.

  Returns:
    A normalized string.
  """
  x = x if isinstance(x, six.text_type) else six.text_type(x)
  # Remove diacritics.
  x = "".join(
      c for c in unicodedata.normalize("NFKD", x)
      if unicodedata.category(c) != "Mn")
  # Normalize quotes and dashes.
  x = re.sub(u"[‘’´`]", "'", x)
  x = re.sub(u"[“”]", '"', x)
  x = re.sub(u"[‐‑‒–—−]", "-", x)
  x = re.sub(u"[‐]", "", x)
  while True:
    old_x = x
    # Remove citations.
    x = re.sub(u"((?<!^)\\[[^\\]]*\\]|\\[\\d+\\]|[•♦†‡*#+])*$", "",
               x.strip())
    # Remove details in parenthesis.
    x = re.sub(u"(?<!^)( \\([^)]*\\))*$", "", x.strip())
    # Remove outermost quotation mark.
    x = re.sub(u'^"([^"]*)"$', r"\1", x.strip())
    if x == old_x:
      break
  # Remove final '.'.
  if x and x[-1] == ".":
    x = x[:-1]
  # Collapse whitespaces and convert to lower case.
  x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
  x = re.sub("<[^<]+?>", "", x)
  x = x.replace("\n", " ")
  return x


_TOKENIZER = re.compile(r"\w+|[^\w\s]+", re.UNICODE)


def tokenize_string(x):
  return list(_TOKENIZER.findall(x.lower()))


# List of string normalization functions to be applied in order. We go from
# simplest to more complex normalization procedures.
STRING_NORMALIZATIONS = (
    lambda x: x,
    lambda x: x.lower(),
    tokenize_string,
    wtq_normalize,
)


def _split_thousands(delimiter, value):
  split = value.split(delimiter)
  return len(split) > 1 and any(map(lambda x: len(x) == 3, split))


def convert_to_float(value):
  """Converts value to a float using a series of increasingly complex heuristics.

  Args:
    value: object that needs to be converted. Allowed types include
      float/int/strings.

  Returns:
    A float interpretation of value.

  Raises:
    ValueError if the float conversion of value fails.
  """
  if isinstance(value, float):
    return value
  if isinstance(value, int):
    return float(value)
  if not isinstance(value, six.string_types):
    raise ValueError("Argument value is not a string. Can't parse it as float")
  sanitized = value

  try:
    # Example: 1,000.7
    if "." in sanitized and "," in sanitized:
      return float(sanitized.replace(",", ""))
    # 1,000
    if "," in sanitized and _split_thousands(",", sanitized):
      return float(sanitized.replace(",", ""))
    # 5,5556
    if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
        ",", sanitized):
      return float(sanitized.replace(",", "."))
    # 0.0.0.1
    if sanitized.count(".") > 1:
      return float(sanitized.replace(".", ""))
    # 0,0,0,1
    if sanitized.count(",") > 1:
      return float(sanitized.replace(",", ""))
    return float(sanitized)
  except ValueError:
    # Avoid adding the sanitized value in the error message.
    raise ValueError("Unable to convert value to float")


def _normalize_float(answer):
  if answer is None:
    return None
  try:
    value = convert_to_float(answer)
    if isinstance(value, float) and math.isnan(value):
      return None
    return value
  except ValueError:
    return answer.lower()


def normalize_answers(answers):
  normalized_answers = (_normalize_float(a) for a in answers)
  normalized_answers = (a for a in normalized_answers if a is not None)
  normalized_answers = (str(a) for a in normalized_answers)
  normalized_answers = list(normalized_answers)
  normalized_answers.sort()
  return normalized_answers


def get_all_spans(text,
                  max_ngram_length):
  """Split a text into all possible ngrams up to 'max_ngram_length'.

  Split points are white space and punctuation.

  Args:
    text: Text to split.
    max_ngram_length: maximal ngram length.

  Yields:
    Spans, tuples of begin-end index.
  """
  start_indexes = []
  for index, char in enumerate(text):
    if not char.isalnum():
      continue
    if index == 0 or not text[index - 1].isalnum():
      start_indexes.append(index)
    if index + 1 == len(text) or not text[index + 1].isalnum():
      for start_index in start_indexes[-max_ngram_length:]:
        yield start_index, index + 1


def normalize_for_match(text):
  return " ".join(text.lower().split())


def format_text(text):
  """Lowercases and strips punctuation."""
  text = text.lower().strip()
  if text == "n/a" or text == "?":
    text = constants.EMPTY_TEXT

  text = re.sub(r"[^\w\d]+", " ", text).replace("_", " ")
  text = " ".join(text.split())
  text = text.strip()
  if text:
    return text
  return constants.EMPTY_TEXT


def tokenize_text(text):
  """Splits tokens by white-space."""
  return text.split()


def format_and_tokenize_text(text):
  """Runs format_text and tokenizes."""
  return tokenize_text(format_text(text))


DEFAULT_INTS_LENGTH = 64


def str_to_ints(text, length):
  ints = [ord(char) + 1 for char in text]
  if len(ints) > length:
    raise ValueError(f"{text} too long: {len(text)} > {length}")
  while len(ints) < length:
    ints.append(0)
  return ints


def ints_to_str(ints):
  return "".join(chr(i - 1) for i in ints if i > 0)


def get_sequence_id(example_id, annotator):
  if "-" in annotator:
    raise ValueError('"-" not allowed in annotator.')
  return f"{example_id}-{annotator}"


def get_question_id(sequence_id, position):
  return f"{sequence_id}_{position}"


def _split_compound_string(text, separator):
  return tuple(text.rsplit(separator, 1))


def parse_question_id(question_id):
  id_annotator, position = _split_compound_string(question_id, "_")
  example_id, annotator = _split_compound_string(id_annotator, "-")
  return example_id, annotator, int(position)


def get_padded_question_id():
  """"Returns the id that marks examples added for TPU padding."""
  return get_question_id(get_sequence_id(constants.EMPTY_TEXT, "0"), 0)


def to_float32(v):
  """If v is a float reduce precision to that of a 32 bit float."""
  if not isinstance(v, float):
    return v
  return struct.unpack("!f", struct.pack("!f", v))[0]


def find_all_substrings(needle,
                        haystack):
  """Get all possible spans of substring allowing full tokens only."""
  # If needle starts or end with non word character, just \b won't work
  regexp = r"(?:\b|\s)(" + re.escape(needle) + r")(?:\s|\b)"
  return [m.span(1) for m in re.finditer(regexp, haystack)]
