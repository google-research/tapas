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
"""Helper package to find relevant paragraphs in a website using a query."""

import functools
import threading
from typing import List, Text

import nltk


@functools.lru_cache()
def _load_sentence_tokenizer():
  """Returns a sentence tokenization function."""
  # Lock to avoid a race-condition in the creation of the download directory.
  with threading.Lock():
    nltk.download("punkt")
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def tokenize(document):
  """Split text into sentences."""
  sentence_tokenizer = _load_sentence_tokenizer()
  result = []
  for sentence in sentence_tokenizer.tokenize(document):
    sentence = sentence.strip()
    if sentence:
      result.append(sentence)
  return result
