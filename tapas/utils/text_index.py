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

import dataclasses
import enum
import heapq
from typing import Callable, Generic, Sequence, Text, Tuple, TypeVar

from sklearn.feature_extraction.text import TfidfVectorizer

T = TypeVar('T')


class Analyzer(enum.Enum):
  """Helper enum to determine how to compute n-grams to build the TextIndex.

  Option ‘char_wb’ creates character n-grams only from text inside word
  boundaries; n-grams at the edges of words are padded with space.
  """
  WORD = 'word'
  CHAR = 'char'
  CHAR_WB = 'char_wb'


@dataclasses.dataclass
class SearchResult(Generic[T]):
  document: T
  score: float


class TextIndex(Generic[T]):
  """A simple text index from a corpus of text using tf-idf similarity."""

  def __init__(self,
               documents,
               text_getter,
               ngram_range = (1, 2),
               analyzer = Analyzer.WORD,
               min_df = 1,
               max_df = .9):
    """Init parameters for TextIndex.

    Args:
      documents: Corpus of documents to be indexed and retrieved.
      text_getter: Function to extract text from documents.
      ngram_range: tuple (min_n, max_n), default=(1, 2) The lower and upper
        boundary of the range of n-values for different n-grams to be extracted.
        All values of n such that min_n <= n <= max_n will be used. For example
        an ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
      analyzer: Analyzer, {‘word’, ‘char’, ‘char_wb’}. Whether the
        feature should be made of word or character n-grams. Option
        ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
      min_df: float in range [0.0, 1.0] or int (default=1) When building the
        vocabulary ignore terms that have a document frequency strictly lower
        than the given threshold. This value is also called cut-off in the
        literature. If float, the parameter represents a proportion of
        documents, integer absolute counts.
      max_df: float in range [0.0, 1.0] or int (default=0.9) When building the
        vocabulary ignore terms that have a document frequency strictly higher
        than the given threshold (corpus-specific stop words). If float, the
        parameter represents a proportion of documents, integer absolute counts.
    """
    self._vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        analyzer=analyzer.value)

    self._documents = documents
    self._index = self._vectorizer.fit_transform(
        map(text_getter, self._documents))

  def search(self,
             query,
             retrieval_threshold = 0.0,
             num_results = 5):
    """Retrieve matching text in the corpus.

    Args:
      query: Text used to search for candidates in the corpus.s
      retrieval_threshold: Filter results above this threshold.
      num_results: Number of results to return.

    Returns:
      Tuple of text and float score. Top `num_results` elements in the corpus.
    """
    query_vector = self._vectorizer.transform([query])
    scores = zip(self._documents,
                 self._index.dot(query_vector.T).T.toarray()[0])
    filtered_scores = (SearchResult(doc, score)
                       for doc, score in scores
                       if score > retrieval_threshold)
    return heapq.nlargest(num_results, filtered_scores, key=lambda p: p.score)
