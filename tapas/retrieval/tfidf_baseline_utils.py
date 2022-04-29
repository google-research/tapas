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
"""Impl. of a simple TF-IDF baseline for table retrieval."""

import collections
import math
from typing import Text, Iterable, List, Tuple, Mapping, Optional

from absl import logging
import dataclasses
from gensim.summarization import bm25
from tapas.protos import interaction_pb2
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf
import tqdm


def iterate_tables(table_file):
  for value in tf.python_io.tf_record_iterator(table_file):
    table = interaction_pb2.Table()
    table.ParseFromString(value)
    yield table


def iterate_interaction_tables(
    interaction_file):
  for value in tf.python_io.tf_record_iterator(interaction_file):
    interaction = interaction_pb2.Interaction()
    interaction.ParseFromString(value)
    yield interaction.table


def _iterate_table_texts(table,
                         title_multiplicator):
  for _ in range(title_multiplicator):
    if table.document_title:
      yield table.document_title
    for column in table.columns:
      yield column.text
  for row in table.rows:
    for cell in row.cells:
      yield cell.text


def _iterate_tokenized_table_texts(table,
                                   title_multiplicator):
  for text in _iterate_table_texts(table, title_multiplicator):
    yield from _tokenize(text)


def _tokenize(text):
  return text_utils.tokenize_text(text_utils.format_text(text))


@dataclasses.dataclass(frozen=True)
class TableFrequency:
  table_index: int
  score: float


@dataclasses.dataclass(frozen=True)
class IndexEntry:
  table_counts: List[TableFrequency]


class InvertedIndex:
  """Inverted Index implementation."""

  def __init__(self, table_ids, index):
    self.table_ids_ = table_ids
    self.index_ = index

  def retrieve(self, question):
    """Retrieves tables sorted by descending score."""
    hits = collections.defaultdict(list)

    num_tokens = 0
    for token in _tokenize(question):
      num_tokens += 1
      index_entry = self.index_.get(token, None)
      if index_entry is None:
        continue

      for table_count in index_entry.table_counts:
        scores = hits[table_count.table_index]
        scores.append(table_count.score)

    scored_hits = []
    for table_index, inv_document_freqs in hits.items():
      score = sum(inv_document_freqs) / num_tokens
      scored_hits.append((self.table_ids_[table_index], score))

    scored_hits.sort(key=lambda name_score: name_score[1], reverse=True)
    return scored_hits


def _remove_duplicates(
    tables):
  table_id_set = set()
  for table in tables:
    if table.table_id in table_id_set:
      logging.info('Duplicate table ids: %s', table.table_id)
      continue
    table_id_set.add(table.table_id)
    yield table


def create_inverted_index(tables,
                          title_multiplicator=1,
                          min_rank = 0,
                          drop_term_frequency = True):
  """Creates an index for some tables.

  Args:
    tables: Tables to index
    title_multiplicator: Emphasize words in title or header.
    min_rank: Word types with a frequency rank lower than this will be ignored.
      Can be useful to remove stop words.
    drop_term_frequency: Don't consider term frequency.

  Returns:
    the inverted index.
  """
  table_ids = []
  token_to_info = collections.defaultdict(lambda: collections.defaultdict(int))
  for table in _remove_duplicates(tables):
    table_index = len(table_ids)
    table_ids.append(table.table_id)
    for token in _iterate_tokenized_table_texts(table, title_multiplicator):
      token_to_info[token][table_index] += 1

  logging.info('Table Ids: %d', len(table_ids))
  logging.info('Num types: %d', len(token_to_info))

  def count_fn(table_counts):
    return sum(table_counts.values())

  token_to_info = list(token_to_info.items())
  token_to_info.sort(
      key=lambda token_info: count_fn(token_info[1]), reverse=True)

  index = {}
  for freq_rank, (token, table_counts) in enumerate(token_to_info):
    df = count_fn(table_counts)
    if freq_rank < min_rank:
      logging.info(
          'Filter "%s" for index (%d, rank: %d).', token, df, freq_rank)
      continue
    idf = 1.0 / (math.log(df, 2) + 1)
    counts = []
    for table, count in table_counts.items():
      if drop_term_frequency:
        count = 1.0
      counts.append(TableFrequency(table, idf * count))
    index[token] = IndexEntry(counts)

  return InvertedIndex(table_ids, index)


class BM25Index:
  """Index based on gensim BM25."""

  def __init__(self, corpus, table_ids):
    self._table_ids = table_ids
    self._model = bm25.BM25(corpus)

  def retrieve(self, question):
    q_tokens = _tokenize(question)
    # TODO This is slow maybe we can improve efficiency.
    scores = self._model.get_scores(q_tokens)
    table_scores = [(self._table_ids[index], score)
                    for index, score in enumerate(scores)
                    if score > 0.0]
    table_scores.sort(key=lambda table_score: table_score[1], reverse=True)
    return table_scores


def create_bm25_index(
    tables,
    title_multiplicator=1,
    num_tables = None,
):
  """Creates a new index."""
  corpus = []
  table_ids = []
  for table in tqdm.tqdm(_remove_duplicates(tables), total=num_tables):
    corpus.append(
        list(_iterate_tokenized_table_texts(table, title_multiplicator)))
    table_ids.append(table.table_id)
  return BM25Index(corpus, table_ids)
