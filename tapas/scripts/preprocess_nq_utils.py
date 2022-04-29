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
"""Natural Questions preprocessing utils for table retrieval."""

import collections
import enum
import hashlib
import html
import json
import math
import os
import re
from typing import List, Mapping, Text, Tuple, Any, Iterable, Optional, Set, Dict
import urllib.parse

from absl import logging
import apache_beam as beam
import bs4
import dataclasses
import numpy as np
import pandas as pd
from tapas.protos import interaction_pb2
import tensorflow.compat.v1 as tf

_NS = "main"

_SpanT = Tuple[int, int]


class Split(enum.Enum):
  train = 0
  dev = 1


@dataclasses.dataclass
class Table:
  header: List[Text]
  rows: List[List[Text]]


def _str(value):
  if isinstance(value, str):
    return value
  beam.metrics.Metrics.counter(_NS,
                               f"Dataframe cell types: {type(value)}").inc()
  if isinstance(value, tuple):
    logging.warning("Dataframe cell tuple: %s", value)
    return " ".join(_str(item) for item in value)
  return str(value)


def _preprocess_html(table_html):
  """Parses HTML with bs4 and fixes some glitches."""
  table_html = table_html.replace("<br />", "<br /> ")
  table = bs4.BeautifulSoup(table_html, "html5lib")
  table = table.find("table")
  # Delete hidden style annotations.
  for tag in table.find_all(attrs={"style": "display:none"}):
    tag.decompose()
  # Make sure "rowspan" is not set to an illegal value.
  for tag in table.find_all("td"):
    for attr in list(tag.attrs):
      if attr == "rowspan":
        tag.attrs[attr] = ""
  return table


def _strip_tags(html_text):
  """We turn the text into a table to make sure we parse just like in tables."""
  try:
    html_table = f"""
      <table><tbody><tr><td>{html_text}</td></tr></tbody></table>
    """
    table_df = _read_html(str(_preprocess_html(html_table)))
    return _str(table_df[0][0])
  except (ValueError, IndexError, KeyError) as e:
    logging.warning('Failed in parsing answer HTML: "%s" "%s"', e, html_text)
    beam.metrics.Metrics.counter(_NS, "Failed parsing answer HTML").inc()
    return html_text


def get_filenames(path, split):
  """Reads NQ files by yielding each line of serialized json."""
  num_files = 50 if split == Split.train else 5
  for i in range(num_files):
    split_name = split.name
    filepath = os.path.join(path, split_name, f"nq-{split_name}-{i:02d}.jsonl")
    yield filepath + ".gz"


def _is_non_standard_table(table):
  """Filters non standard tables (those that are not a simple rectangular)."""
  headers = list(table.columns)
  # Check whether there is one row of headers.
  all_strings = all([isinstance(header, str) for header in headers])
  # check wheather there are headers (for some tables there are no headers).
  is_headers = headers != list(range(len(headers)))

  # Check whether more than half of the rows have only one cell (should filter
  # tables like en.wikipedia.org//w/index.php?title=List_of_Maid_Sama!_episodes
  values = table.values
  is_too_many_rows_eq = sum([len(set(row)) == 1 for row in values
                            ]) > (values.shape[0] / 2) - 1
  return not all([all_strings, is_headers, not is_too_many_rows_eq])


def _parse_infobox_table(table):
  """Parses an infobox table to a transposed single row horizontal table."""
  table_dict_intermediate = table.transpose().to_dict()
  table_dict = Table(header=[], rows=[])
  single_row = []

  def _append(header, value):
    table_dict.header.append(_str(header))
    single_row.append(_str(value))

  for column in table_dict_intermediate.values():
    if not single_row:
      # Add first key as additional value. Usually that's the entity name.
      keys = [_str(k) for k in column.keys()]
      _append(header="", value=keys[0])
    values = [_str(v) for v in column.values()]
    # Check whether these are standard header and cell.
    if len(values) < 2:
      _append(header="", value=values[0])
    elif len(values) == 2 and values[0] == values[1]:
      _append(header="", value=values[0])
    else:
      _append(header=values[0], value=" ".join(values[1:]))
  table_dict.rows.append(single_row)
  return table_dict if table_dict.header else None


def _parse_horizontal_table(table):
  """Parses a valid horizontal table."""
  if _is_non_standard_table(table):
    return None
  table_dict_intermediate = table.to_dict()
  # table_dict = {"header": [], "rows": []}
  table_dict = Table(header=[], rows=[])
  for header in table_dict_intermediate:
    table_dict.header.append(_str(header))
    rows = table_dict.rows
    for row_index in table_dict_intermediate[header]:
      if len(rows) == row_index:  # add new row.
        rows.append([])
      rows[row_index].append(_str(table_dict_intermediate[header][row_index]))
  return table_dict


def _read_html(table):
  """Parse text into data frame."""
  decimal_point = "@"
  if decimal_point in table:
    logging.info('%s in table: """%s"""', decimal_point, table)
    beam.metrics.Metrics.counter(_NS, "Decimal point in table").inc()
  return pd.read_html(
      table,
      flavor="html5lib",
      thousands=decimal_point,
      decimal=decimal_point,
      na_values=[],
      keep_default_na=False,
      displayed_only=False,
  )[0]


def _parse_table(table_html):
  """Parses an HTML table to a dictionary of headers and rows."""
  table = _preprocess_html(table_html.decode("utf-8"))
  try:
    table_df = _read_html(str(table))
  except (ValueError, IndexError) as e:
    logging.warning('Failed in parsing a table: "%s" """%s"""', e, table_html)
    beam.metrics.Metrics.counter(_NS, "Failed parsing table HTML").inc()
    return None
  is_table_infobox = "infobox" in table.attrs[
      "class"] if "class" in table.attrs else False
  if is_table_infobox:
    table_dict = _parse_infobox_table(table_df)
  else:
    table_dict = _parse_horizontal_table(table_df)
  return table_dict


def _format_text(text):
  """Lowercases and normalizes spaces."""
  text = text.lower()
  text = " ".join(text.split())
  text = text.strip()
  return text


def _get_table_fingerprint(table):
  table_texts = []
  table_texts.extend(table.header)
  for row in table.rows:
    table_texts.extend(row)
  table_texts = [_format_text(text) for text in table_texts]
  hash_number = _to_hash("###".join(table_texts).lower())
  return hex(hash_number % _MAX_INT)[2:].upper()


def _get_span_to_table(
    document_title,
    document_url,
    table_spans,
    tables,
):
  """Builds a mapping between a table span and its content."""
  if len(table_spans) != len(tables):
    raise ValueError("Number of table spans and tables do not match.")
  span_to_table = {}
  for span, table in zip(table_spans, tables):
    table_dict = _parse_table(table)
    if table_dict is None:
      continue
    fp = _get_table_fingerprint(table_dict)
    table_id = f"{document_title}_{fp}"
    table = _get_table_proto(
        table_id,
        document_title,
        document_url,
        table_dict,
    )
    span_to_table[span] = table
  return span_to_table


def _get_table_proto(
    table_id,
    document_title,
    document_url,
    table_dict,
):
  """Converts a table dictionary to a Table proto."""
  table = interaction_pb2.Table()
  table.table_id = table_id
  table.document_title = document_title
  table.document_url = document_url

  for column in table_dict.header:
    table.columns.add().text = column

  for row in table_dict.rows:
    new_row = table.rows.add()
    for cell in row:
      new_row.cells.add().text = cell
  return table


def _matches(answer, cell):
  return answer in cell


def find_answer(
    table,
    answer,
    search_in_header = True,
):
  """Locates text answer in table cells."""
  if search_in_header:
    for col_index, column in enumerate(table.columns):
      cell_text = column.text
      if _matches(answer, cell_text):
        yield -1, col_index
  for row_index, row in enumerate(table.rows):
    for col_index, cell in enumerate(row.cells):
      cell_text = cell.text
      if _matches(answer, cell_text):
        yield row_index, col_index


def get_interaction_id(example_id, answer_index):
  return f"{example_id}_{answer_index}"


def parse_interaction_id(interaction_id):
  index = interaction_id.rfind("_")
  if index < 0:
    raise ValueError(f"Invalid interaction id: {interaction_id}")
  example_id = interaction_id[:index]
  answer_index = interaction_id[index + 1:]
  answer_index = int(answer_index)
  return example_id, answer_index


def _is_valid_answer_text(answer_text):
  return bool(answer_text) and answer_text != "-"


def table_contains_all_answers(table,
                               question):
  """Checks if a table contains all the answers.

  Args:
    table: the table to check.
    question: containing the answers to check.

  Returns:
    False if no answer to check from the question.
    False if any of the answers text and alternative answers tests is not found.
    A answer text is considered as non existing if the field is_valid == False.
    True if all the answers are found.
  """

  def _check_one_answer(answer):
    if answer.HasField("is_valid") and not answer.is_valid:
      return False
    if not answer.answer_texts:
      return False
    for answer_text in answer.answer_texts:
      coords = list(find_answer(table, answer_text))
      if not coords:
        return False
    return True

  # Checks if there is at least an answer to check.
  if not question.HasField("answer"):
    return False
  # If no alternative_answers is given then consider it as True.
  alternative_answers = [
      _check_one_answer(answer) for answer in question.alternative_answers
  ]
  if not alternative_answers:
    found_answers = False
  else:
    found_answers = any(alternative_answers)
  return found_answers or _check_one_answer(question.answer)


def _create_interaction(
    example_id,
    question_text,
    answer_groups,
    table,
):
  """Creates an interaction instance that corresponds to some short answer."""
  interaction = interaction_pb2.Interaction()
  question = interaction.questions.add()

  main_answer_index = None
  for answer_group_index, answers in answer_groups:
    if answers:
      all_answer_valid = True
      for answer_text in answers:
        if not _is_valid_answer_text(answer_text):
          all_answer_valid = False
          break
        coords = list(find_answer(table, answer_text))
        if not coords:
          all_answer_valid = False
          break
      if all_answer_valid:
        if main_answer_index is None:
          answer = question.answer
          main_answer_index = answer_group_index
        else:
          answer = question.alternative_answers.add()
        for answer_text in answers:
          answer.answer_texts.append(answer_text)
  if main_answer_index is None:
    beam.metrics.Metrics.counter(_NS, "Failed to find answer in table").inc()
    return None

  interaction_id = get_interaction_id(example_id, main_answer_index)
  interaction.id = interaction_id
  question.id = f"{interaction.id}_0"
  question.original_text = question_text
  interaction.table.CopyFrom(table)
  return interaction


def _extract_interaction(
    annotations_spans,
    span_to_table,
    answers,
    example_id,
    question_text,
):
  """Greedily extracts the first interaction that resides in a valid table."""
  for (table_start, table_end), table in span_to_table.items():
    if table is None:
      continue
    answer_texts = []
    for answer_index, (spans,
                       texts) in enumerate(zip(annotations_spans, answers)):
      matches = [(start, end)
                 for (start, end) in spans
                 if table_start <= start and end <= table_end]
      if len(matches) == len(spans):
        answer_texts.append((answer_index, texts))
    if answer_texts:
      interaction = _create_interaction(
          example_id,
          question_text,
          answer_groups=answer_texts,
          table=table,
      )
      if interaction is not None:
        yield interaction


def _get_all_spans(
    start,
    end,
    html_bytes,
):
  """Get all spans for a given label."""
  boundaries = [(m.start(), True) for m in start.finditer(html_bytes)]
  boundaries += [(m.end(), False) for m in end.finditer(html_bytes)]
  boundaries.sort()

  start_indexes = []
  spans = []
  for index, is_start in boundaries:
    if is_start:
      start_indexes.append(index)
    else:
      if not start_indexes:
        logging.warning("Invalid HTML string: %s", html_bytes)
      else:
        start_index = start_indexes.pop()
        spans.append((start_index, index))

  if start_indexes:
    logging.warning("Invalid HTML string: %s", html_bytes)
  spans.sort()
  return spans


def _get_spans(
    html_bytes, annotations
):
  """Extracts spans for all answers, table cells and tables."""
  table_start = re.compile(b"<table", re.I)
  table_end = re.compile(b"</table>", re.I)

  annotations_spans = []  # spans for all annotations.
  for annot in annotations:
    answers = [
        (ans["start_byte"], ans["end_byte"]) for ans in annot["short_answers"]
    ]
    if not answers:
      beam.metrics.Metrics.counter(
          _NS, "Removing annotation without short answers").inc()
      continue
    elif len(answers) != 1:
      beam.metrics.Metrics.counter(
          _NS, "Annotations with multiple short answers").inc()
    annotations_spans.append(answers)
  table_spans = _get_all_spans(table_start, table_end, html_bytes)
  return annotations_spans, table_spans


def parse(line,):
  """Parses example and extracts an interaction if any answer is in a table.

  Args:
    line: serialized NQ json dict.

  Returns:
    Mapping with interactions and tables.
  """
  sample = json.loads(line)
  doc_html = sample["document_html"]
  example_id = sample["example_id"]
  question_text = sample["question_text"]
  document_title = sample["document_title"]
  document_url = sample["document_url"]
  html_bytes = doc_html.encode()

  annotations_spans, table_spans = _get_spans(html_bytes, sample["annotations"])

  # Stop if there are no tables in the document, or there are no short answers.
  if not table_spans or not annotations_spans:
    if not table_spans:
      beam.metrics.Metrics.counter(_NS, "Examples without tables").inc()
    if not annotations_spans:
      beam.metrics.Metrics.counter(_NS, "Examples without short answers").inc()
    return {
        "example_id": example_id,
        "contained": False,
        "tables": [],
        "interactions": [],
    }

  span_to_table = _get_span_to_table(
      document_title,
      document_url,
      table_spans,
      tables=[html_bytes[begin:end] for begin, end in table_spans],
  )

  answer_texts = []
  for answers in annotations_spans:
    answer_texts.append([
        _strip_tags(html_bytes[ans_start:ans_end].decode("utf-8"))
        for ans_start, ans_end in answers
    ])
  interactions = list(
      _extract_interaction(
          annotations_spans,
          span_to_table,
          answer_texts,
          example_id,
          question_text,
      ))
  return {
      "example_id": example_id,
      "contained": bool(interactions),
      "tables": [table for table in span_to_table.values() if table],
      "interactions": interactions,
  }


_MAX_INT = 2**64 - 1


def _to_hash(text):
  return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def _to_random_float(text):
  """Converts an object to a random float between 0 and 1."""
  hash_index = _to_hash(text)
  return (hash_index % _MAX_INT) / _MAX_INT


def write_interactions_to_files(
    tables,
    interactions,
    output_path,
):
  """Write interactions to tf records. Replaces tables."""

  table_dict = {}
  for table in tables:
    table_dict[table.table_id] = table
    for table_id in table.alternative_table_ids:
      table_dict[table_id] = table

  # Split them between train/dev/test sets.
  interaction_dir = os.path.join(output_path, "interactions")
  tf.io.gfile.makedirs(interaction_dir)

  stats = collections.defaultdict(int)

  def tfrecord(name):
    return tf.io.TFRecordWriter(
        os.path.join(interaction_dir, f"{name}.tfrecord"))

  with tfrecord("train") as train_writer, tfrecord(
      "dev") as dev_writer, tfrecord("test") as test_writer:
    for interaction in interactions:
      stats["total"] += 1
      if not table_contains_all_answers(interaction.table,
                                        interaction.questions[0]):
        stats["num_missing_answer"] += 1
        continue
      table_id = interaction.table.table_id
      interaction.table.CopyFrom(table_dict[table_id])
      if table_id != interaction.table.table_id:
        # The table was merged and changed.
        stats["num_new_table_id"] += 1
      del table_id
      if not table_contains_all_answers(interaction.table,
                                        interaction.questions[0]):
        stats["num_missing_answer_after_merge"] += 1
        continue
      if interaction.id.startswith(Split.train.name):
        rand = _to_random_float(interaction.table.document_title)
        if rand < 0.9:
          train_writer.write(interaction.SerializeToString())
          stats["train_size"] += 1
        else:
          dev_writer.write(interaction.SerializeToString())
          stats["dev_size"] += 1
      elif interaction.id.startswith(Split.dev.name):
        test_writer.write(interaction.SerializeToString())
        stats["test_size"] += 1
      else:
        raise ValueError(f"Invalid id: {interaction.id}")
  return stats


def get_version(table):
  query = urllib.parse.urlparse(html.unescape(table.document_url)).query
  parsed_query = urllib.parse.parse_qs(query)
  values = parsed_query["oldid"]
  value = values[0]
  return int(value)


def _clusters_are_compatible(
    clustering,
    cluster_id,
    other_cluster_id,
):
  for table in clustering[cluster_id]:
    for other_table in clustering[other_cluster_id]:
      if not _tables_are_compatible(table, other_table):
        return False
  return True


def _get_document_urls(table):
  versions = set(table.alternative_document_urls)
  versions.add(table.document_url)
  return versions


def _tables_are_compatible(
    table,
    other_table,
):
  """Checks some simple heuristics to improve quality and efficiency."""
  if _get_document_urls(table) & _get_document_urls(other_table):
    # Tables occur on the same version of the page.
    return False
  if math.fabs(len(table.rows) - len(other_table.rows)) > 2:
    return False
  if len(table.columns) != len(other_table.columns):
    return False
  return True


def _get_table_text(table,
                    header_weight):
  for _ in range(header_weight):
    yield [column.text for column in table.columns]
  for row in table.rows:
    yield [cell.text for cell in row.cells]


_SPECIAL_CHAR = re.compile(r"\W")


def _tokenize_text(text):
  text = _SPECIAL_CHAR.sub(" ", text)
  return text.lower().split()


def _get_ngrams(
    table,
    header_weight,
    max_ngram_length,
):
  """Serializes table and returns token ngrams up to a maximum length."""
  for row in _get_table_text(table, header_weight=header_weight):
    tokens = []
    for text in row:
      tokens.extend(_tokenize_text(text))
    for index in range(len(tokens)):
      for length in range(1, min(max_ngram_length, len(tokens) - index) + 1):
        ngram = tokens[index:index + length]
        yield " ".join(ngram)


class TableSimilarity:
  """Helper class for computing similarity between tables."""

  def __init__(
      self,
      header_weight,
      ngram_length,
      hash_dimension,
  ):
    self._cache = {}
    self._header_weight = header_weight
    self._ngram_length = ngram_length
    self._hash_dimension = hash_dimension

  def get_similarity(
      self,
      table,
      other_table,
  ):
    if not _tables_are_compatible(table, other_table):
      return 0.0
    return self._get_embedding(table).dot(self._get_embedding(other_table))

  def _get_embedding(
      self,
      table,
  ):
    """Computes hashed normalized ngram vector."""
    embedding = self._cache.get(table.table_id)
    if embedding is None:
      embedding = np.zeros(self._hash_dimension)
      for ngram in _get_ngrams(
          table,
          header_weight=self._header_weight,
          max_ngram_length=self._ngram_length,
      ):
        embedding[_to_hash(ngram) % self._hash_dimension] += 1
      # Apply l2-norm.
      embedding /= math.sqrt(embedding.dot(embedding))
      self._cache[table.table_id] = embedding
    return embedding


def group_similar_tables(
    tables,
    header_weight = 1,
    ngram_length = 4,
    threshold = 0.91,
    hash_dimension = 2048,
):
  """Computes single link clustering based on ngram similarity."""
  table_sim = TableSimilarity(
      header_weight=header_weight,
      ngram_length=ngram_length,
      hash_dimension=hash_dimension,
  )
  clustering = collections.defaultdict(list)
  for table in tables:
    clustering[table.table_id].append(table)
  similarities = []
  for table_id in clustering:
    table = clustering[table_id][0]
    for other_table_id in clustering:
      if table_id >= other_table_id:
        continue
      other_table = clustering[other_table_id][0]
      sim = table_sim.get_similarity(
          table,
          other_table,
      )
      if sim >= threshold:
        similarities.append((table.table_id, other_table.table_id, sim))
  similarities.sort(key=lambda item: item[-1], reverse=True)
  new_cluster_id = {}
  for a, b, _ in similarities:
    # Find the new cluster ids of a and b.
    while a in new_cluster_id:
      a = new_cluster_id[a]
    while b in new_cluster_id:
      b = new_cluster_id[b]
    if a == b:
      # a and b are already in the same cluster.
      continue
    if not _clusters_are_compatible(clustering, a, b):
      continue
    clustering[a].extend(clustering[b])
    del clustering[b]
    new_cluster_id[b] = a
  return list(clustering.values())
