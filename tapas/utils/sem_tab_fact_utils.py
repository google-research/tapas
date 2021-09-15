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
"""Utilities for reading sem tab fact data."""
import collections
import enum
import os
from typing import Text, Mapping, Iterable, List
from absl import logging
import bs4
import numpy as np
from tapas.protos import interaction_pb2
from tapas.scripts import prediction_utils
import tensorflow.compat.v1 as tf
import tqdm


class Type(enum.Enum):
  refuted = 0
  entailed = 1
  unknown = 2


class Version(enum.Enum):
  """Different ways to convert XML to interpretation protos.

  V1: Add evidence. Header (columns) is always empty.
  V2: Do not add evidence. Header (columns) is set from first row.
  """
  V1 = 1
  V2 = 2


def _format(text):
  text = " ".join(text.split())
  return text


def _to_type_index(type_name):
  return Type[type_name].value


def _index_to_type(index):
  return Type(index).name


def _fill_proto_table(
    doc_id,
    table,
    table_proto,
    version = Version.V2,
):
  """Fills the cells in the proto table."""
  table_proto.table_id = _get_table_id(doc_id, table["id"])

  for index, caption in enumerate(table.find_all("caption")):
    if index > 0:
      raise ValueError("Multiple captions in table.")
    table_proto.document_title = _format(caption["text"])

  cells = collections.defaultdict(dict)

  for row in table.find_all("row"):

    for cell in row.find_all("cell"):
      col_start = int(cell["col-start"])
      col_end = int(cell["col-end"]) + 1
      row_start = int(cell["row-start"])
      row_end = int(cell["row-end"]) + 1

      cell_text = _format(cell["text"])

      # Table cells can potentially cover multiple cells and columns.
      # Here we split such cells into multiple 1x1 cells and put the content in
      # the upper left corner.
      # Input:
      #   A | B | C | D
      #   a | \   / | d
      #   a |   b   | d
      #   a | /   \ | d
      #   a | b | b | d
      # Output:
      #   A | B | C | D
      #   a | b |   | d
      #   a |   |   | d
      #   a |   |   | d
      #   a | b | b | d
      for column_index in range(col_start, col_end):
        for row_index in range(row_start, row_end):
          if column_index in cells[row_index]:
            raise ValueError(f"Conflicting index information in table: {table}")
          cells[row_index][column_index] = cell_text
          cell_text = ""

  for row_index in sorted(cells):
    row_proto = table_proto.rows.add()
    for col_index in sorted(cells[row_index]):
      row_proto.cells.add(text=cells[row_index][col_index])

  if not table_proto.rows:
    raise ValueError("Empty table")

  num_columns = len(table_proto.rows[0].cells)
  for row in table_proto.rows:
    if len(row.cells) != num_columns:
      raise ValueError(f"Ragged table: {table_proto}")

  if version == Version.V1:
    # Add empty header.
    for _ in range(num_columns):
      table_proto.columns.add()
  elif version == Version.V2:
    # Add first row as header.
    for cell in table_proto.rows[0].cells:
      table_proto.columns.add().CopyFrom(cell)
    del table_proto.rows[0]
  else:
    raise ValueError(f"Unsupported version: {version.name}")


def _process_evidence(
    statement_id,
    cell,
    evidence,
    table,
    question,
):
  """Adds relevant evidence to as answer coordinate."""
  version = int(evidence["version"])
  if version > 0:
    logging.info("Ignoring evidence: %s", evidence)
    return
  if version != 0:
    raise ValueError(f"Unexpected version {evidence['version']}")

  if evidence["type"] == "irrelevant":
    return

  if evidence["type"] != "relevant":
    raise ValueError(f"Unknown evidence type: {evidence['type']}")

  if int(evidence["statement_id"]) != statement_id:
    return

  new_coord = question.answer.answer_coordinates.add(
      row_index=int(cell["row-start"]),
      column_index=int(cell["col-start"]),
  )

  if new_coord.row_index >= len(table.rows):
    raise ValueError("Row index error")

  if new_coord.column_index >= len(table.columns):
    raise ValueError("Column index error")

  text = table.rows[new_coord.row_index].cells[new_coord.column_index].text
  if text != _format(cell["text"]):
    raise ValueError("Unexpected cell text")

  question.answer.answer_texts.append(text)


def _process_table(
    doc_id,
    table,
    version = Version.V2,
):
  """Processes a single xml table."""
  interaction = interaction_pb2.Interaction()
  table_proto = interaction.table
  _fill_proto_table(doc_id, table, table_proto, version)

  for statement in table.find_all("statement"):

    new_interaction = interaction_pb2.Interaction()
    new_interaction.CopyFrom(interaction)

    statement_id = int(statement["id"])

    question = new_interaction.questions.add()
    new_interaction.id = _get_interaction_id(table_proto.table_id, statement_id)
    question.id = _get_question_id(new_interaction.id)
    question.original_text = _format(statement["text"])

    type_index = _to_type_index(statement["type"])
    question.answer.class_index = type_index

    if version == Version.V1:
      for row in table.find_all("row"):
        for cell in row.find_all("cell"):
          for evidence in cell.find_all("evidence"):
            _process_evidence(statement_id, cell, evidence, table_proto,
                              question)
    yield new_interaction


def _process_doc(
    filepath,
    version = Version.V2,
):
  """Parses XML into multiple interactions."""

  doc_id = _get_doc_id(filepath)

  with tf.io.gfile.GFile(filepath) as input_file:
    soup = bs4.BeautifulSoup(input_file.read(), "lxml")
    for table in soup.find_all("table"):
      yield from _process_table(doc_id, table, version)


def _process_docs(
    filepaths,
    name,
    version = Version.V2,
):
  for filepath in tqdm.tqdm(filepaths, desc=name):
    yield from _process_doc(filepath, version)


def convert(
    input_dir,
    skip_auto_train = True,
    version = Version.V2,
):
  """Reads dev and train data from task dir."""
  output = collections.OrderedDict()

  corpora = [
      ("dev", "v1.1"),
      ("test", "test_phase_a"),
      ("train", "v1.3.2"),
  ]
  if not skip_auto_train:
    # This set is large so we skip by default.
    corpora.append(("auto_train", "autogenerated_train_v1.3.2"))

  for corpus, dirname in corpora:
    pattern = os.path.join(input_dir, dirname, "output", "*.xml")
    filepaths = tf.io.gfile.glob(pattern)
    output[corpus] = _process_docs(
        filepaths,
        name=corpus,
        version=version,
    )
  return output


def _get_doc_id(filepath):
  return os.path.splitext(os.path.basename(filepath))[0]


def _get_table_id(doc_id, table_id):
  table_id = table_id.replace(" ", "_")
  return f"{doc_id}_{table_id}"


def _get_interaction_id(table_id, statement_id):
  return f"{table_id}_{statement_id}"


def _get_question_id(interaction_id):
  return f"{interaction_id}_0"


def _get_majority_label(prediction_list):
  counter = collections.Counter(
      np.argmax(prediction) for prediction in prediction_list)
  return counter.most_common(1)[0][0]


def create_outputs(
    input_dir,
    prediction_files,
    output_dir,
):
  """Iterates over XML files in input dir and adds types to statements."""
  prediction_dict = collections.defaultdict(list)
  for prediction_file in prediction_files:
    for row in prediction_utils.iterate_predictions(prediction_file):
      prediction_dict[row["question_id"]].append(row["logits_cls"])

  for filename in tf.io.gfile.listdir(input_dir):
    if not filename.endswith("xml"):
      continue

    filepath = os.path.join(input_dir, filename)
    doc_id = _get_doc_id(filepath)

    with tf.io.gfile.GFile(filepath) as input_file:
      soup = bs4.BeautifulSoup(input_file.read(), "lxml")
      for table in soup.find_all("table"):
        table_id = _get_table_id(doc_id, table["id"])
        for statement in table.find_all("statement"):
          interaction_id = _get_interaction_id(
              table_id,
              int(statement["id"]),
          )
          question_id = _get_question_id(interaction_id)
          prediction_list = prediction_dict[question_id]
          index = _get_majority_label(prediction_list)
          statement["type"] = _index_to_type(index)

    tf.io.gfile.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    with tf.io.gfile.GFile(output_path, "w") as output_file:
      output_file.write(str(soup))
