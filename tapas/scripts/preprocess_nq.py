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
"""Natural Questions preprocessing for table retrieval.

"""

import os
from typing import Any, Iterable, Mapping, Sequence, Text, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from tapas.protos import interaction_pb2
from tapas.scripts import preprocess_nq_utils
from tapas.utils import beam_runner
import tensorflow.compat.v1 as tf


flags.DEFINE_string("input_path", None,
                    "A path to a dir that contains the input files.")
flags.DEFINE_string("output_path", None,
                    "A path to a dir for writing the output files.")

_NS = "main"


def to_table(result,):
  for table in result["tables"]:
    beam.metrics.Metrics.counter(_NS, "Tables").inc()
    yield table


def to_interaction(
    result,):
  for interaction in result["interactions"]:
    beam.metrics.Metrics.counter(_NS, "Interactions").inc()
    new_interaction = interaction_pb2.Interaction()
    new_interaction.CopyFrom(interaction)
    new_interaction.id = f'{result["split"]}_{interaction.id}'
    for i in range(len(new_interaction.questions)):
      new_interaction.questions[i].id = f"{new_interaction.id}_{i}"
    yield new_interaction


def process_line(line_split,):
  """Parses json and yields result dictionary."""
  beam.metrics.Metrics.counter(_NS, "Lines").inc()
  line, split = line_split
  result = dict(preprocess_nq_utils.parse(line=line,))
  if result["contained"]:
    beam.metrics.Metrics.counter(_NS, "Lines with table").inc()
  result["split"] = split
  return result


def add_split(line, split):
  return line, split


def _merge_group(
    tables,):
  """Merges tables to the table with the latest version."""
  beam.metrics.Metrics.counter(_NS, "Tables: Duplicates").inc(len(tables))
  beam.metrics.Metrics.counter(_NS, "Tables: Unique").inc()
  table = interaction_pb2.Table()
  table.CopyFrom(max(tables, key=preprocess_nq_utils.get_version))
  # Set alternative document urls.
  document_urls = {table.document_url for table in tables}
  document_urls.remove(table.document_url)
  if document_urls:
    table.alternative_document_urls.extend(sorted(document_urls))
  # Set alternative table ids.
  table_ids = {table.table_id for table in tables}
  table_ids.remove(table.table_id)
  if table_ids:
    table.alternative_table_ids.extend(sorted(table_ids))
  return table


def _merge_tables_fn(
    key_tables,
):
  """Merges similar tables returns the latest version."""
  _, tables = key_tables
  tables = list(tables)
  groups = preprocess_nq_utils.group_similar_tables(tables)
  for group in groups:
    yield _merge_group(group)


def _remove_duplicate_tables(tables):
  return (
      tables
      | "KeyByDocTitle" >> beam.Map(lambda table: (table.document_title, table))
      | "GroupByDocTitle" >> beam.GroupByKey()
      | "Pre-Shuffle-Tables" >> beam.transforms.util.Reshuffle()
      | "MergeTables" >> beam.FlatMap(_merge_tables_fn)
      | "Post-Shuffle-Tables" >> beam.transforms.util.Reshuffle())


def build_pipeline(
    filenames,
    output_path,
):
  """Builds the processing pipeline."""

  def _pipeline(root):

    lines = []
    for split, files in filenames.items():
      for filename in files:
        lines.append(
            root | f"Read {filename}" >> beam.io.textio.ReadFromText(
                filename,
                coder=beam.coders.BytesCoder(),
            )
            | f"Add Split {filename}" >> beam.Map(add_split, split=split.name),)

    results = (
        lines | "Flatten" >> beam.Flatten()
        | "Pre-Shuffle" >> beam.transforms.util.Reshuffle()
        | "Parse" >> beam.Map(process_line)
        | "Post-Shuffle" >> beam.transforms.util.Reshuffle())

    tables = (results | "Tables" >> beam.FlatMap(to_table))

    _ = (
        _remove_duplicate_tables(tables)
        | "WriteMergedTables" >> beam.io.WriteToTFRecord(
            file_path_prefix=get_tables(output_path),
            shard_name_template="",
            coder=beam.coders.ProtoCoder(interaction_pb2.Table)))

    _ = (
        results | "Intractions" >> beam.FlatMap(to_interaction)
        | "WriteInteractions" >> beam.io.WriteToTFRecord(
            file_path_prefix=get_interactions(output_path),
            shard_name_template="",
            coder=beam.coders.ProtoCoder(interaction_pb2.Interaction)))

  return _pipeline


def get_interactions(output_dir):
  return os.path.join(output_dir, "nq_premerge.tfrecord")


def get_tables(output_dir, name = "tables"):
  return os.path.join(output_dir, "tables", f"{name}.tfrecord")


def _iterate_proto(filepath, message):
  for value in tf.python_io.tf_record_iterator(filepath):
    i = message()
    i.ParseFromString(value)
    yield i


def iterate_interactions(
    filepath):
  return _iterate_proto(filepath, message=interaction_pb2.Interaction)


def iterate_tables(filepath):
  return _iterate_proto(filepath, message=interaction_pb2.Table)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  input_path = flags.FLAGS.input_path
  output_path = flags.FLAGS.output_path

  filenames = {}

  for corpus in preprocess_nq_utils.Split:
    filenames[corpus] = list(
        preprocess_nq_utils.get_filenames(
            path=input_path,
            split=corpus,
        ))

  pipeline = build_pipeline(
      filenames,
      output_path,
  )
  beam_runner.run(pipeline)

  stats = preprocess_nq_utils.write_interactions_to_files(
      iterate_tables(get_tables(output_path)),
      iterate_interactions(get_interactions(output_path)),
      output_path,
  )
  for name in sorted(stats):
    value = stats[name]
    print(name, value)
    logging.info("%s: %s", name, value)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_path")
  flags.mark_flag_as_required("output_path")
  app.run(main)
