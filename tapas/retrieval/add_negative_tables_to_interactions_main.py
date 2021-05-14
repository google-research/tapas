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
"""Adds nearest neigbors as negatives to interactions.

Given 'interaction_dir' extracts all files and searches for
the nearest neighbor files in json format in 'json_dir'.

For example, for an interaction file 'test' we expect to find a
file 'test_results.jsonl' in 'json_dir'.

The table ids in the json files should have a corresponding table in
'input_table_file'.

After the processing the interactions are written to 'output_dir' keeping the
original file name.
"""
import os
from typing import List, Text

from absl import app
from absl import flags
import dataclasses
from tapas.retrieval.add_negative_tables_to_interactions import add_negative_tables_to_interactions
from tapas.utils import beam_runner
import tensorflow.compat.v1 as tf


flags.DEFINE_string("interaction_dir", None,
                    "Directory with interaction tfrecords.")

flags.DEFINE_string("json_dir", None, "Directory with jsonl files.")

flags.DEFINE_string("input_tables_file", None, "The tfrecord tables file.")

flags.DEFINE_string("output_dir", None,
                    "Directory where new interactions are written to.")

flags.DEFINE_integer(
    "max_num_negatives",
    None,
    "Max negatives examples to add to interaction.",
)

FLAGS = flags.FLAGS


@dataclasses.dataclass(frozen=True)
class InputsOutputs:
  input_interaction_files: List[Text]
  input_json_files: List[Text]
  output_interaction_files: List[Text]


def _get_inputs_outputs(
    interaction_dir,
    json_dir,
    output_dir,
):
  """Gets input and output files."""

  interaction_paths = tf.io.gfile.glob(
      os.path.join(interaction_dir, "*.tfrecord"))
  if not interaction_paths:
    raise ValueError(f"No interactions found: {interaction_dir}")

  interaction_names = [os.path.basename(path) for path in interaction_paths]

  json_files = []
  for name in interaction_names:
    base_name = os.path.splitext(name)[0]
    json_file = os.path.join(json_dir, f"{base_name}_results.jsonl")
    json_files.append(json_file)
    if not tf.io.gfile.exists(json_file):
      raise ValueError(f"Missing file: {json_file}")

  outputs = [os.path.join(output_dir, name) for name in interaction_names]
  return InputsOutputs(
      input_interaction_files=interaction_paths,
      input_json_files=json_files,
      output_interaction_files=outputs,
  )


def main(unused_argv):
  r"""Reads nearest neigbors adds them to the interactions."""
  del unused_argv

  inputs_outputs = _get_inputs_outputs(
      FLAGS.interaction_dir,
      FLAGS.json_dir,
      FLAGS.output_dir,
  )
  pipeline = add_negative_tables_to_interactions(
      max_num_negatives=FLAGS.max_num_negatives,
      input_interactions_files=inputs_outputs.input_interaction_files,
      input_tables_file=FLAGS.input_tables_file,
      input_json_files=inputs_outputs.input_json_files,
      output_files=inputs_outputs.output_interaction_files,
  )
  beam_runner.run(pipeline)


if __name__ == "__main__":
  flags.mark_flag_as_required("interaction_dir")
  flags.mark_flag_as_required("input_tables_file")
  flags.mark_flag_as_required("json_dir")
  flags.mark_flag_as_required("output_dir")

  app.run(main)
