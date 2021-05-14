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
"""Helpers for mapping input to output filenames."""

import os
from typing import Text, Tuple, List, Set

import tensorflow.compat.v1 as tf



def _check_basename(
    basenames,
    basename,
    input_dir,
):
  if basename in basenames:
    raise ValueError("Basename should be unique:"
                     f"basename: {basename}, input_dir:{input_dir}")
  basenames.add(basename)


def _is_supported(filename):
  extension = os.path.splitext(filename)[1]
  return extension in [
      ".txtpb.gz",
      ".txtpb",
      ".tfrecord",
      ".tfrecords",
  ]


def get_inputs_and_outputs(input_dir,
                           output_dir):
  """Reads files from 'input_dir' and creates corresponding paired outputs.

  Args:
    input_dir: Where to read inputs from.
    output_dir: Where to read outputs from.

  Returns:
    inputs and outputs.
  """
  input_files = tf.io.gfile.listdir(input_dir)

  basenames = set()

  inputs = []
  outputs = []

  for filename in input_files:
    if not _is_supported(filename):
      print(f"Skipping unsupported file: {filename}")
      continue
    basename, _ = os.path.splitext(filename)
    _check_basename(basenames, basename, input_dir)
    inputs.append(filename)
    output = f"{basename}.tfrecord"
    outputs.append(output)

  inputs = [os.path.join(input_dir, i) for i in inputs]
  outputs = [os.path.join(output_dir, o) for o in outputs]
  return inputs, outputs
