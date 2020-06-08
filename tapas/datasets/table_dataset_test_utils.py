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
"""Test utility functions for table dataset."""

import random
import string

import numpy as np
from tapas.datasets import table_dataset
import tensorflow.compat.v1 as tf


def make_tf_example(value_dict):
  """Converts a dict of numpy arrays into a tf.Example proto."""
  features = {}
  for item, values in value_dict.items():
    if values.dtype == np.float32:
      features[item] = tf.train.Feature(
          float_list=tf.train.FloatList(value=values.ravel()))
    elif values.dtype == np.int32:
      features[item] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=values.ravel()))
    else:  # Handle feature as string.
      features[item] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[tf.compat.as_bytes(t) for t in values]))
  return tf.train.Example(features=tf.train.Features(feature=features))


def create_random_example(
    max_seq_length,
    max_predictions_per_seq,
    task_type,
    add_aggregation_function_id,
    add_classification_labels,
    add_answer,
    include_id,
    vocab_size,
    segment_vocab_size,
    num_columns,
    num_rows,
    add_candidate_answers,
    max_num_candidates,
):
  """Returns a random table example."""
  values = dict(
      input_ids=np.random.randint(
          vocab_size, size=[max_seq_length], dtype=np.int32),
      input_mask=np.random.randint(2, size=[max_seq_length], dtype=np.int32),
      segment_ids=np.random.randint(
          segment_vocab_size, size=[max_seq_length], dtype=np.int32),
      column_ids=np.random.randint(
          num_columns, size=[max_seq_length], dtype=np.int32),
      row_ids=np.random.randint(
          num_rows, size=[max_seq_length], dtype=np.int32),
      prev_label_ids=np.zeros([max_seq_length], dtype=np.int32),
      column_ranks=np.random.randint(
          num_columns, size=[max_seq_length], dtype=np.int32),
      inv_column_ranks=np.random.randint(
          num_columns, size=[max_seq_length], dtype=np.int32),
      numeric_relations=np.random.randint(
          10, size=[max_seq_length], dtype=np.int32))

  if task_type == table_dataset.TableTask.PRETRAINING:
    values["masked_lm_positions"] = np.random.randint(
        2, size=[max_predictions_per_seq], dtype=np.int32)
    values["masked_lm_ids"] = np.random.randint(
        2, size=[max_predictions_per_seq], dtype=np.int32)
    values["masked_lm_weights"] = np.random.rand(
        max_predictions_per_seq).astype("f")
    values["next_sentence_labels"] = np.random.randint(
        2, size=[1], dtype=np.int32)
  elif task_type == table_dataset.TableTask.CLASSIFICATION:
    values["label_ids"] = np.random.randint(
        2, size=[max_seq_length], dtype=np.int32)
    if add_aggregation_function_id:
      values["aggregation_function_id"] = np.random.randint(
          4, size=[1], dtype=np.int32)
    if add_classification_labels:
      values["classification_class_index"] = np.random.randint(
          6, size=[1], dtype=np.int32)
    if add_answer:
      answer = np.array([np.nan
                        ]) if np.random.rand() < 0.5 else np.random.uniform(
                            -100, 100, size=[1])
      values["answer"] = answer.astype(np.float32)
      values["numeric_values_scale"] = np.random.randint(
          low=1, high=5, size=[max_seq_length],
          dtype=np.int32).astype(np.float32)
      numeric_values = np.where(
          np.random.rand(max_seq_length) < 0.5, np.nan,
          np.random.uniform(-100, 100, size=[max_seq_length]))
      values["numeric_values"] = numeric_values.astype(np.float32)
  else:
    raise ValueError(f"Unsupported task type: {task_type}")

  if add_candidate_answers:
    values["cand_num"] = np.random.randint(
        low=1, high=max_num_candidates + 1, size=(), dtype=np.int32)
    values["can_aggregation_function_ids"] = np.random.randint(
        low=0, high=4, size=[max_num_candidates], dtype=np.int32)
    values["can_sizes"] = np.random.randint(
        low=0, high=max_seq_length, size=[max_num_candidates], dtype=np.int32)
    indices = []
    for size in values["can_sizes"]:
      indices.append(
          np.random.choice(max_seq_length, size=[size],
                           replace=False).astype(np.int32))
    values["can_indexes"] = np.concatenate(indices)

  if include_id:
    question_id = "".join(
        random.choice(string.ascii_lowercase) for i in range(10))
    values["question_id"] = np.array([str.encode(question_id)])

  return values


def create_random_dataset(num_examples, batch_size, repeat, generator_kwargs):
  """Creates a dataset out of random examples.

  Args:
    num_examples: Number of examples to generate.
    batch_size: Batch size.
    repeat: Whether to repeat the examples forever.
    generator_kwargs: dict of arguments for create_random_example.

  Returns:
    A tf.data.Dataset with parsed examples.
  """
  examples = []
  for _ in range(num_examples):
    example = make_tf_example(create_random_example(**generator_kwargs))
    examples.append(example.SerializeToString())

  dataset = tf.data.Dataset.from_tensor_slices(examples)
  if repeat:
    dataset = dataset.repeat()

  parse_fn = table_dataset.parse_table_examples(
      max_seq_length=generator_kwargs["max_seq_length"],
      max_predictions_per_seq=generator_kwargs["max_predictions_per_seq"],
      task_type=generator_kwargs["task_type"],
      add_aggregation_function_id=generator_kwargs[
          "add_aggregation_function_id"],
      add_classification_labels=generator_kwargs["add_classification_labels"],
      add_answer=generator_kwargs["add_answer"],
      include_id=generator_kwargs["include_id"],
      add_candidate_answers=generator_kwargs["add_candidate_answers"],
      max_num_candidates=generator_kwargs["max_num_candidates"],
      params={"batch_size": batch_size})
  dataset = dataset.map(parse_fn)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  return dataset
