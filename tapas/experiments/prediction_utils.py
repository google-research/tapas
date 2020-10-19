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
"""Writes prediction to a csv file."""

import collections
import copy
import csv
import json
from typing import Mapping, Text, Tuple, Iterable, List

from absl import logging
import dataclasses
import numpy as np
from tapas.models import tapas_classifier_model
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf


def read_classifier_dataset(
    predict_data,
    data_format,
    compression_type,
    max_seq_length,
    max_predictions_per_seq,
    add_aggregation_function_id,
    add_classification_labels,
    add_answer,
):
  """Reads the classification dataset into memory as numpy arrays."""
  dataset = tapas_classifier_model.input_fn(
      name="predict",
      file_patterns=predict_data,
      data_format=data_format,
      compression_type=compression_type,
      is_training=False,
      max_seq_length=max_seq_length,
      max_predictions_per_seq=max_predictions_per_seq,
      add_aggregation_function_id=add_aggregation_function_id,
      add_classification_labels=add_classification_labels,
      add_answer=add_answer,
      include_id=True,
      params={"batch_size": 1})
  get_next = dataset.make_one_shot_iterator().get_next()

  examples_by_position = collections.defaultdict(dict)
  with tf.Session() as sess:
    try:
      while True:
        example = sess.run(get_next)
        question_id = example["question_id"][0, 0].decode("utf-8")
        table_id, annotator, position = text_utils.parse_question_id(
            question_id)
        example_id = (table_id, annotator)
        examples_by_position[position][example_id] = example
    except tf.errors.OutOfRangeError:
      pass
  return examples_by_position


def _get_in_memory_input_fn(examples):
  """An input function that reads examples from numpy arrays in memory."""

  def _input_fn(params):
    """The input function."""
    features = collections.defaultdict(list)
    for example in examples:
      for feature_key in example:
        features[feature_key].append(example[feature_key])
    dataset = tf.data.Dataset.zip({
        feature_key: tf.data.Dataset.from_tensor_slices(
            np.concatenate(features[feature_key], axis=0))
        for feature_key in features
    })
    return dataset.batch(params["batch_size"])

  return _input_fn


def compute_prediction_sequence(estimator, examples_by_position):
  """Computes predictions using model's answers to the previous questions."""
  all_results = []
  prev_answers = None

  for position in range(len(examples_by_position)):
    examples = copy.deepcopy(examples_by_position[position])
    if prev_answers is not None:
      for example_id in examples:
        coords_to_answer = prev_answers[example_id]
        example = examples[example_id]
        prev_label_ids = example["prev_label_ids"]
        model_label_ids = np.zeros_like(prev_label_ids)
        for i in range(model_label_ids.shape[1]):
          row_id = example["row_ids"][0, i] - 1
          col_id = example["column_ids"][0, i] - 1
          if row_id >= 0 and col_id >= 0 and example["segment_ids"][0, i] == 1:
            model_label_ids[0, i] = int(coords_to_answer[(col_id, row_id)])
        examples[example_id]["prev_label_ids"] = model_label_ids

    results = list(
        estimator.predict(input_fn=_get_in_memory_input_fn(examples.values())))
    all_results.extend(results)

    prev_answers = {}
    for prediction in results:
      question_id = prediction["question_id"][0].decode("utf-8")
      table_id, annotator, _ = text_utils.parse_question_id(question_id)
      example_id = (table_id, annotator)
      example = examples[example_id]
      probabilities = prediction["probabilities"]

      # Compute average probability per cell, aggregating over tokens.
      coords_to_probs = collections.defaultdict(list)
      for i, p in enumerate(probabilities):
        segment_id = prediction["segment_ids"][i]
        col = prediction["column_ids"][i] - 1
        row = prediction["row_ids"][i] - 1
        if col >= 0 and row >= 0 and segment_id == 1:
          coords_to_probs[(col, row)].append(p)

      coords_to_answer = {}
      for key in coords_to_probs:
        coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.5
      prev_answers[example_id] = coords_to_answer

  return all_results


def _get_question_id(features):
  """Restores question id from int sequence."""
  if "question_id_ints" in features:
    question_id = text_utils.ints_to_str(list(features["question_id_ints"]))
    if question_id:
      return question_id
  # TODO Remove once the data has been updated.
  return features["question_id"][0].decode("utf-8")


def get_cell_token_probs(prediction):
  probabilities = prediction["probabilities"]
  for i, p in enumerate(probabilities):
    segment_id = prediction["segment_ids"][i]
    col = prediction["column_ids"][i] - 1
    row = prediction["row_ids"][i] - 1
    if col >= 0 and row >= 0 and segment_id == 1:
      yield i, p


def get_mean_cell_probs(prediction):
  """Computes average probability per cell, aggregating over tokens."""

  coords_to_probs = collections.defaultdict(list)
  for i, prob in get_cell_token_probs(prediction):
    col = prediction["column_ids"][i] - 1
    row = prediction["row_ids"][i] - 1
    coords_to_probs[(col, row)].append(prob)
  return {
      coords: np.array(cell_probs).mean()
      for coords, cell_probs in coords_to_probs.items()
  }


def _geometric_mean(values):
  """Computes geometric mean, assumes all values > 0."""
  return np.exp(np.mean(np.log(values)))


@dataclasses.dataclass(frozen=True)
class TokenAnswer:
  """An answer extracted from a table cell."""
  # Index of the column the answer is in.
  column_index: int
  # Index of the row the answer is in.
  row_index: int
  # Start token index with-in the cell.
  begin_token_index: int
  # End token index with-in the cell (exlusive).
  end_token_index: int
  # Indexes of the word pieces in the vocabulary.
  token_ids: List[int]
  # Answer score.
  score: float


def _to_token_answer(
    prediction,
    begin_token_index,
    end_token_index,
    score,
):
  """Gets the cell cordinates and relative token indexes."""
  input_ids = prediction["input_ids"]
  column_ids = prediction["column_ids"]
  row_ids = prediction["row_ids"]

  column_id = column_ids[begin_token_index]
  row_id = row_ids[begin_token_index]

  cell_begin_token_index = list(zip(row_ids, column_ids)).index(
      (row_id, column_id))

  return TokenAnswer(
      column_index=int(column_id - 1),
      row_index=int(row_id - 1),
      begin_token_index=int(begin_token_index - cell_begin_token_index),
      end_token_index=int(end_token_index - cell_begin_token_index),
      token_ids=[
          int(input_ids[i]) for i in range(begin_token_index, end_token_index)
      ],
      score=float(score),
  )


def _get_token_answers(
    prediction,
    cell_classification_threshold,
):
  """Computes answer indexes."""
  span_indexes = prediction.get("span_indexes")
  span_logits = prediction.get("span_logits")
  if span_indexes is not None and span_logits is not None:
    best_logit, best_span = max(zip(span_logits, span_indexes.tolist()),)
    logging.log_every_n(
        logging.INFO,
        "best_span: %s, score: %s",
        500,
        best_span,
        best_logit,
    )
    return [
        _to_token_answer(
            prediction,
            best_span[0],
            best_span[1] + 1,
            best_logit,
        )
    ]

  answers = []

  answer_begin_index = None
  answer_end_index = None
  answer_probs = []
  for i, prob in get_cell_token_probs(prediction):
    if prob > cell_classification_threshold:

      if answer_end_index is not None:
        if answer_end_index < i:
          # There is a gap between the current answer and the new index.
          answers.append(
              _to_token_answer(
                  prediction,
                  answer_begin_index,
                  answer_end_index,
                  _geometric_mean(answer_probs),
              ))
          answer_begin_index = None
          answer_end_index = None
          answer_probs.clear()

      if answer_begin_index is None:
        answer_begin_index = i
      answer_end_index = i + 1
      answer_probs.append(prob)
  if answer_begin_index is not None:
    answers.append(
        _to_token_answer(
            prediction,
            begin_token_index=answer_begin_index,
            end_token_index=answer_end_index,
            score=_geometric_mean(answer_probs),
        ))
  return answers


def _write_prediction(
    prediction,
    cell_classification_threshold,
    do_model_aggregation,
    do_model_classification,
    writer,
):
  """Writes a single prediction to TSV."""
  question_id = _get_question_id(prediction)
  max_width = prediction["column_ids"].max()
  max_height = prediction["row_ids"].max()

  if (max_width == 0 and max_height == 0 and
      question_id == text_utils.get_padded_question_id()):
    logging.info("Removing padded example: %s", question_id)
    return

  cell_coords_to_prob = get_mean_cell_probs(prediction)

  answers = _get_token_answers(
      prediction,
      cell_classification_threshold,
  )

  # Select the answers above a classification threshold.
  answer_coordinates = []
  for col in range(max_width):
    for row in range(max_height):
      cell_prob = cell_coords_to_prob.get((col, row), None)
      if cell_prob is not None:
        if cell_prob > cell_classification_threshold:
          answer_coordinates.append(str((row, col)))

  try:
    example_id, annotator, position = text_utils.parse_question_id(question_id)
    position = str(position)
  except ValueError:
    example_id = "_"
    annotator = "_"
    position = "_"
  prediction_to_write = {
      "question_id": question_id,
      "id": example_id,
      "annotator": annotator,
      "position": position,
      "answer_coordinates": str(answer_coordinates),
      "answers": token_answers_to_text(answers),
  }
  if do_model_aggregation:
    prediction_to_write["gold_aggr"] = str(prediction["gold_aggr"][0])
    prediction_to_write["pred_aggr"] = str(prediction["pred_aggr"])
  if do_model_classification:
    prediction_to_write["gold_cls"] = str(prediction["gold_cls"][0])
    prediction_to_write["pred_cls"] = str(prediction["pred_cls"])
    prediction_to_write["logits_cls"] = str(prediction["logits_cls"])
  writer.writerow(prediction_to_write)


def token_answers_to_text(answers):
  return json.dumps([dataclasses.asdict(answer) for answer in answers])


def token_answers_from_text(text):
  return [TokenAnswer(**answer_dict) for answer_dict in json.loads(text)]


def write_predictions(
    predictions,
    output_predict_file,
    do_model_aggregation,
    do_model_classification,
    cell_classification_threshold,
):
  """Writes predictions to an output TSV file.

  Predictions header: [id, annotator, position, answer_coordinates, gold_aggr,
  pred_aggr]

  Args:
    predictions: model predictions
    output_predict_file: Path for wrinting the predicitons.
    do_model_aggregation: Indicates whther to write predicted aggregations.
    do_model_classification: Indicates whther to write predicted classes.
    cell_classification_threshold: Threshold for selecting a cell.
  """
  with tf.io.gfile.GFile(output_predict_file, "w") as write_file:
    writer = None
    header = [
        "question_id",
        "id",
        "annotator",
        "position",
        "answer_coordinates",
        "answers",
    ]
    if do_model_aggregation:
      header.extend(["gold_aggr", "pred_aggr"])
    if do_model_classification:
      header.extend(["gold_cls", "pred_cls", "logits_cls"])
    for prediction in predictions:
      if writer is None:
        writer = csv.DictWriter(write_file, fieldnames=header, delimiter="\t")
        writer.writeheader()

      _write_prediction(
          prediction,
          cell_classification_threshold=cell_classification_threshold,
          do_model_classification=do_model_classification,
          do_model_aggregation=do_model_aggregation,
          writer=writer,
      )
    # If there are no predictions we should still write the file
    if writer is None:
      writer = csv.DictWriter(write_file, fieldnames=header, delimiter="\t")
      writer.writeheader()
