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
"""Utilities to convert predictions file to other formats."""

import enum
import os
from typing import Text, List

from tapas.scripts import calc_metrics_utils
from tapas.scripts import prediction_utils
import tensorflow.compat.v1 as tf


class DatasetFormat(enum.Enum):
  WIKITABLEQUESTIONS = 0


def _convert_single_wtq(interaction_file, prediction_file,
                        output_file):
  """Convert predictions to WikiTablequestions format."""

  interactions = dict(
      (prediction_utils.parse_interaction_id(i.id), i)
      for i in prediction_utils.iterate_interactions(interaction_file))
  missing_interaction_ids = set(interactions.keys())

  with tf.io.gfile.GFile(output_file, 'w') as output_file:
    for prediction in prediction_utils.iterate_predictions(prediction_file):
      interaction_id = prediction['id']
      if interaction_id in missing_interaction_ids:
        missing_interaction_ids.remove(interaction_id)

      coordinates = prediction_utils.parse_coordinates(
          prediction['answer_coordinates'])

      denot_pred, _ = calc_metrics_utils.execute(
          int(prediction.get('pred_aggr', '0')), coordinates,
          prediction_utils.table_to_panda_frame(
              interactions[interaction_id].table))

      answers = '\t'.join(sorted(map(str, denot_pred)))
      output_file.write('{}\t{}\n'.format(interaction_id, answers))

    for interaction_id in missing_interaction_ids:
      output_file.write('{}\n'.format(interaction_id))


def _convert_single(interaction_file, prediction_file,
                    output_file, dataset_format):
  if dataset_format == DatasetFormat.WIKITABLEQUESTIONS:
    return _convert_single_wtq(interaction_file, prediction_file, output_file)
  else:
    raise ValueError('Unknown dataset format {}'.format(dataset_format))


def convert(interactions, predictions,
            output_directory, dataset_format):
  assert len(interactions) == len(predictions)
  for interaction_file, prediction_file in zip(interactions, predictions):
    output_file = os.path.join(output_directory,
                               os.path.basename(prediction_file))
    _convert_single(interaction_file, prediction_file, output_file,
                    dataset_format)
