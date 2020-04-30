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
"""Evaluates WikiSQL predictions against NSM gold json files."""

import json

from absl import app
from absl import flags
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_list('reference_files', None, 'NSM json gold file.')
flags.DEFINE_list('prediction_files', None,
                  'Produced by "convert_predictions".')
flags.DEFINE_string('dataset', None, '"dev" or "test".')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert FLAGS.dataset
  assert len(FLAGS.reference_files) == len(FLAGS.prediction_files)
  for reference_file, prediction_file in zip(FLAGS.reference_files,
                                             FLAGS.prediction_files):

    with tf.io.gfile.GFile(prediction_file, 'r') as input_file:
      predictions = {}
      for line in input_file:
        line = line.strip()
        if line:
          segments = line.split('\t')
          key = segments[0]
          value = segments[1:]
          predictions[key] = text_utils.normalize_answers(value)

    with tf.io.gfile.GFile(reference_file, 'r') as input_file:
      reference = json.load(input_file)

    references = {}
    for index, data in enumerate(reference):
      key = '%s-%d' % (FLAGS.dataset, index)
      references[key] = text_utils.normalize_answers(data)

    num_correct = 0
    for key, gold_answer in references.items():
      pred_answer = predictions[key]
      is_correct = gold_answer == pred_answer
      if is_correct:
        num_correct += 1

    print('Correct: ', num_correct, len(references),
          num_correct / float(len(references)))


if __name__ == '__main__':
  app.run(main)
