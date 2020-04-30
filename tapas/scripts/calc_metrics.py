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
"""Denotation accuracy calculation for TAPAS predictions over WikiSQL."""

import os

from absl import app
from absl import flags
from tapas.scripts import calc_metrics_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('interactions_file', None,
                    'The file that contains interactions protos.')

flags.DEFINE_string('prediction_files', None,
                    'A list of files that contain model prediction.')

flags.DEFINE_string('denotation_errors_path', None,
                    'If not None, denotation errors are written there.')

flags.DEFINE_bool('is_strong_supervision_available', False,
                  'Whether to store all tables compactly in one file.')


def main(_):
  examples = calc_metrics_utils.read_data_examples_from_interactions(
      FLAGS.interactions_file)

  prediction_file_name = os.path.basename(FLAGS.prediction_files)
  calc_metrics_utils.read_predictions(FLAGS.prediction_files, examples)
  if FLAGS.is_strong_supervision_available:
    results = calc_metrics_utils.calc_structure_metrics(
        examples, FLAGS.denotation_errors_path)
    print('%s: joint_accuracy=%s' % (FLAGS.prediction_files, results.joint_acc))

  denotation_accuracy = calc_metrics_utils.calc_denotation_accuracy(
      examples, FLAGS.denotation_errors_path, prediction_file_name)
  print('%s: denotation_accuracy=%s' %
        (FLAGS.prediction_files, denotation_accuracy))


if __name__ == '__main__':
  flags.mark_flag_as_required('prediction_files')
  flags.mark_flag_as_required('interactions_file')
  app.run(main)
