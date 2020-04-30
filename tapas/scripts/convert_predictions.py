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
"""Scripts to convert predictions file to other formats."""

from absl import app
from absl import flags
from tapas.scripts import convert_predictions_utils

FLAGS = flags.FLAGS

flags.DEFINE_list('interaction_files', None,
                  'A list of files contain interactions protos.')
flags.DEFINE_list('prediction_files', None,
                  'A list of files that contain model prediction.')
flags.DEFINE_string('output_directory', None,
                    'Output directory where converted files will be stored.')
flags.DEFINE_enum_class('dataset_format', None,
                        convert_predictions_utils.DatasetFormat,
                        'Dataset format.')


def main(_):
  convert_predictions_utils.convert(FLAGS.interaction_files,
                                    FLAGS.prediction_files,
                                    FLAGS.output_directory,
                                    FLAGS.dataset_format)


if __name__ == '__main__':
  flags.mark_flag_as_required('interaction_files')
  flags.mark_flag_as_required('prediction_files')
  flags.mark_flag_as_required('output_directory')
  flags.mark_flag_as_required('dataset_format')
  app.run(main)
