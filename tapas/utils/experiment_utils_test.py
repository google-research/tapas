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
import os
import tempfile

from absl.testing import absltest
from tapas.utils import experiment_utils

import tensorflow.compat.v1 as tf
from google.protobuf import text_format


class ExperimentUtilsTest(absltest.TestCase):

  def test_iterate_checkpoints_single_step(self):
    results = list(
        experiment_utils.iterate_checkpoints(
            model_dir='path',
            single_step=100,
            marker_file_prefix='path',
            total_steps=None))
    self.assertEqual(results, [(100, 'path/model.ckpt-100')])

  def test_iterate_checkpoints_multi_step(self):
    test_tmpdir = tempfile.mkdtemp()
    checkpoints = [
        os.path.join(test_tmpdir, checkpoint) for checkpoint in
        ['model.ckpt-00001', 'model.ckpt-00002', 'model.ckpt-00003']
    ]
    # Write fake checkpoint file to tmpdir.
    state = tf.train.generate_checkpoint_state_proto(
        test_tmpdir,
        model_checkpoint_path=checkpoints[-1],
        all_model_checkpoint_paths=checkpoints)
    with open(os.path.join(test_tmpdir, 'checkpoint'), 'w') as f:
      f.write(text_format.MessageToString(state))
    for checkpoint in checkpoints:
      with open(f'{checkpoint}.index', 'w') as f:
        f.write('\n')

    marker_file_prefix = os.path.join(test_tmpdir, 'marker')
    results = list(
        experiment_utils.iterate_checkpoints(
            model_dir=test_tmpdir,
            total_steps=3,
            marker_file_prefix=marker_file_prefix))

    expected_steps = [1, 2, 3]
    self.assertEqual(results, list(zip(expected_steps, checkpoints)))
    for step in expected_steps:
      self.assertTrue(tf.gfile.Exists(f'{marker_file_prefix}-{step}.done'))

    results = list(
        experiment_utils.iterate_checkpoints(
            model_dir=test_tmpdir,
            total_steps=3,
            marker_file_prefix=marker_file_prefix))
    self.assertEmpty(results)


if __name__ == '__main__':
  absltest.main()
