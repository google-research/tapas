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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from tapas.utils import beam_runner
from tapas.utils import pretrain_utils
from tapas.utils import tf_example_utils

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
TEST_PATH = "tapas/utils/testdata/"

_RESERVED_SYMBOLS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EMPTY]")


def read_examples(filepath):
  for value in tf.python_io.tf_record_iterator(filepath):
    example = tf.train.Example()
    example.ParseFromString(value)
    yield example


class CreatePretrainingDataTest(parameterized.TestCase):

  def setUp(self):
    super(CreatePretrainingDataTest, self).setUp()

    self._test_dir = TEST_PATH

  def _create_vocab(self, vocab, vocab_path):
    with tf.gfile.Open(vocab_path, "w") as input_handle:
      input_handle.write("\n".join(vocab))

  @parameterized.parameters(
      (beam_runner.RunnerType.DIRECT,),
  )
  def test_end_to_end(self, runner_type):

    with tempfile.TemporaryDirectory() as temp_dir:
      vocab_path = os.path.join(temp_dir, "vocab.txt")

      self._create_vocab(list(_RESERVED_SYMBOLS) + ["released"], vocab_path)

      pipeline = pretrain_utils.build_pretrain_data_pipeline(
          input_files=[
              os.path.join(self._test_dir, "pretrain_interactions.txtpb")
          ],
          output_dir=temp_dir,
          config=tf_example_utils.PretrainConversionConfig(
              vocab_file=vocab_path,
              max_seq_length=10,
              max_predictions_per_seq=10,
              random_seed=5,
              masked_lm_prob=0.5,
              max_column_id=3,
              max_row_id=3,
              min_question_length=1,
              max_question_length=4,
              always_continue_cells=True,
              strip_column_names=False),
          dupe_factor=2,
          min_num_columns=0,
          min_num_rows=0,
      )

      beam_runner.run_type(pipeline, runner_type).wait_until_finish()

      for name, expected_len in [("train", 20), ("test", 2)]:
        examples = list(
            read_examples(os.path.join(temp_dir, f"{name}.tfrecord")))
        self.assertLen(examples, expected_len)


if __name__ == "__main__":
  absltest.main()
