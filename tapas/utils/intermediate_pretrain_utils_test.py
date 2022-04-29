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

import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from tapas.protos import interaction_pb2
from tapas.utils import beam_runner
from tapas.utils import contrastive_statements_test_utils
from tapas.utils import intermediate_pretrain_utils
from tapas.utils import synthesize_entablement
from tapas.utils import tf_example_utils

import tensorflow.compat.v1 as tf
from google.protobuf import text_format


FLAGS = flags.FLAGS
TEST_PATH = "tapas/utils/testdata/"
_RESERVED_SYMBOLS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EMPTY]")


def _read_record(filepath, message):
  for value in tf.python_io.tf_record_iterator(filepath):
    element = message()
    element.ParseFromString(value)
    yield element


def _create_vocab(vocab, vocab_path):
  with tf.gfile.Open(vocab_path, "w") as input_handle:
    input_handle.write("\n".join(vocab))


class CreatePretrainingDataTest(parameterized.TestCase):

  def setUp(self):
    super(CreatePretrainingDataTest, self).setUp()

    self._test_dir = TEST_PATH


  @parameterized.parameters(
      (beam_runner.RunnerType.DIRECT, False),
      (beam_runner.RunnerType.DIRECT, True),
  )
  def test_end_to_end(self, runner_type, add_example_conversion):
    mode = intermediate_pretrain_utils.Mode.ALL
    prob_count_aggregation = 0.2
    use_fake_table = False
    add_opposite_table = False
    drop_without_support_rate = 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
      config = None
      if add_example_conversion:
        vocab_path = os.path.join(temp_dir, "vocab.txt")
        _create_vocab(list(_RESERVED_SYMBOLS) + ["released"], vocab_path)
        config = tf_example_utils.ClassifierConversionConfig(
            vocab_file=vocab_path,
            max_seq_length=32,
            max_column_id=32,
            max_row_id=32,
            strip_column_names=False,
        )

      pipeline = intermediate_pretrain_utils.build_pipeline(
          mode=mode,
          config=synthesize_entablement.SynthesizationConfig(
              prob_count_aggregation=prob_count_aggregation),
          use_fake_table=use_fake_table,
          add_opposite_table=add_opposite_table,
          drop_without_support_rate=drop_without_support_rate,
          input_file=os.path.join(self._test_dir,
                                  "pretrain_interactions.txtpb"),
          output_dir=temp_dir,
          output_suffix=".tfrecord",
          num_splits=3,
          conversion_config=config,
      )

      beam_runner.run_type(pipeline, runner_type).wait_until_finish()

      message_type = interaction_pb2.Interaction
      if add_example_conversion:
        message_type = tf.train.Example

      for name in [("train"), ("test")]:
        self.assertNotEmpty(
            list(
                _read_record(
                    os.path.join(temp_dir, f"{name}.tfrecord"),
                    message_type,
                )))

      if add_example_conversion:
        self.assertNotEmpty(
            list(
                _read_record(
                    os.path.join(temp_dir, "interactions.tfrecord"),
                    interaction_pb2.Interaction,
                ),))


if __name__ == "__main__":
  absltest.main()
