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
"""Pipeline for generating synthetic statements for tables.

   This implements the intermediate pre-training generation discussed in
   "Understanding tables with intermediate pre-training"
   (https://arxiv.org/abs/2010.00571).

   The script will output TF examples (default) or interactions
   (if `convert_to_examples` is set to false).

  The outputs will always be deterministically split into a train and a test
  set.

  If a `convert_to_examples` is true the intermediate interactions will be
  written to an additional interaction file
  (with path "f{output_dir}/interaction{output_suffix}").
"""

from absl import app
from absl import flags
from tapas.utils import beam_runner
from tapas.utils import intermediate_pretrain_utils
from tapas.utils import synthesize_entablement
from tapas.utils import tf_example_utils

flags.DEFINE_string(
    "input_file",
    None,
    "'.txtpb' or '.tfrecord' files with interaction protos.",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "output directory.",
)
flags.DEFINE_string(
    "output_suffix",
    ".tfrecord",
    "Should be '.tfrecod' or '.txtpb'",
)
flags.DEFINE_string(
    "vocab_file",
    None,
    "The vocabulary file that the BERT model was trained on.",
)
flags.DEFINE_bool(
    "convert_to_examples",
    True,
    "If true convert interactions to examples.",
)
flags.DEFINE_enum_class(
    "mode",
    intermediate_pretrain_utils.Mode.ALL,
    intermediate_pretrain_utils.Mode,
    "Mode to run in.",
)
flags.DEFINE_integer(
    "max_seq_length",
    128,
    "See tf_example_utils.ClassifierConversionConfig",
)
flags.DEFINE_boolean(
    "use_fake_table",
    False,
    "Replace table with a constant.",
)
flags.DEFINE_boolean(
    "add_opposite_table",
    False,
    "If, true add opposite table.",
)
flags.DEFINE_float(
    "prob_count_aggregation",
    0.02,
    "See SynthesizationConfig.",
)
flags.DEFINE_float(
    "drop_without_support_rate",
    1.0,
    "If true, drop contrastive examples without support.",
)

FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv
  config = synthesize_entablement.SynthesizationConfig(
      prob_count_aggregation=FLAGS.prob_count_aggregation,)
  conversion_config = None
  if FLAGS.convert_to_examples:
    conversion_config = tf_example_utils.ClassifierConversionConfig(
        vocab_file=FLAGS.vocab_file,
        max_seq_length=FLAGS.max_seq_length,
        max_column_id=FLAGS.max_seq_length,
        max_row_id=FLAGS.max_seq_length,
        strip_column_names=False,
    )
  pipeline = intermediate_pretrain_utils.build_pipeline(
      mode=FLAGS.mode,
      config=config,
      use_fake_table=FLAGS.use_fake_table,
      add_opposite_table=FLAGS.add_opposite_table,
      drop_without_support_rate=FLAGS.drop_without_support_rate,
      input_file=FLAGS.input_file,
      output_dir=FLAGS.output_dir,
      output_suffix=FLAGS.output_suffix,
      conversion_config=conversion_config)
  beam_runner.run(pipeline)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
