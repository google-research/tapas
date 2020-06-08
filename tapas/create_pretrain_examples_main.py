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
r"""Converts interactions in text format to TF examples.

# Direct runner.
python3 tapas/create_pretrain_examples_main.py \
  --input_file="/tmp/interactions_sample.txtpb.gz" \
  --output_dir="/tmp/" \
  --vocab_file="/tmp/vocab.txt" \
  --runner_type="DIRECT"

# Dataflow runner (runs on Google Cloud).
python3 setup.py sdist
python3 tapas/create_pretrain_examples_main.py \
  --input_file="gs://tapas_models/2020_05_11/interactions_sample.txtpb.gz" \
  --output_dir="gs://your_bucket/output" \
  --vocab_file="gs://tapas_models/2020_05_11/vocab.txt" \
  --runner_type=DATAFLOW \
  --gc_project="you-project" \
  --gc_region="us-west1" \
  --gc_job_name="create-pretrain" \
  --gc_staging_location="gs://your_bucket/staging" \
  --gc_temp_location="gs://your_bucket/tmp" \
  --extra_packages=dist/tapas-0.0.1.dev0.tar.gz

"""

from absl import app
from absl import flags

from tapas.utils import beam_runner
from tapas.utils import pretrain_utils
from tapas.utils import tf_example_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Compressed interaction in text format.")
flags.DEFINE_string("output_dir", None,
                    "Directory where new data is written to.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("max_seq_length", 128,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("random_seed", 12345,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("dupe_factor", 10,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_float("masked_lm_prob", 0.15,
                   "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("max_column_id", 512,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("max_row_id", 512,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("min_num_rows", 0,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("min_num_columns", 0,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("min_question_length", 8,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_integer("max_question_length", 32,
                     "See tf_example_utils.PretrainConversionConfig")
flags.DEFINE_bool("always_continue_cells", True,
                  "See tf_example_utils.PretrainConversionConfig")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  config = tf_example_utils.PretrainConversionConfig(
      vocab_file=FLAGS.vocab_file,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      random_seed=FLAGS.random_seed,
      masked_lm_prob=FLAGS.masked_lm_prob,
      max_column_id=FLAGS.max_column_id,
      max_row_id=FLAGS.max_row_id,
      min_question_length=FLAGS.min_question_length,
      max_question_length=FLAGS.max_question_length,
      always_continue_cells=FLAGS.always_continue_cells,
      strip_column_names=False,
  )
  pipeline = pretrain_utils.build_pretrain_data_pipeline(
      input_files=[FLAGS.input_file],
      output_dir=FLAGS.output_dir,
      config=config,
      dupe_factor=FLAGS.dupe_factor,
      min_num_rows=FLAGS.min_num_rows,
      min_num_columns=FLAGS.min_num_columns,
  )
  beam_runner.run(pipeline).wait_until_finish()


if __name__ == "__main__":
  app.run(main)
