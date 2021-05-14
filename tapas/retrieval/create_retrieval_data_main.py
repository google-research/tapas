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
"""Convert interactions to Tensorflow examples."""

from absl import app
from absl import flags
from tapas.retrieval import tf_example_utils
from tapas.utils import beam_runner
from tapas.utils import create_data
from tapas.utils import create_data_file_io


flags.DEFINE_string("input_interactions_dir", None, "Directory with inputs.")
flags.DEFINE_string("input_tables_dir", None, "Directory with inputs.")
flags.DEFINE_string("output_dir", None, "Directory with outputs.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_seq_length", None,
                     "Max length of a sequence in word pieces.")
flags.DEFINE_float("max_column_id", None, "Max column id to extract.")
flags.DEFINE_float("max_row_id", None, "Max row id to extract.")
flags.DEFINE_integer("cell_trim_length", -1,
                     "If > 0: Trim cells so that the length is <= this value.")
flags.DEFINE_boolean("use_document_title", None,
                     "Include document title text in the tf example.")
flags.DEFINE_enum_class("converter_impl", create_data.ConverterImplType.PYTHON,
                        create_data.ConverterImplType,
                        "Implementation to map interactions to tf examples.")
FLAGS = flags.FLAGS


def run(inputs, outputs, input_format):
  beam_runner.run(
      create_data.build_retrieval_pipeline(
          input_files=inputs,
          input_format=input_format,
          output_files=outputs,
          config=tf_example_utils.RetrievalConversionConfig(
              vocab_file=FLAGS.vocab_file,
              max_seq_length=FLAGS.max_seq_length,
              max_column_id=FLAGS.max_column_id,
              max_row_id=FLAGS.max_row_id,
              strip_column_names=False,
              cell_trim_length=FLAGS.cell_trim_length,
              use_document_title=FLAGS.use_document_title,
          ),
          converter_impl=FLAGS.converter_impl,
      )).wait_until_finish()


def main(_):
  inputs, outputs = create_data_file_io.get_inputs_and_outputs(
      FLAGS.input_interactions_dir, FLAGS.output_dir)
  if not inputs:
    raise ValueError(f"Input dir is empty: '{FLAGS.input_interactions_dir}'")

  run(inputs, outputs, create_data.InputFormat.INTERACTION)

  if FLAGS.input_tables_dir is not None:
    table_inputs, table_outputs = create_data_file_io.get_inputs_and_outputs(
        FLAGS.input_tables_dir, FLAGS.output_dir)
    run(table_inputs, table_outputs, create_data.InputFormat.TABLE)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_interactions_dir")
  flags.mark_flag_as_required("max_column_id")
  flags.mark_flag_as_required("max_row_id")
  flags.mark_flag_as_required("max_seq_length")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("vocab_file")
  app.run(main)
