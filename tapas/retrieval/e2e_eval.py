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
"""Runs e2e eval on a pair of interactions / predctions."""

from absl import app
from absl import flags

from tapas.retrieval import e2e_eval_utils

flags.DEFINE_string("interaction_file", None, "TFRecord of interactions.")
flags.DEFINE_string("prediction_file", None, "Predictions in TSV format.")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  result = e2e_eval_utils.evaluate_retrieval_e2e(
      FLAGS.interaction_file,
      FLAGS.prediction_file,
  )
  print(result)


if __name__ == "__main__":
  flags.mark_flag_as_required("interaction_file")
  flags.mark_flag_as_required("prediction_file")
  app.run(main)
