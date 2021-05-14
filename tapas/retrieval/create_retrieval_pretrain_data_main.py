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

from typing import Iterable, Optional, Text, Tuple

from absl import app
from absl import flags
import apache_beam as beam
import nltk
from tapas.protos import interaction_pb2
from tapas.utils import beam_runner
from tapas.utils import beam_utils
from tapas.utils import create_data
from tapas.utils import sentence_tokenizer


flags.DEFINE_string("inputs", None, "Interaction tables.")
flags.DEFINE_string("output_dir", None, "Directory with outputs.")

FLAGS = flags.FLAGS


_NS = "main"
_TITLE_QUESTION_ID = "TITLE"


def get_title(interaction):
  for question in interaction.questions:
    if question.id == _TITLE_QUESTION_ID:
      return question.original_text
  return None


def _to_retrieval_interaction_fn(
    interaction
):
  """Converts pretraining interaction to retrieval interaction."""
  beam.metrics.Metrics.counter(_NS, "Interactions").inc()
  title = get_title(interaction)
  if title is None or not title:
    beam.metrics.Metrics.counter(_NS, "Interactions without title").inc()
    return

  interaction = beam_utils.rekey(interaction)
  interaction.table.document_title = title

  word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()

  for question in interaction.questions:
    if question.id == _TITLE_QUESTION_ID:
      continue

    text = question.original_text

    for paragraph in text.split("\n"):
      for sentence in sentence_tokenizer.tokenize(paragraph):
        sentence = sentence.strip()
        if not sentence:
          continue

        beam.metrics.Metrics.counter(_NS, "Sentences").inc()
        num_tokens = word_tok.tokenize(sentence)
        if len(num_tokens) < 4:
          beam.metrics.Metrics.counter(_NS, "Sentence too short").inc()
          continue
        if len(num_tokens) > 32:
          beam.metrics.Metrics.counter(_NS, "Sentence too long").inc()
          continue

        new_interaction = interaction_pb2.Interaction()
        new_interaction.CopyFrom(interaction)
        del new_interaction.questions[:]
        new_question = new_interaction.questions.add()
        new_question.id = hex(
            beam_utils.to_numpy_seed(obj=(interaction.id, sentence)))
        new_interaction.id = new_question.id
        new_question.original_text = sentence

        beam.metrics.Metrics.counter(_NS, "Examples").inc()
        yield new_interaction.id, new_interaction


def build_pipeline(inputs, output_dir):
  """Builds the pipeline."""

  def _pipeline(root):

    interactions = (
        create_data.read_interactions(root, inputs, name="input")
        | "DropKey" >> beam.Map(beam.Values())
        | "ToRetrievalExample" >> beam.FlatMap(_to_retrieval_interaction_fn)
        | "Reshuffle" >> beam.transforms.util.Reshuffle())

    # We expect ~37,568,664 interactions by taking 1 / 5000 for test test we
    # get a reasonable test set size of ~7513.
    beam_utils.split_by_table_id_and_write(
        interactions,
        output_dir,
        train_suffix="@*",
        test_suffix="@*",
        num_splits=5000,
    )

  return _pipeline


def main(_):
  beam_runner.run(
      build_pipeline(inputs=FLAGS.inputs,
                     output_dir=FLAGS.output_dir)).wait_until_finish()


if __name__ == "__main__":
  flags.mark_flag_as_required("inputs")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
