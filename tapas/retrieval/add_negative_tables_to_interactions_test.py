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
from typing import List, Text

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import mock
from tapas.protos import interaction_pb2
from tapas.protos import negative_retrieval_examples_pb2
from tapas.retrieval import add_negative_tables_to_interactions
from tapas.utils import beam_runner
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
TEST_PATH = "tapas/retrieval/testdata/"
_NeuralRetrievalExamples = negative_retrieval_examples_pb2.NegativeRetrievalExamples


def _get_table(table_id):
  return interaction_pb2.Table(
      columns=[
          interaction_pb2.Cell(text="Position"),
          interaction_pb2.Cell(text="Player"),
          interaction_pb2.Cell(text="Team"),
      ],
      rows=[
          interaction_pb2.Cells(cells=[
              interaction_pb2.Cell(text="1"),
              interaction_pb2.Cell(text="player 1"),
              interaction_pb2.Cell(text="team 1"),
          ]),
          interaction_pb2.Cells(cells=[
              interaction_pb2.Cell(text="2"),
              interaction_pb2.Cell(text="player 2"),
              interaction_pb2.Cell(text="team 2"),
          ]),
          interaction_pb2.Cells(cells=[
              interaction_pb2.Cell(text="1"),
              interaction_pb2.Cell(text="player 3"),
              interaction_pb2.Cell(text="team 2"),
          ]),
      ],
      table_id=table_id,
  )


def _get_interaction(interaction_id, question_id,
                     table):
  interaction = interaction_pb2.Interaction(
      id=interaction_id,
      questions=[
          interaction_pb2.Question(
              id=question_id,
              text="",
              original_text="What position does the player who played for team 1?",
              answer=interaction_pb2.Answer(
                  answer_coordinates=[
                      interaction_pb2.AnswerCoordinate(
                          row_index=0, column_index=0),
                  ],
                  answer_texts=["first_answer", "second_answer"],
              ),
          ),
      ],
  )
  interaction.table.CopyFrom(table)
  return interaction


class AddNegativeTablesToInteractionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = TEST_PATH

  def _verify_negative_table(self, output,
                             interaction_id, expected_table_id,
                             expected_rank, expected_score):
    ne_ext = _NeuralRetrievalExamples.negative_retrieval_examples_ext
    examples = [e for e in output if e.id == interaction_id]
    self.assertLen(examples, 1)
    example = examples[0]
    questions = example.questions
    self.assertLen(questions, 1)
    nr_examples = questions[0].Extensions[ne_ext].examples
    logging.info(nr_examples)
    self.assertLen(nr_examples, 2)
    nr_example = nr_examples[0]
    self.assertEqual(nr_example.rank, expected_rank)
    self.assertEqual(
        nr_example.type,
        negative_retrieval_examples_pb2.NegativeRetrievalExample.Type.BASELINE)
    self.assertEqual(nr_example.table.table_id, expected_table_id)
    self.assertEqual(nr_example.score, expected_score)

  @parameterized.parameters(
      (beam_runner.RunnerType.DIRECT,),
  )
  @mock.patch.object(beam.io, "ReadFromTFRecord")
  def test_flume_add_negative_tables(
      self,
      runner_type,
      mock_read,
  ):  # pylint: disable=g-doc-args
    r"""Tests the flume pipeline."""
    input_json_file = "neural_retrieval_00.jsonl"
    table_1 = _get_table("table_1")
    interaction_1 = _get_interaction("interaction_1", "q_1_0", table_1)
    table_2 = _get_table("table_2")
    interaction_2 = _get_interaction("interaction_2", "q_2_0", table_2)
    table_3 = _get_table("table_3")

    def dummy_read(file_pattern, coder, validate):
      del coder, validate  # Unused.
      if "tables" in file_pattern:
        return beam.Create([table_1, table_2, table_3])
      else:
        return beam.Create([interaction_1, interaction_2])

    mock_read.side_effect = dummy_read

    with tempfile.TemporaryDirectory() as temp_dir:
      output_path = os.path.join(temp_dir, "output.tfrecord")
      pipeline = add_negative_tables_to_interactions.add_negative_tables_to_interactions(
          max_num_negatives=None,
          input_interactions_files=["input.tfrecord"],
          input_tables_file="tables.tfrecord",
          input_json_files=[os.path.join(self._test_dir, input_json_file)],
          output_files=[output_path])
      beam_runner.run_type(pipeline, runner_type).wait_until_finish()

      output = []
      for value in tf.python_io.tf_record_iterator(output_path):
        example = interaction_pb2.Interaction()
        example.ParseFromString(value)
        output.append(example)
        logging.info(example)

      self.assertLen(output, 2)
      self._verify_negative_table(
          output=output,
          interaction_id="interaction_1",
          expected_table_id=table_3.table_id,
          expected_rank=1,
          expected_score=8.0)
      self._verify_negative_table(
          output=output,
          interaction_id="interaction_2",
          expected_table_id=table_1.table_id,
          expected_rank=1,
          expected_score=14.0)


if __name__ == "__main__":
  absltest.main()
