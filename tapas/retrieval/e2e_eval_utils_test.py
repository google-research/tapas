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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tapas.experiments import prediction_utils
from tapas.protos import interaction_pb2
from tapas.retrieval import e2e_eval_utils

from google.protobuf import text_format
from official.nlp.bert import tokenization

_RESERVED_SYMBOLS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EMPTY]")


def _create_vocab(vocab_file, tokens):
  with open(vocab_file, "tw") as output_file:
    for token in list(_RESERVED_SYMBOLS) + tokens:
      output_file.write(token)
      output_file.write("\n")


class E2EEvalUtilsTest(parameterized.TestCase):

  def test_empty(self):
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, "vocab.txt")
      _create_vocab(vocab_file, [])
      result = e2e_eval_utils._evaluate_retrieval_e2e(
          vocab_file,
          interaction_list=[],
          predictions=[],
      )
    logging.info("result: %s", result)
    for name, value in result.to_dict().items():
      if name in {
          "answer_accuracy_table",
          "answer_accuracy_passage",
          "answer_token_f1_table",
          "answer_token_f1_passage",
      }:
        self.assertIsNone(value)
      else:
        self.assertEqual(value, 1.0)

  def test_empty_predictions(self):
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, "vocab.txt")
      _create_vocab(vocab_file, [])
      interactions = [
          text_format.Parse(
              """
            table {
              rows {
                cells { text: "ANSWER" }
                cells { text: "OTHER" }
              }
            }
            questions {
              id: "example_id-0_0"
              answer {
                class_index: 1
                answer_texts: "ANSWER"
              }
            }
          """, interaction_pb2.Interaction()),
          text_format.Parse(
              """
            table {
              rows {
                cells { text: "ANSWER" }
                cells { text: "OTHER" }
              }
            }
            questions {
              id: "example_id-1_0"
              answer {
                class_index: 1
                answer_texts: "ANSWER"
              }
            }
          """, interaction_pb2.Interaction())
      ]
      result = e2e_eval_utils._evaluate_retrieval_e2e(
          vocab_file,
          interactions,
          predictions=[],
      )

    logging.info("result: %s", result)
    for name, value in result.to_dict().items():
      if name in {
          "answer_accuracy_table",
          "answer_accuracy_passage",
          "answer_token_f1_table",
          "answer_token_f1_passage",
      }:
        self.assertIsNone(value)
      elif name in [
          "table_precision",
          "answer_precision",
          "answer_token_precision",
      ]:
        self.assertEqual(value, 1.0)
      else:
        self.assertEqual(value, 0.0)

  @parameterized.parameters(
      ("[6]",),
      ("[6, 1]",),
      ("[1]",),
  )
  def test_simple(self, answer):
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, "vocab.txt")
      _create_vocab(vocab_file, ["answer"])
      interactions = [
          text_format.Parse(
              """
            table {
              rows {
                cells { text: "ANSWER UNKNOWN" }
              }
            }
            questions {
              id: "example_id-0_0"
              answer {
                class_index: 1
                answer_texts: "OTHER"
              }
              alternative_answers {
                answer_texts: "ANSWER"
              }
            }
          """, interaction_pb2.Interaction()),
          text_format.Parse(
              """
            table {
              rows {
                cells { text: "ANSWER UNKNOWN" }
              }
            }
            questions {
              id: "example_id-1_0"
              answer {
                class_index: 1
                answer_texts: "ANSWER"
              }
            }
          """, interaction_pb2.Interaction())
      ]
      predictions = [
          {
              "question_id": "example_id-0_0",
              "logits_cls": "2",
              "answer": "[0]",
          },
          {
              "question_id": "example_id-1_0",
              "logits_cls": "11",
              "answer": answer,
          },
      ]
      result = e2e_eval_utils._evaluate_retrieval_e2e(
          vocab_file,
          interactions,
          predictions,
      )

    logging.info("result: %s", result)
    for name, value in result.to_dict().items():
      if name in {
          "answer_accuracy_table",
          "answer_accuracy_passage",
          "answer_token_f1_table",
          "answer_token_f1_passage",
      }:
        self.assertIsNone(value)
      elif name in [
          "table_accuracy",
          "table_recall",
      ]:
        continue
      elif name in [
          "table_precision",
      ]:
        self.assertEqual(value, 1.0)
      else:
        if answer == "[6]":
          self.assertEqual(value, 1.0)
        elif answer == "[6, 1]":
          values = {
              "answer_accuracy": 0.0,
              "answer_token_recall": 1.0,
              "answer_precision": 0.0,
              "answer_token_f1": 0.6666666666666666,
              "answer_token_precision": 0.5,
              "oracle_answer_token_f1": 0.6666666666666666,
              "oracle_answer_accuracy": 0.0,
          }
          self.assertEqual(value, values[name])
        elif answer == "[1]":
          self.assertEqual(value, 0.0)

  def test_simple_token_answer(self):
    with tempfile.TemporaryDirectory() as input_dir:
      vocab_file = os.path.join(input_dir, "vocab.txt")
      _create_vocab(vocab_file, ["answer", "wrong"])
      interactions = [
          text_format.Parse(
              """
            table {
              rows {
                cells { text: "WRONG WRONG" }
              }
            }
            questions {
              id: "example_id-0_0"
              answer {
                class_index: 1
                answer_texts: "OTHER"
              }
              alternative_answers {
                answer_texts: "ANSWER"
              }
            }
          """, interaction_pb2.Interaction()),
          text_format.Parse(
              """
            table {
              rows {
                cells { text: "ANSWER WRONG" }
              }
            }
            questions {
              id: "example_id-1_0"
              answer {
                class_index: 1
                answer_texts: "ANSWER"
              }
            }
          """, interaction_pb2.Interaction())
      ]
      predictions = [
          {
              "question_id":
                  "example_id-0_0",
              "logits_cls":
                  "0",
              "answers":
                  prediction_utils.token_answers_to_text([
                      prediction_utils.TokenAnswer(
                          row_index=-1,
                          column_index=-1,
                          begin_token_index=-1,
                          end_token_index=-1,
                          token_ids=[1],
                          score=10.0,
                      )
                  ]),
          },
          {
              "question_id":
                  "example_id-1_0",
              "logits_cls":
                  "1",
              "answers":
                  prediction_utils.token_answers_to_text([
                      prediction_utils.TokenAnswer(
                          row_index=0,
                          column_index=0,
                          begin_token_index=-1,
                          end_token_index=-1,
                          token_ids=[6, 7],
                          score=10.0,
                      )
                  ]),
          },
      ]
      result = e2e_eval_utils._evaluate_retrieval_e2e(
          vocab_file,
          interactions,
          predictions,
      )
      logging.info("result: %s", result.to_dict())
      self.assertEqual(
          {
              "answer_accuracy": 0.0,
              "answer_precision": 0.0,
              "answer_token_f1": 0.6666666666666666,
              "answer_token_precision": 0.5,
              "answer_token_recall": 1.0,
              "oracle_answer_accuracy": 0.0,
              "oracle_answer_token_f1": 0.6666666666666666,
              "table_accuracy": 1.0,
              "table_precision": 1.0,
              "table_recall": 1.0,
              "answer_accuracy_table": None,
              "answer_accuracy_passage": None,
              "answer_token_f1_table": None,
              "answer_token_f1_passage": None,
          }, result.to_dict())


class DetokenizerTest(parameterized.TestCase):

  def _get_vocab_file(self, vocab_file, vocab):
    _create_vocab(vocab_file, vocab)
    return vocab_file

  @parameterized.parameters(
      ("a", "a"), ("bb B", "B"), ("a Ab", "Ab"), ("3.5", "3.5"),
      ("A A A", "A", "A"),
      ("Instituto Reação", "instituto reacao", "Instituto Reação"),
      ("bbd", "bb", "bbd"))
  def test_single_cell(self, cell, text, exepected=None):
    with tempfile.TemporaryDirectory() as temp_dir:
      vocab_file = os.path.join(temp_dir, "vocab.txt")
      self._get_vocab_file(
          vocab_file,
          [
              "a",
              "b",
              "bb",
              "##b",
              "3",
              ".",
              "5",
              "insti",
              "##tuto",
              "reacao",
              "##d",
          ],
      )
      detokenizer = e2e_eval_utils.DeTokenizer(vocab_file)
      tokenizer = tokenization.FullTokenizer(
          vocab_file,
          do_lower_case=True,
          split_on_punc=True,
      )
    table = interaction_pb2.Table()
    table.rows.add().cells.add().text = cell
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    actual = detokenizer.detokenize(
        table,
        token_ids,
    )
    if exepected is None:
      exepected = text
    self.assertEqual(actual, exepected)


if __name__ == "__main__":
  absltest.main()
