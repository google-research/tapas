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
"""Functions for evaluating end to end retrieval QA.

   Partially based on SQUAD
   (https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
"""

import ast
import collections
import dataclasses
import os
import re
import string
from typing import Any, Iterable, List, Mapping, Optional, Set, Text, Tuple

from absl import logging
import numpy as np
from tapas.experiments import prediction_utils as xprediction_utils
from tapas.protos import interaction_pb2
from tapas.scripts import prediction_utils
from tapas.utils import hybridqa_utils
from tapas.utils import text_utils

from official.nlp.bert import tokenization


@dataclasses.dataclass(frozen=True)
class EvalResult:
  """Retrieval + QA eval results."""
  table_accuracy: float
  table_precision: float
  table_recall: float
  answer_accuracy: float
  answer_accuracy_table: Optional[float]
  answer_accuracy_passage: Optional[float]
  answer_precision: float
  answer_token_precision: float
  answer_token_recall: float
  answer_token_f1: float
  answer_token_f1_passage: Optional[float]
  answer_token_f1_table: Optional[float]
  oracle_answer_token_f1: float
  oracle_answer_accuracy: float

  def to_dict(self):
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ScoredCandidate:
  interaction: interaction_pb2.Interaction
  answer: Text
  score: float


@dataclasses.dataclass(frozen=True)
class Wordpiece:
  start: int
  end: int
  token_id: int


def _find(subpieces, pieces):
  """Returns the index for which subpieces starts as a sublist of pieces or -1."""
  for i in range(len(pieces) - len(subpieces) + 1):
    if pieces[i:i + len(subpieces)] == subpieces:
      return i
  return -1


class DeTokenizer:
  """Detokenizes a list of pieces in the context of a table.

    This is needed because the Standard Bert tokenization is lossy.
    For example, it will add spaces around white space (3.5 -> 3 . 5) and
    remove diacritics.

    This class tries to find a list of pieces in a tokenized table and returns
    the smallest byte span in the original cell text that yields the pieces
    when tokenized.
  """

  def __init__(self, vocab_file, do_lower_case=True, split_on_punc=True):
    self._whitespace_tokenizer = tokenization.BasicTokenizer(
        do_lower_case=False, split_on_punc=False)
    self._punctuation_tokenizer = tokenization.BasicTokenizer(
        do_lower_case=False, split_on_punc=split_on_punc)
    self._full_tokenizer = tokenization.FullTokenizer(
        vocab_file, do_lower_case=do_lower_case, split_on_punc=split_on_punc)
    self._vocab = list(self._full_tokenizer.vocab.keys())

  def _tokenize_text(self, text):
    """Tokenizes text by white-space, punctuation and then into word pieces."""
    tokens = []
    whitespace_normalized_text = ""
    for whitespace_token in self._whitespace_tokenizer.tokenize(text):
      for punctuation_token in self._punctuation_tokenizer.tokenize(
          whitespace_token):
        start = len(whitespace_normalized_text)
        end = start + len(punctuation_token)
        for wordpiece in self._full_tokenizer.tokenize(punctuation_token):
          token_id = self._full_tokenizer.vocab[wordpiece]
          tokens.append(Wordpiece(start=start, end=end, token_id=token_id))
        whitespace_normalized_text += punctuation_token
      whitespace_normalized_text += " "
    return tokens, whitespace_normalized_text

  def _convert_ids_to_text(
      self,
      token_ids,
  ):
    """Maps token ids to text."""
    tokens = [self._vocab[token_id] for token_id in token_ids]
    result = []
    for token in tokens:
      if token.startswith("##"):
        result.append(token[2:])
      else:
        if result:
          result.append(" ")
        result.append(token)
    return "".join(result)

  def _get_unique_texts(
      self,
      table,
  ):
    """Yields all cell texts of `table` without duplicates."""
    texts = set()
    for row in table.rows:
      for cell in row.cells:
        if cell.text not in texts:
          texts.add(cell.text)
          yield cell.text

  def _detokenize_text(
      self,
      text,
      token_ids,
  ):
    """"Searches for 'token_ids' in the tokenzied 'text'."""
    tokens, normalized_text = self._tokenize_text(text)
    logging.info(
        "tokens: %s normalized text: %s token ids: %s",
        tokens,
        normalized_text,
        token_ids,
    )
    start_token_index = _find(token_ids, [token.token_id for token in tokens])
    if start_token_index < 0:
      return None
    end_token_index = start_token_index + len(token_ids) - 1
    start_token = tokens[start_token_index]
    end_token = tokens[end_token_index]
    return normalized_text[start_token.start:end_token.end]

  def detokenize(
      self,
      table,
      pieces,
  ):
    """Detokenizes a list of pieces in a given table."""
    for text in self._get_unique_texts(table):
      detokenized_text = self._detokenize_text(text, pieces)
      if detokenized_text is not None:
        return detokenized_text
    # Tokens couldn't be found in the table.
    # We back off to direct detokenization
    text = self._convert_ids_to_text(pieces)
    logging.warning(
        "Backing off to direct detokenization: '%s' ('%s')",
        pieces,
        text,
    )
    return text

  def detokenize_token_answer(
      self,
      table,
      token_answer,
  ):
    """Tries to search for the token ids in the specified cell."""
    column = token_answer.column_index
    row = token_answer.row_index
    pieces = token_answer.token_ids
    if 0 <= column < len(table.columns) and 0 <= row < len(table.rows):
      cell = table.rows[row].cells[column]
      text = self._detokenize_text(cell.text, pieces)
      if text is not None:
        return text
      logging.error(
          "Token answer not contained in specified cell: '%s' answer: '%s'",
          cell.text,
          self._convert_ids_to_text(pieces),
      )
    else:
      logging.error(
          "Invalid cell coordinates: "
          "column: %d row: %d vs table columns: %d table rows: %d",
          column,
          row,
          len(table.columns),
          len(table.rows),
      )
    # Back-off to searching for the correct cell.
    return self.detokenize(table, pieces)


def _get_example_id(question_id):
  example_id, _, _ = text_utils.parse_question_id(question_id)
  return example_id


def _guess_vocab_file(interaction_file):
  interaction_dir = os.path.dirname(interaction_file)
  vocab_dir = os.path.dirname(interaction_dir)
  vocab_file = os.path.join(vocab_dir, "vocab.txt")
  return vocab_file


def _is_valid_answer(token_ids):
  if not token_ids:
    return False
  # Token id 1 correspond to the [EMPTY] token.
  if len(token_ids) == 1 and token_ids[0] == 1:
    return False
  return True


def _get_reconstructed_answer_text(
    detokenizer,
    table,
    prediction,
):
  """Reconstructs answer text using the detokenizer."""
  if "answers" in prediction:
    token_answers = xprediction_utils.token_answers_from_text(
        prediction["answers"])
    # Remove empty answers.
    answers = [
        detokenizer.detokenize_token_answer(table, token_answer)
        for token_answer in token_answers
        if _is_valid_answer(token_answer.token_ids)
    ]
    if not answers:
      return None
    return _list_to_answer(answers)
  # Support for legacy prediction files.
  if "answer" in prediction:
    answer_ids = ast.literal_eval(prediction["answer"])
    if not _is_valid_answer(answer_ids):
      return None
    return detokenizer.detokenize(table, answer_ids)
  raise ValueError("No answer value in prediction.")


def _get_scored_candidates(
    detokenizer,
    interactions,
    predictions_by_qid,
):
  """Returns highest scoring interaction with answers."""
  for interaction in interactions:
    qid = interaction.questions[0].id
    prediction = predictions_by_qid.get(qid, None)
    if prediction is not None:
      answer = _get_reconstructed_answer_text(
          detokenizer,
          interaction.table,
          prediction,
      )
      if answer is not None:
        # Assign score as 0.0 when "logits_cls" not present in predictions.
        score = float(prediction.get("logits_cls", 0.0))
        yield ScoredCandidate(
            interaction=interaction,
            answer=answer,
            score=score,
        )


def _is_correct_table(interaction):
  return interaction.questions[0].answer.class_index == 1


def _list_to_answer(texts):
  return " ".join(text for text in sorted(texts) if text)


def _get_answer_texts(
    interactions):
  """Returns answer text."""
  answer_texts = set()
  for interaction in interactions:
    question = interaction.questions[0]
    for answer in [question.answer] + list(question.alternative_answers):
      text = _list_to_answer(answer.answer_texts)
      if text:
        answer_texts.add(text)
  return answer_texts


def _remove_articles(text):
  regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
  return re.sub(regex, " ", text)


def _fix_whitespace(text):
  return " ".join(text.split())


def _remove_punctuation(text):
  return text.translate(str.maketrans("", "", string.punctuation))


def _lower(text):
  return text.lower()


def _normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  return _fix_whitespace(_remove_articles(_remove_punctuation(_lower(s))))


def _tokenize(s):
  if not s:
    return []
  return _normalize_answer(s).split()


def _compute_exact(a_gold, a_pred):
  return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))


def _compute_f1(a_gold, a_pred):
  """Computes token-based precision, recall and f-score."""
  gold_toks = _tokenize(a_gold)
  pred_toks = _tokenize(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  precision = num_same / len(pred_toks) if pred_toks else 1.0
  recall = num_same / len(gold_toks) if gold_toks else 1.0
  if precision + recall < 1.0e-5:
    return 0.0, 0.0, 0.0
  f1 = (2.0 * precision * recall) / (precision + recall)
  return precision, recall, f1


def _mean(values):
  if not values:
    return 1.0
  return np.mean(values)


def _get_best_metrics(
    reference_answer_texts,
    best_answer_text,
):
  """Evaluates prediction against all references."""
  metrics = []
  for reference_answer_text in reference_answer_texts:
    pr, rc, f1 = _compute_f1(
        reference_answer_text,
        best_answer_text,
    )
    em = _compute_exact(
        reference_answer_text,
        best_answer_text,
    )
    metrics.append({
        "exact": em,
        "f1": f1,
        "precision": pr,
        "recall": rc,
    })
  return max(metrics, key=lambda m: m["f1"])


def _get_question(
    candidates,):
  if not candidates:
    return None
  candidate = candidates[0]
  if not candidate.questions:
    return None
  return candidate.questions[0].original_text


def _get_oracle_metrics(
    candidates,
    reference_answer_texts,
    metric,
):
  """Get best value for metric."""
  if not candidates:
    return 0.0
  values = []
  for candidate in candidates:
    metrics = _get_best_metrics(reference_answer_texts, candidate.answer)
    values.append(metrics[metric])
  return max(values)


def is_table_answer(question_id,
                    references):
  return False if references is None else question_id in references["table"]


def is_passage_answer(question_id,
                      references):
  return False if references is None else question_id in references["passage"]


def _evaluate_retrieval_e2e(
    vocab_file,
    interaction_list,
    predictions,
    references = None):
  """Computes e2e retrieval-QA metrics."""
  detokenizer = DeTokenizer(vocab_file)
  interactions = collections.defaultdict(list)
  for interaction in interaction_list:
    qid = interaction.questions[0].id
    interactions[_get_example_id(qid)].append(interaction)

  predictions_by_qid = {}
  for prediction in predictions:
    qid = prediction["question_id"]
    predictions_by_qid[qid] = prediction

  num_correct_tables = 0
  num_pred_tables = 0

  token_precisions = []
  token_recalls = []
  token_f1_scores = []
  token_f1_scores_table = []
  token_f1_scores_passage = []
  token_correct = []
  token_correct_table = []
  token_correct_passage = []

  oracle_token_f1_scores = []
  oracle_token_correct = []

  num_pred_answers = 0

  num_examples_with_reference_table = 0

  num_errors_logged = 0

  for candidates in interactions.values():
    logging.log_every_n(
        logging.INFO,
        "Processed: %5d / %5d",
        1000,
        len(token_correct),
        len(interactions),
    )
    reference_answer_texts = _get_answer_texts(candidates)
    if not reference_answer_texts:
      token_precisions.append(0.0)
      token_recalls.append(0.0)
      token_f1_scores.append(0.0)
      token_correct.append(0.0)
      continue

    for interaction in candidates:
      if _is_correct_table(interaction):
        num_examples_with_reference_table += 1
        break

    results = list(
        _get_scored_candidates(
            detokenizer,
            candidates,
            predictions_by_qid,
        ))
    if not results:
      logging.log_every_n(logging.INFO, "No candidate.", 1000)
      best_answer_text = ""
      example_id = "NOT_FOUND"
    else:
      best_result = max(results, key=lambda result: result.score)
      example_id = text_utils.get_example_id(best_result.interaction.id)
      best_answer_text = best_result.answer
      num_pred_tables += 1
      # We reconstruct the orginal text. This is can be problematic for example
      # when the answer contains the "[UNK]" token.
      if best_answer_text:
        num_pred_answers += 1

      if _is_correct_table(best_result.interaction):
        num_correct_tables += 1

    metrics = _get_best_metrics(reference_answer_texts, best_answer_text)
    token_precisions.append(metrics["precision"])
    token_recalls.append(metrics["recall"])
    token_f1_scores.append(metrics["f1"])
    token_correct.append(metrics["exact"])

    if is_passage_answer(example_id, references):
      token_f1_scores_passage.append(metrics["f1"])
      token_correct_passage.append(metrics["exact"])

    if is_table_answer(example_id, references):
      token_f1_scores_table.append(metrics["f1"])
      token_correct_table.append(metrics["exact"])

    oracle_token_correct.append(
        _get_oracle_metrics(
            results,
            reference_answer_texts,
            metric="exact",
        ))
    oracle_token_f1_scores.append(
        _get_oracle_metrics(
            results,
            reference_answer_texts,
            metric="f1",
        ))

    if metrics["exact"] != 1 and num_errors_logged < 100:
      num_errors_logged += 1
      logging.info("question: '%s' references: %s prediction: '%s'",
                   _get_question(candidates), reference_answer_texts,
                   best_answer_text)

  precision = _mean(token_precisions)
  recall = _mean(token_recalls)
  f1_score = _mean(token_f1_scores)
  answer_accuracy = _mean(token_correct)

  if references is not None:
    f1_score_table = _mean(token_f1_scores_table)
    f1_score_passage = _mean(token_f1_scores_passage)
    answer_accuracy_table = _mean(token_correct_table)
    answer_accuracy_passage = _mean(token_correct_passage)
  else:
    f1_score_table = None
    f1_score_passage = None
    answer_accuracy_table = None
    answer_accuracy_passage = None

  oracle_f1_score = _mean(oracle_token_f1_scores)
  oracle_answer_accuracy = _mean(oracle_token_correct)

  total = len(interactions)
  table_accuracy = num_correct_tables / total if total > 0 else 1.0

  table_precision = 1.0
  if num_pred_tables > 0:
    table_precision = num_correct_tables / num_pred_tables

  table_recall = 1.0
  if num_examples_with_reference_table > 0:
    table_recall = num_correct_tables / num_examples_with_reference_table

  answer_precision = 1.0
  if num_pred_answers > 0:
    answer_precision = sum(token_correct) / num_pred_answers

  if total:
    oracle_table_accuracy = num_examples_with_reference_table / total
    logging.info("oracle table accuracy: %s", oracle_table_accuracy)

  return EvalResult(
      table_accuracy=table_accuracy,
      table_precision=table_precision,
      table_recall=table_recall,
      answer_accuracy=answer_accuracy,
      answer_accuracy_table=answer_accuracy_table,
      answer_accuracy_passage=answer_accuracy_passage,
      answer_precision=answer_precision,
      answer_token_precision=precision,
      answer_token_recall=recall,
      answer_token_f1=f1_score,
      answer_token_f1_table=f1_score_table,
      answer_token_f1_passage=f1_score_passage,
      oracle_answer_token_f1=oracle_f1_score,
      oracle_answer_accuracy=oracle_answer_accuracy)


def evaluate_retrieval_e2e(
    interaction_file,
    prediction_file,
    references_file = None,
    vocab_file = None,
):
  """Computes e2e retrieval-QA metrics."""
  vocab_file = vocab_file or _guess_vocab_file(interaction_file)
  references = None
  if references_file is not None:
    references = hybridqa_utils.get_hybridqa_references(references_file)
  logging.info("Vocab file: %s ", vocab_file)
  logging.info("Read: %s ", interaction_file)
  interactions = prediction_utils.iterate_interactions(interaction_file)
  logging.info("Read: %s ", prediction_file)
  predictions = prediction_utils.iterate_predictions(prediction_file)
  return _evaluate_retrieval_e2e(vocab_file, interactions, predictions,
                                 references)


# TODO(eisenjulian): merge this with _evaluate_retrieval_e2e() or pull the
# common logic in a separate function and call that in _evaluate_retrieval_e2e.
def generate_hybridqa_codalab_predictions(
    interaction_file,
    prediction_file):
  """Generates Codaab prediction files for HybridQA Competition.

  This function generates the json prediction files used to submit to HybridQA
  competition hosted on Codalab. (go/hybridqa-competition)

  Args:
    interaction_file: A TF record file containing the examples as interactions.
    prediction_file: A TSV file that is the output of the table-classifier
      predict job on the input interactions.

  Yields:
    An iterable of json serializable python dicts.
  """
  vocab_file = _guess_vocab_file(interaction_file)
  logging.info("Vocab file: %s ", vocab_file)
  logging.info("Read: %s ", interaction_file)
  interactions = prediction_utils.iterate_interactions(interaction_file)
  logging.info("Read: %s ", prediction_file)
  predictions = prediction_utils.iterate_predictions(prediction_file)

  detokenizer = DeTokenizer(vocab_file)

  interactions_by_qid = collections.defaultdict(list)
  for interaction in interactions:
    qid = interaction.questions[0].id
    interactions_by_qid[_get_example_id(qid)].append(interaction)

  predictions_by_qid = {}
  for prediction in predictions:
    qid = prediction["question_id"]
    # TODO(eisenjulian): Select the best answer using model scores.
    predictions_by_qid[qid] = prediction

  for qid, candidates in interactions_by_qid.items():
    answer_text = ""
    results = list(
        _get_scored_candidates(
            detokenizer,
            candidates,
            predictions_by_qid,
        ))
    example_id = text_utils.get_example_id(qid)
    if results:
      best_result = max(results, key=lambda result: result.score)
      answer_text = best_result.answer

    yield {"question_id": example_id, "pred": answer_text}
