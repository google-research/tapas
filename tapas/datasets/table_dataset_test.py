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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tapas.datasets import table_dataset
from tapas.datasets import table_dataset_test_utils
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class TableDatasetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("train_with_aggregation", 10, 10, table_dataset.TableTask.CLASSIFICATION,
       True, False, False, False, False),
      ("train_with_weak_supervision", 10, 10,
       table_dataset.TableTask.CLASSIFICATION, True, True, False, False, False),
      ("train", 10, 10, table_dataset.TableTask.CLASSIFICATION, False, False,
       False, False, False),
      ("pretrain", 10, 10, table_dataset.TableTask.PRETRAINING, False, False,
       False, False, False),
      ("predict", 10, 10, table_dataset.TableTask.CLASSIFICATION, False, False,
       True, False, False),
      ("predict_with_aggregation", 10, 10,
       table_dataset.TableTask.CLASSIFICATION, True, False, True, False, False),
      ("predict_with_weak_supervision", 10, 10,
       table_dataset.TableTask.CLASSIFICATION, True, True, True, False, False),
      ("train_with_candidate_answers", 10, 10,
       table_dataset.TableTask.CLASSIFICATION, True, True, False, True, False),
      ("train_with_classification", 10, 10,
       table_dataset.TableTask.CLASSIFICATION, True, False, False, False, True),
      ("predict_with_classification", 10, 10,
       table_dataset.TableTask.CLASSIFICATION, True, False, True, False, True),
  )
  def test_parse_table_examples(self, max_seq_length, max_predictions_per_seq,
                                task_type, add_aggregation_function_id,
                                add_answer, include_id, add_candidate_answers,
                                add_classification_labels):
    logging.info("Setting random seed to 42")
    np.random.seed(42)
    max_num_candidates = 10
    values = table_dataset_test_utils.create_random_example(
        max_seq_length,
        max_predictions_per_seq,
        task_type,
        add_aggregation_function_id,
        add_classification_labels,
        add_answer,
        include_id,
        vocab_size=10,
        segment_vocab_size=3,
        num_columns=3,
        num_rows=2,
        add_candidate_answers=add_candidate_answers,
        max_num_candidates=max_num_candidates)
    example = table_dataset_test_utils.make_tf_example(values)

    params = {}
    parse_fn = table_dataset.parse_table_examples(
        max_seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        task_type=task_type,
        add_aggregation_function_id=add_aggregation_function_id,
        add_classification_labels=add_classification_labels,
        add_answer=add_answer,
        include_id=include_id,
        add_candidate_answers=add_candidate_answers,
        max_num_candidates=max_num_candidates,
        params=params)
    features = parse_fn(example.SerializeToString())

    with self.cached_session() as sess:
      features_vals = sess.run(features)

    for value in values:
      if value == "can_indexes":
        continue
      if values[value].dtype == np.float32 or values[value].dtype == np.int32:
        np.testing.assert_almost_equal(features_vals[value], values[value])
      else:  # Handle feature as string.
        np.testing.assert_equal(features_vals[value], values[value])

    if add_candidate_answers:
      self.assertEqual(features_vals["can_label_ids"].dtype, np.int32)
      self.assertAllEqual(features_vals["can_label_ids"].shape,
                          [max_num_candidates, max_seq_length])

      # The total number of label_ids set to 1 must match the total number
      # of indices.
      num_indices = len(values["can_indexes"])
      self.assertEqual(features_vals["can_label_ids"].sum(), num_indices)

      # Check that the correct indices are set to 1.
      cand_id = 0
      cand_start = 0
      for i in range(len(values["can_indexes"])):
        while i - cand_start >= values["can_sizes"][cand_id]:
          cand_id += 1
          cand_start = i
        token_id = values["can_indexes"][i]
        self.assertEqual(features_vals["can_label_ids"][cand_id, token_id], 1)


if __name__ == "__main__":
  absltest.main()
