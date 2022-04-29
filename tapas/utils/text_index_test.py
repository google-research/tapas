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
"""Tests for tapas.utils.google.text_index."""

from absl.testing import absltest
from tapas.utils import text_index


class TextIndexTest(absltest.TestCase):

  def test_num_results(self):
    index = text_index.TextIndex([
        'A sentence about dogs.',
        'Dogs are cute amimals',
        'Cats are OK as well',
    ], lambda x: x)
    results = index.search('Two dogs.', num_results=1, retrieval_threshold=0.0)
    self.assertLen(results, 1)
    self.assertEqual(results[0].document, 'A sentence about dogs.')

  def test_bad_results_filtered(self):
    index = text_index.TextIndex([
        'A sentence about dogs and cats.',
        'Dogs are cute amimals. I like dogs.',
        'Cats are OK as well',
    ], lambda x: x)
    results = index.search('Two dogs.', num_results=3, retrieval_threshold=0.0)
    self.assertLen(results, 2)
    first, second = results
    self.assertEqual(first.document, 'Dogs are cute amimals. I like dogs.')
    self.assertEqual(second.document, 'A sentence about dogs and cats.')
    self.assertGreater(first.score, second.score)


if __name__ == '__main__':
  absltest.main()
