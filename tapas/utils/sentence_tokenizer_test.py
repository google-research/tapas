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

from absl.testing import absltest
from tapas.utils import sentence_tokenizer


class SentenceTokenizerTest(absltest.TestCase):

  def test_sentence_tokenizer(self):
    sentences = sentence_tokenizer.tokenize(
        'A sentence about dogs. Dogs are cute amimals. Cats are OK as well')
    self.assertEqual(sentences, [
        'A sentence about dogs.',
        'Dogs are cute amimals.',
        'Cats are OK as well',
    ])


if __name__ == '__main__':
  absltest.main()
