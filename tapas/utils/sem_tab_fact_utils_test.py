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
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import interaction_pb2
from tapas.utils import sem_tab_fact_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS
TEST_PATH = 'tapas/utils/testdata/'


class GetTableDimensionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = TEST_PATH

  @parameterized.parameters(
      (sem_tab_fact_utils.Version.V1),
      (sem_tab_fact_utils.Version.V2),
  )
  def test_process_doc(self, version):
    interactions = list(
        sem_tab_fact_utils._process_doc(
            os.path.join(
                self.test_data_dir,
                'sem_tab_fact_20502.xml',
            ),
            version,
        ))

    if version == sem_tab_fact_utils.Version.V1:
      name = 'sem_tab_fact_20502_interaction.txtpb'
    elif version == sem_tab_fact_utils.Version.V2:
      name = 'sem_tab_fact_20502_interaction_v2.txtpb'
    else:
      raise ValueError(f'Unsupported version: {version.name}')
    interaction_file = os.path.join(self.test_data_dir, name)
    with open(interaction_file) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())
    self.assertLen(interactions, 4)
    logging.info(interactions[0])
    self.assertEqual(interactions[0], interaction)
    questions = [
        (  # pylint: disable=g-complex-comprehension
            i.questions[0].id,
            i.questions[0].original_text,
            i.questions[0].answer.class_index,
        ) for i in interactions
    ]
    self.assertEqual(questions, [
        (
            'sem_tab_fact_20502_Table_2_2_0',
            'At the same time, these networks often occur in tandem at the firm level.',
            1,
        ),
        (
            'sem_tab_fact_20502_Table_2_3_0',
            'For each network interaction, there is considerable variation both across and within countries.',
            1,
        ),
        (
            'sem_tab_fact_20502_Table_2_5_0',
            'The n value is same for Hong Kong and Malaysia.',
            0,
        ),
        (
            'sem_tab_fact_20502_Table_2_8_0',
            'There are 9 different types country in the given table.',
            1,
        ),
    ])


if __name__ == '__main__':
  absltest.main()
