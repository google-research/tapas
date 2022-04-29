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
from tapas.scripts import preprocess_nq_utils

FLAGS = flags.FLAGS
TEST_PATH = 'tapas/scripts/testdata/'


def _build_table(columns, rows, **kwargs):
  table = interaction_pb2.Table()
  for column in columns:
    table.columns.add().text = column
  for row in rows:
    new_row = table.rows.add()
    for cell in row:
      new_row.cells.add().text = cell
  for key, value in kwargs.items():
    if key == 'table_id':
      table.table_id = value
    elif key == 'document_url':
      table.document_url = value
    else:
      raise ValueError(f'Unknown argument: {key}')
  return table


class PreprocessNQTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data_dir = TEST_PATH

  def _load_synthetic_lines(self):
    with open(os.path.join(self.test_data_dir, 'nq_raw_examples.txt'),
              'rb') as f:
      for line in f:
        yield line.decode('utf-8')

  def _load_html_table(self, name):
    with open(os.path.join(self.test_data_dir, name), 'rb') as f:
      return f.read()

  def test_parse_lines(self):
    lines = list(self._load_synthetic_lines())
    results = [preprocess_nq_utils.parse(line) for line in lines]
    results.sort(key=lambda r: r['example_id'])

    self.assertEqual(sum([result['contained'] for result in results]), 2)

    self.assertLen(results[0]['tables'], 3)
    self.assertLen(results[1]['tables'], 2)

    self.assertLen(results[1]['interactions'], 1)
    self.assertEqual(results[1]['interactions'][0].table.table_id,
                     'Kaa_7711FF1B1001D3C3')
    self.assertEqual([cell.text for cell in results[1]['tables'][0].columns], [
        '',
        '',
        '',
        'First appearance',
        'Last appearance',
        'Created by',
        '',
        'Species',
        'Gender',
    ])

    table = results[1]['tables'][0]
    self.assertEqual(table.table_id, 'Kaa_7711FF1B1001D3C3')
    self.assertEqual(table.document_title, 'Kaa')
    self.assertEqual(
        table.document_url,
        'https://en.wikipedia.org//w/index.php?title=Kaa&amp;oldid=820536956')
    self.assertEqual(preprocess_nq_utils.get_version(table), 820536956)
    rows = []
    for row in table.rows:
      row_values = []
      for cell in row.cells:
        row_values.append(cell.text)
      rows.append(row_values)
    logging.info(rows)
    self.assertEqual(rows, [[
        'Kaa',
        'The Jungle Book character',
        'Kaa (left), as illustrated in the 1895 edition of '
        'The Two Jungle Books',
        '"Kaa\'s Hunting"',
        '"The Spring Running"',
        'Rudyard Kipling',
        'Information',
        'Indian rock python',
        'Male',
    ]])

  @parameterized.parameters(
      ('table_00.html',),
      ('table_01.html',),
      ('table_02.html',),
  )
  def test_infobox(self, name):
    data = self._load_html_table(name)
    table = preprocess_nq_utils._parse_table(data)
    logging.info(table)
    if name == 'table_00.html':
      self.assertEqual(
          table,
          preprocess_nq_utils.Table(
              header=[
                  '',
                  '',
                  'Starring',
                  'Country of origin',
                  'No. of episodes',
                  '',
                  'Original network',
                  'Original release',
                  '',
                  '',
                  '',
              ],
              rows=[[
                  "Grey's Anatomy (season\xa014)",
                  'Promotional poster',
                  ('Ellen Pompeo Justin Chambers Chandra Wilson James Pickens, '
                   'Jr. Kevin McKidd Jessica Capshaw Sarah Drew Jesse Williams '
                   'Caterina Scorsone Camilla Luddington Kelly McCreary Jason '
                   'George Martin Henderson Giacomo Gianniotti'),
                  'United States',
                  '24',
                  'Release',
                  'ABC',
                  'September 28, 2017\xa0– May 17, 2018',
                  'Season chronology',
                  '←\xa0Previous Season 13',
                  "List of Grey's Anatomy episodes",
              ]]))
    elif name == 'table_01.html':
      self.assertEqual(
          table,
          preprocess_nq_utils.Table(
              header=[
                  'Metric diameter',
                  'US Knitting needle number',
                  'Crochet hook size',
              ],
              rows=[
                  ['2.25 mm', '1', 'B-1'],
                  ['2.75 mm', '2', 'C-2'],
                  ['3.25 mm', '3', 'D-3'],
                  ['3.5 mm', '4', 'E-4'],
                  ['3.75 mm', '5', 'F-5'],
                  ['4 mm', '6', 'G-6'],
                  ['4.5 mm', '7', '7'],
                  ['5 mm', '8', 'H-8'],
                  ['5.5 mm', '9', 'I-9'],
                  ['6 mm', '10', 'J-10'],
                  ['6.5 mm', '10.5', 'K-10.5'],
                  ['8 mm', '11', 'L-11'],
                  ['9 mm', '13', 'M/N-13'],
                  ['10 mm', '15', 'N/P-15'],
                  ['12.75 mm', '17', '—'],
                  ['15 mm', '19', 'P/Q'],
                  ['16 mm', '—', 'Q'],
                  ['19 mm', '35', 'S'],
                  ['25 mm', '50', 'U'],
              ]))
    elif name == 'table_02.html':
      self.assertEqual(
          table,
          preprocess_nq_utils.Table(
              header=[
                  'Year', 'Award', 'Category', 'Recipient', 'Result', 'Ref.'
              ],
              rows=[[
                  '2014', '15th Golden Trailer Awards',
                  'Best Summer 2014 Blockbuster Trailer',
                  'Godzilla "Ravaged/Event"', 'Won', '[177]'
              ],
                    [
                        'Best Action TV Spot', 'Godzilla "Fight"', 'Nominated',
                        '[177]', '', ''
                    ],
                    [
                        'Best Summer 2014 Blockbuster TV Spot',
                        'Godzilla "Fight"', 'Won', '[177]', '', ''
                    ],
                    [
                        'Best International Poster', 'Godzilla', 'Won', '[177]',
                        '', ''
                    ],
                    [
                        'Best Summer 2014 Blockbuster Poster', 'Godzilla',
                        'Won', '[177]', '', ''
                    ],
                    [
                        'Teen Choice Awards', 'Choice Movie: Action',
                        'Godzilla', 'Nominated', '[232]', ''
                    ],
                    [
                        'Choice Movie: Breakout Star', 'Elizabeth Olsen',
                        'Nominated', '', '', ''
                    ],
                    [
                        'Choice Movie: Hissy Fit',
                        'For the character, "Godzilla"', 'Nominated', '', '', ''
                    ],
                    [
                        'Hollywood Post Alliance',
                        'Outstanding Sound - Feature Film',
                        'Erik Aadahl, Ethan Van der Ryn, Tim LeBlanc, Gregg '
                        'Landaker, Rick Kline', 'Nominated', '', ''
                    ],
                    [
                        'World Soundtrack Academy', 'Film Composer of the Year',
                        'Alexandre Desplat (also for The Grand Budapest Hotel, '
                        'Marius, The Monuments Men, Philomena, Venus in Fur, '
                        'and Zulu)', 'Won', '', ''
                    ],
                    [
                        '22nd Annual Japan Cool Content Contribution Award', '',
                        'Alex Garcia', 'Won', '[233]', ''
                    ],
                    [
                        '2015', 'Houston Film Critics Society', 'Best Poster',
                        'Godzilla', 'Nominated', ''
                    ],
                    [
                        'Japan Academy Prize',
                        'Outstanding Foreign Language Film', 'Godzilla',
                        'Nominated', '[234]', ''
                    ],
                    [
                        '41st Saturn Awards', 'Best Science Fiction Film',
                        'Godzilla', 'Nominated', '[235]', ''
                    ],
                    [
                        'Best Music', 'Alexandre Desplat', 'Nominated', '', '',
                        ''
                    ]]))

  def test_ngrams(self):
    table = _build_table(
        columns=['Number', 'Name'],
        rows=[
            ['100,000', '15'],
            ['Zurich', 'Bern'],
        ])
    ngrams = list(
        preprocess_nq_utils._get_ngrams(
            table,
            header_weight=2,
            max_ngram_length=2,
        ))
    logging.info(ngrams)
    self.assertEqual(ngrams, [
        'number',
        'number name',
        'name',
        'number',
        'number name',
        'name',
        '100',
        '100 000',
        '000',
        '000 15',
        '15',
        'zurich',
        'zurich bern',
        'bern',
    ])

  def test_group_similar_tables(self):
    tables = [
        _build_table(
            columns=['A'], rows=[['a']], table_id='a', document_url='a'),
        _build_table(
            columns=['A'], rows=[['a']], table_id='b', document_url='b'),
        _build_table(
            columns=['B'], rows=[['a']], table_id='c', document_url='c'),
    ]
    groups = preprocess_nq_utils.group_similar_tables(
        tables,
        hash_dimension=16,
    )
    group_ids = [[table.table_id for table in group] for group in groups]
    self.assertEqual(group_ids, [['a', 'b'], ['c']])

  def test_group_similar_tables_same_version(self):
    tables = [
        _build_table(
            columns=['A'], rows=[['a']], table_id='a', document_url='a'),
        _build_table(
            columns=['A'], rows=[['a']], table_id='b', document_url='a'),
    ]
    groups = preprocess_nq_utils.group_similar_tables(
        tables,
        hash_dimension=16,
    )
    group_ids = [[table.table_id for table in group] for group in groups]
    self.assertEqual(group_ids, [['a'], ['b']])

  def test_group_similar_tables_same_id(self):
    tables = [
        _build_table(
            columns=['A'], rows=[['a']], table_id='a', document_url='a'),
        _build_table(
            columns=['A'], rows=[['a']], table_id='a', document_url='a'),
    ]
    groups = preprocess_nq_utils.group_similar_tables(
        tables,
        hash_dimension=16,
    )
    group_ids = [[table.table_id for table in group] for group in groups]
    self.assertEqual(group_ids, [['a', 'a']])


if __name__ == '__main__':
  absltest.main()
