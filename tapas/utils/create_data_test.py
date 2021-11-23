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
import string
import tempfile

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import mock
from tapas.protos import interaction_pb2
from tapas.retrieval import tf_example_utils as retrieval_utils
from tapas.utils import beam_runner
from tapas.utils import create_data
from tapas.utils import tf_example_utils
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

FLAGS = flags.FLAGS
TEST_PATH = 'tapas/utils/testdata/'

_ImplType = create_data.ConverterImplType
_InputFormat = create_data.InputFormat

_PretrainConfig = tf_example_utils.PretrainConversionConfig
_ClassifierConfig = tf_example_utils.ClassifierConversionConfig

_RESERVED_SYMBOLS = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EMPTY]')


def _read_examples(filepath):
  examples = []
  for value in tf.python_io.tf_record_iterator(filepath):
    example = tf.train.Example()
    example.ParseFromString(value)
    examples.append(example)
  return examples


def _set_mock_read(mock_read, output):

  def dummy_read(file_pattern, coder, validate):
    del file_pattern, coder, validate  # Unused.
    return beam.Create(output)

  mock_read.side_effect = dummy_read


class CreatePretrainingDataTest(parameterized.TestCase):

  def setUp(self):
    super(CreatePretrainingDataTest, self).setUp()

    self._test_dir = TEST_PATH
    self._temp_dir = tempfile.TemporaryDirectory()
    self._vocab_path = os.path.join(self._temp_dir.name, 'vocab.txt')
    self._output_path = self._temp_dir.name

  def tearDown(self):
    super(CreatePretrainingDataTest, self).tearDown()
    self._temp_dir.cleanup()

  def _create_vocab(self, vocab):
    with tf.gfile.Open(self._vocab_path, 'w') as input_handle:
      input_handle.write('\n'.join(vocab))

  @parameterized.parameters(
      (beam_runner.RunnerType.DIRECT, True),
      (beam_runner.RunnerType.DIRECT, False),
  )
  def test_end_to_end(self, runner_type, always_continue_cells):

    self._create_vocab(list(_RESERVED_SYMBOLS) + ['released'])

    pipeline = create_data.build_pretraining_pipeline(
        input_file=os.path.join(self._test_dir, 'pretrain_interactions.txtpb'),
        output_suffix='.tfrecord',
        output_dir=self._output_path,
        config=tf_example_utils.PretrainConversionConfig(
            vocab_file=self._vocab_path,
            max_seq_length=10,
            max_predictions_per_seq=10,
            random_seed=5,
            masked_lm_prob=0.5,
            max_column_id=3,
            max_row_id=3,
            min_question_length=1,
            max_question_length=4,
            always_continue_cells=always_continue_cells,
            strip_column_names=False),
        dupe_factor=2,
        min_num_columns=0,
        min_num_rows=0,
        num_corpus_bins=2,
    )

    beam_runner.run_type(pipeline, runner_type).wait_until_finish()

    for name in ['train', 'test']:
      examples = _read_examples(
          os.path.join(self._output_path, f'{name}.tfrecord'))
      self.assertNotEmpty(examples)

  @mock.patch.object(beam.io, 'ReadFromTFRecord')
  def test_end_to_end_multiple_interactions(self, mock_read):
    with tf.gfile.Open(os.path.join(self._test_dir,
                                    'interaction_01.pbtxt')) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())

    interactions = []
    for trial in range(100):
      table_id = f'table_id_{trial}'
      new_interaction = interaction_pb2.Interaction()
      new_interaction.CopyFrom(interaction)
      new_interaction.table.table_id = table_id
      new_interaction.id = table_id
      interactions.append(new_interaction)

    _set_mock_read(mock_read, interactions)

    self._create_vocab(
        list(_RESERVED_SYMBOLS) + list(string.ascii_lowercase) +
        ['##' + letter for letter in string.ascii_lowercase])

    pipeline = create_data.build_pretraining_pipeline(
        input_file='input.tfrecord',
        output_suffix='.tfrecord',
        output_dir=self._output_path,
        config=_PretrainConfig(
            vocab_file=self._vocab_path,
            max_seq_length=40,
            max_predictions_per_seq=10,
            random_seed=5,
            masked_lm_prob=0.5,
            max_column_id=5,
            max_row_id=5,
            min_question_length=5,
            max_question_length=10,
            always_continue_cells=True,
            strip_column_names=False,
        ),
        dupe_factor=1,
        min_num_columns=0,
        min_num_rows=0,
        num_random_table_bins=10,
        num_corpus_bins=100000,  # High number sends all examples to train set.
        add_random_table=True,
    )

    result = beam.runners.direct.direct_runner.DirectRunner().run(pipeline)
    result.wait_until_finish()

    counters = {
        metric_result.key.metric.name: metric_result.committed
        for metric_result in result.metrics().query()['counters']
    }

    self.assertEqual(
        counters, {
            'Examples': 100,
            'Examples with tables': 100,
            'Interactions': 100,
            'Interactions without random interaction': 11,
            'Question Length: < inf': 31,
            'Question Length: <= 10': 53,
            'Question Length: <= 7': 16,
            'Real Table Size: <= 8': 100,
            'Trimmed Table Size: <= 8': 100,
            'Column Sizes: <= 8': 100,
            'Row Sizes: <= 8': 100,
            'Table Token Sizes: <= 8': 100,
            'Inputs': 100,
        })

    output = _read_examples(os.path.join(self._output_path, 'train.tfrecord'))
    self.assertLen(output, 100)


COMP_INTERACTION = """
  id: "nt-3533-1"
  table: {
    columns: {
      text: "Attendance"
    }
    rows: {
      cells: {
        text: "76,194"
      }
    }
    rows: {
      cells: {
        text: "76,333"
      }
    }
    table_id: "table_csv/204_250.csv"
  }
  questions: {
    id: "nt-3533-1_1"
    original_text:
      "of these games, which ones had an attendance of greater than 70,000?"
    answer: {
      answer_coordinates: {
        row_index: 0
        column_index: 0
      }
    }
  }"""


class CreateClassifierDataTest(parameterized.TestCase):

  def setUp(self):
    super(CreateClassifierDataTest, self).setUp()

    self._test_dir = TEST_PATH
    self._temp_dir = tempfile.TemporaryDirectory()
    self._vocab_path = os.path.join(self._temp_dir.name, 'vocab.txt')
    self._output_path = os.path.join(self._temp_dir.name, 'output.tfrecord')
    self._create_vocab(list(_RESERVED_SYMBOLS) + list(string.ascii_lowercase))

  def tearDown(self):
    super(CreateClassifierDataTest, self).tearDown()
    self._temp_dir.cleanup()

  def _create_vocab(self, vocab):
    with tf.gfile.Open(self._vocab_path, 'w') as input_handle:
      input_handle.write('\n'.join(vocab))

  @parameterized.parameters((1,), (0,), (2,), (0.9,), (1.1,), (0.5,))
  def test_get_samples(self, rate):
    keys = map(str, range(100))
    if rate == int(rate):
      for key in keys:
        self.assertEqual(create_data._get_samples(key, rate), rate)
    else:
      self.assertAlmostEqual(
          sum(create_data._get_samples(key, rate) for key in keys) / 100,
          rate,
          places=1,
      )

  @parameterized.parameters((None, None, None, {
      'Conversion success': 2,
      'Input question': 2,
      'Question Ids added': 2,
      'Table Ids added': 1,
      'Relation Set Index: 1': 1,
      'Relation Set Index: 2': 2,
      'Relation Set Index: 4': 15,
      'Found answers: <= 1': 3,
      'Example emitted': 2,
  }), (None, 1, None, {
      'Input question': 2,
      'Conversion error': 2,
      'Too many columns': 2,
      'Question Ids added': 2,
      'Table Ids added': 1,
  }), (None, None, 1, {
      'Input question': 2,
      'Conversion error': 2,
      'Too many rows': 2,
      'Question Ids added': 2,
      'Table Ids added': 1,
  }), (1, None, None, {
      'Input question': 2,
      'Conversion error': 2,
      'Sequence too long': 2,
      'Question Ids added': 2,
      'Table Ids added': 1,
  }))
  @mock.patch.object(beam.io, 'ReadFromTFRecord')
  def test_gracefully_handle_big_examples(self, max_seq_length, max_column_id,
                                          max_row_id, expected_counters,
                                          mock_read):

    with tf.gfile.Open(os.path.join(self._test_dir,
                                    'interaction_02.pbtxt')) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())

    _set_mock_read(mock_read, [interaction])

    pipeline = create_data.build_classifier_pipeline(
        input_files=['input.tfrecord'],
        output_files=[self._output_path],
        config=_ClassifierConfig(
            vocab_file=self._vocab_path,
            max_seq_length=60 if max_seq_length is None else max_seq_length,
            max_column_id=5 if max_column_id is None else max_column_id,
            max_row_id=10 if max_row_id is None else max_row_id,
            strip_column_names=False,
            add_aggregation_candidates=False,
        ))

    result = beam.runners.direct.direct_runner.DirectRunner().run(pipeline)
    result.wait_until_finish()

    self.assertEqual(
        {
            metric_result.key.metric.name: metric_result.committed
            for metric_result in result.metrics().query()['counters']
        }, expected_counters)

    if max_seq_length is None and max_column_id is None and max_row_id is None:
      output = _read_examples(self._output_path)

      with tf.gfile.Open(os.path.join(self._test_dir,
                                      'tf_example_02.pbtxt')) as input_file:
        expected_example = text_format.ParseLines(input_file,
                                                  tf.train.Example())
      with tf.gfile.Open(
          os.path.join(self._test_dir,
                       'tf_example_02_conv.pbtxt')) as input_file:
        expected_conversational_example = text_format.ParseLines(
            input_file, tf.train.Example())

      self.assertLen(output, 2)

      actual_example = output[0]
      del actual_example.features.feature['column_ranks']
      del actual_example.features.feature['inv_column_ranks']
      del actual_example.features.feature['numeric_relations']
      del actual_example.features.feature['numeric_values']
      del actual_example.features.feature['numeric_values_scale']
      del actual_example.features.feature['question_id_ints']
      # assertEqual struggles with NaNs inside protos
      del actual_example.features.feature['answer']

      self.assertEqual(actual_example, expected_example)

      actual_example = output[1]
      del actual_example.features.feature['column_ranks']
      del actual_example.features.feature['inv_column_ranks']
      del actual_example.features.feature['numeric_relations']
      del actual_example.features.feature['numeric_values']
      del actual_example.features.feature['numeric_values_scale']
      del actual_example.features.feature['question_id_ints']
      # assertEqual struggles with NaNs inside protos
      del actual_example.features.feature['answer']

      self.assertEqual(actual_example, expected_conversational_example)

  @mock.patch.object(beam.io, 'ReadFromTFRecord')
  def test_numeric_relations(self, mock_read):
    input_file = 'interaction_00.pbtxt'
    expected_counters = {
        'Conversion success': 1,
        'Example emitted': 1,
        'Input question': 1,
        'Relation Set Index: 2': 5,
        'Relation Set Index: 4': 13,
        'Found answers: <= 4': 1,
    }

    with tf.gfile.Open(os.path.join(self._test_dir, input_file)) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())

    _set_mock_read(mock_read, [interaction])

    max_seq_length = 512

    pipeline = create_data.build_classifier_pipeline(
        input_files=['input.tfrecord'],
        output_files=[self._output_path],
        config=_ClassifierConfig(
            vocab_file=os.path.join(self._test_dir, 'vocab.txt'),
            max_seq_length=max_seq_length,
            max_column_id=512,
            max_row_id=512,
            strip_column_names=False,
            add_aggregation_candidates=False,
        ))

    result = beam.runners.direct.direct_runner.DirectRunner().run(pipeline)
    result.wait_until_finish()

    self.assertEqual(
        {
            metric_result.key.metric.name: metric_result.committed
            for metric_result in result.metrics().query()['counters']
        }, expected_counters)

    output = _read_examples(self._output_path)

    self.assertLen(output, 1)
    actual_example = output[0]

    self.assertIn('numeric_relations', actual_example.features.feature.keys())
    relations = actual_example.features.feature[
        'numeric_relations'].int64_list.value

    with tf.gfile.Open(os.path.join(self._test_dir, 'vocab.txt')) as vocab_file:
      vocab = [line.strip() for line in vocab_file]
    inputs = actual_example.features.feature['input_ids'].int64_list.value
    pairs = [(vocab[input_id], relation)
             for (input_id, relation) in zip(inputs, relations)
             if input_id > 0]
    logging.info('pairs: %s', pairs)
    self.assertSequenceEqual(pairs,
                             [('[CLS]', 0), ('which', 0), ('cities', 0),
                              ('had', 0), ('less', 0), ('than', 0), ('2', 0),
                              (',', 0), ('000', 0), ('pass', 0), ('##en', 0),
                              ('##ge', 0), ('##rs', 0), ('?', 0), ('[SEP]', 0),
                              ('ran', 0), ('##k', 0), ('city', 0), ('pass', 0),
                              ('##en', 0), ('##ge', 0), ('##rs', 0), ('ran', 0),
                              ('##ki', 0), ('##ng', 0), ('air', 0), ('##li', 0),
                              ('##ne', 0),
                              ('1', 4), ('united', 0), ('states', 0), (',', 0),
                              ('los', 0), ('angeles', 0), ('14', 2), (',', 2),
                              ('7', 2), ('##4', 2), ('##9', 2), ('[EMPTY]', 0),
                              ('al', 0), ('##as', 0), ('##ka', 0), ('air', 0),
                              ('##li', 0), ('##ne', 0), ('##s', 0), ('2', 4),
                              ('united', 0), ('states', 0), (',', 0), ('h', 0),
                              ('##ous', 0), ('##ton', 0), ('5', 2), (',', 2),
                              ('4', 2), ('##6', 2), ('##5', 2), ('[EMPTY]', 0),
                              ('united', 0), ('e', 0), ('##x', 0), ('##p', 0),
                              ('##re', 0), ('##s', 0), ('##s', 0), ('3', 4),
                              ('canada', 0), (',', 0), ('c', 0), ('##al', 0),
                              ('##ga', 0), ('##ry', 0), ('3', 2), (',', 2),
                              ('7', 2), ('##6', 2), ('##1', 2), ('[EMPTY]', 0),
                              ('air', 0), ('t', 0), ('##ra', 0), ('##ns', 0),
                              ('##a', 0), ('##t', 0), (',', 0), ('west', 0),
                              ('##j', 0), ('##et', 0), ('4', 4), ('canada', 0),
                              (',', 0), ('s', 0), ('##as', 0), ('##ka', 0),
                              ('##to', 0), ('##on', 0), ('2', 2), (',', 2),
                              ('28', 2), ('##2', 2), ('4', 0), ('[EMPTY]', 0),
                              ('5', 4), ('canada', 0), (',', 0), ('van', 0),
                              ('##co', 0), ('##u', 0), ('##ve', 0), ('##r', 0),
                              ('2', 2), (',', 2), ('10', 2), ('##3', 2),
                              ('[EMPTY]', 0), ('air', 0), ('t', 0), ('##ra', 0),
                              ('##ns', 0), ('##a', 0), ('##t', 0), ('6', 4),
                              ('united', 0), ('states', 0), (',', 0), ('p', 0),
                              ('##h', 0), ('##o', 0), ('##en', 0), ('##i', 0),
                              ('##x', 0), ('1', 4), (',', 4), ('8', 4),
                              ('##2', 4), ('##9', 4), ('1', 0), ('us', 0),
                              ('air', 0), ('##w', 0), ('##a', 0), ('##y', 0),
                              ('##s', 0), ('7', 4), ('canada', 0), (',', 0),
                              ('to', 0), ('##ro', 0), ('##nt', 0), ('##o', 0),
                              ('1', 4), (',', 4), ('20', 4), ('##2', 4),
                              ('1', 0), ('air', 0), ('t', 0), ('##ra', 0),
                              ('##ns', 0), ('##a', 0), ('##t', 0), (',', 0),
                              ('can', 0), ('##j', 0), ('##et', 0), ('8', 4),
                              ('canada', 0), (',', 0), ('ed', 0), ('##m', 0),
                              ('##on', 0), ('##ton', 0), ('11', 4), ('##0', 4),
                              ('[EMPTY]', 0), ('[EMPTY]', 0), ('9', 4),
                              ('united', 0), ('states', 0), (',', 0), ('o', 0),
                              ('##a', 0), ('##k', 0), ('##land', 0), ('10', 4),
                              ('##7', 4), ('[EMPTY]', 0), ('[EMPTY]', 0)])

  @mock.patch.object(beam.io, 'ReadFromTFRecord')
  def test_candidates(self, mock_read):

    with tf.gfile.Open(os.path.join(self._test_dir,
                                    'interaction_03.pbtxt')) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())

    _set_mock_read(mock_read, [interaction])
    max_seq_length = 15

    tf_example_utils._MAX_NUM_ROWS = 4
    tf_example_utils._MAX_NUM_CANDIDATES = 10

    pipeline = create_data.build_classifier_pipeline(
        input_files=['input.tfrecord'],
        output_files=[self._output_path],
        config=_ClassifierConfig(
            vocab_file=os.path.join(self._test_dir, 'vocab.txt'),
            max_seq_length=max_seq_length,
            max_column_id=4,
            max_row_id=4,
            strip_column_names=False,
            add_aggregation_candidates=True,
        ),
    )

    result = beam.runners.direct.direct_runner.DirectRunner().run(pipeline)
    result.wait_until_finish()

    output = _read_examples(self._output_path)

    with tf.gfile.Open(os.path.join(self._test_dir,
                                    'tf_example_03.pbtxt')) as input_file:
      expected_example = text_format.ParseLines(input_file, tf.train.Example())

    actual_example = output[0]
    logging.info('%s', actual_example)
    # assertEqual struggles with NaNs inside protos
    del actual_example.features.feature['numeric_values']
    self.assertEqual(actual_example, expected_example)

  @mock.patch.object(beam.io, 'ReadFromTFRecord')
  def test_tfrecord_io(self, mock_read):
    """Reads from TFRecord and writes to TFRecord."""

    with tf.gfile.Open(os.path.join(self._test_dir,
                                    'interaction_03.pbtxt')) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())

    def dummy_read(file_pattern, coder, validate):
      del file_pattern, coder, validate  # Unused.
      return beam.Create([interaction])

    mock_read.side_effect = dummy_read
    max_seq_length = 15

    pipeline = create_data.build_classifier_pipeline(
        input_files=['input.tfrecord'],
        output_files=[self._output_path],
        config=_ClassifierConfig(
            vocab_file=os.path.join(self._test_dir, 'vocab.txt'),
            max_seq_length=max_seq_length,
            max_column_id=4,
            max_row_id=4,
            strip_column_names=False,
            add_aggregation_candidates=False,
        ))

    result = beam.runners.direct.direct_runner.DirectRunner().run(pipeline)
    result.wait_until_finish()

    output = []
    for value in tf.python_io.tf_record_iterator(self._output_path):
      example = tf.train.Example()
      example.ParseFromString(value)
      output.append(example)

    self.assertLen(output, 1)
    sid = output[0].features.feature['segment_ids']
    self.assertLen(sid.int64_list.value, max_seq_length)


class CreateRetrievalDataTest(parameterized.TestCase):

  def setUp(self):
    super(CreateRetrievalDataTest, self).setUp()

    self._test_dir = 'tapas/retrieval/testdata'
    self._temp_dir = tempfile.TemporaryDirectory()
    self._vocab_path = os.path.join(self._temp_dir.name, 'vocab.txt')
    self._output_path = os.path.join(self._temp_dir.name, 'output.tfrecord')

  def tearDown(self):
    super(CreateRetrievalDataTest, self).tearDown()
    self._temp_dir.cleanup()

  def _create_vocab(
      self,
      vocab,
  ):
    vocab_path = self._vocab_path
    with tf.gfile.Open(vocab_path, 'w') as input_handle:
      input_handle.write('\n'.join(vocab))

  @parameterized.parameters(
      (_InputFormat.INTERACTION, _ImplType.PYTHON),
      (_InputFormat.TABLE, _ImplType.PYTHON),
  )
  @mock.patch.object(beam.io, 'ReadFromTFRecord')
  def test_end_to_end(self, input_format, impl, mock_read):
    self._create_vocab(list(_RESERVED_SYMBOLS))

    with tf.gfile.Open(
        os.path.join(self._test_dir,
                     'retrieval_interaction.pbtxt')) as input_file:
      interaction = text_format.ParseLines(input_file,
                                           interaction_pb2.Interaction())
    if input_format == _InputFormat.INTERACTION:
      samples = [interaction]
    elif input_format == _InputFormat.TABLE:
      samples = [interaction.table]
    else:
      raise ValueError(f'Unknown format: {input_format}')

    _set_mock_read(mock_read, samples)

    pipeline = create_data.build_retrieval_pipeline(
        input_files=['input.tfrecord'],
        input_format=input_format,
        output_files=[self._output_path],
        config=retrieval_utils.RetrievalConversionConfig(
            vocab_file=self._vocab_path,
            max_seq_length=15,
            max_column_id=5,
            max_row_id=5,
            strip_column_names=False),
        converter_impl=impl,
    )
    result = beam.runners.direct.direct_runner.DirectRunner().run(pipeline)
    result.wait_until_finish()
    counters = {
        metric_result.key.metric.name: metric_result.committed
        for metric_result in result.metrics().query()['counters']
    }

    if input_format == _InputFormat.INTERACTION:
      self.assertEqual(counters, {
          'Input question': 1,
          'Conversion success': 1,
      })
    else:
      self.assertEqual(
          counters, {
              'Input question': 1,
              'Conversion success': 1,
              'Fake Questions added for table only example': 1,
          })

    output = _read_examples(self._output_path)
    self.assertLen(output, 1)

if __name__ == '__main__':
  absltest.main()
