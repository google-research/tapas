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
"""Helper methods for the HybridQA dataset' reading component."""

import collections
import dataclasses
import enum
import itertools
import json
import os
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Text, Tuple, Union
import urllib.parse

from absl import logging
import numpy as np
from tapas.experiments.prediction_utils import token_answers_from_text
from tapas.experiments.prediction_utils import TokenAnswer
from tapas.protos import annotated_text_pb2
from tapas.protos import interaction_pb2
from tapas.scripts import prediction_utils
from tapas.utils import hybridqa_utils
from tapas.utils import pretrain_utils
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf


_Split = hybridqa_utils.Split
_CoordinateType = hybridqa_utils.CoordinateType
_annotated_cell = annotated_text_pb2.AnnotatedText.annotated_cell_ext
_annotation_descriptions = annotated_text_pb2.AnnotationDescription.annotation_descriptions_ext
_ZIP_NAME = 'WikiTables-WithLinks'
_CONFIG_NAME = 'hybridqa_rc_config.json'


@dataclasses.dataclass(frozen=True)
class Prediction:
  """Dataclass to represent each row in a tsv prediction file."""
  id: Text
  question_id: Text
  annotator: Text
  position: int
  answer_coordinates: Sequence[Tuple[int, int]]
  answers: List[TokenAnswer]
  token_probabilities: Optional[List[Tuple[int, int, float]]] = None
  logits_cls: float = 0.0

  @property
  def example_id(self):
    return self.id

  @property
  def interaction_id(self):
    return text_utils.get_interaction_id(self.question_id)


def get_predictions(prediction_file):
  """Yields an iterable of Prediction objects from a tsv prediction file."""
  fn_map = {
      'logits_cls':
          float,
      'position':
          int,
      'answer_coordinates':
          lambda x: list(prediction_utils.parse_coordinates(x)),
      'answers':
          token_answers_from_text,
      'token_probabilities':
          json.loads,
  }
  for prediction_dict in prediction_utils.iterate_predictions(prediction_file):
    for key in tuple(prediction_dict.keys()):
      fn = fn_map.get(key, lambda x: x)
      prediction_dict[key] = fn(prediction_dict[key])
    yield Prediction(**prediction_dict)


def get_table_dimensions(
    table):
  """Returns size of input table as a pair of ints —— (n_rows, n_columns).

  Args:
    table: Input table, either a json mapping as in the original dataset, or a
      parsed interaction.Table proto.
  """
  if isinstance(table, interaction_pb2.Table):
    return len(table.rows), len(table.columns)

  elif isinstance(table, Mapping):
    try:
      assert isinstance(table['data'], List)
      assert isinstance(table['header'], List)
      return len(table['data']), len(table['header'])
    except KeyError as e:
      raise ValueError(
          f"Invalid table json. Key not present: '{e.args[0]}'") from e
    except AssertionError as e:
      raise ValueError(
          "Invalid table json: Values for keys 'data' and 'header' should be a list."
      ) from e

  else:
    raise TypeError(
        f"Invalid type for table: '{type(table)}'. It must be either a dict or interaction_pb2.Table"
    )


class NoisyCellStrategy(enum.Enum):
  """Enum class for representing Noisy-cells selection strategy."""
  # Does not sample any erroneous cell coordinate
  NONE = 'none'

  # Samples from all possible cell coordinates except the true coordinates.
  FULL_RANDOM = 'full'

  # Samples from the set of cell coordinates that share either the row index
  # or col index with any of the true coordinates but not both.
  SELECTIVE_RANDOM = 'selective'

  # Samples from the set of cell coordinates that share only the col index
  # with any of the true coordinates.
  COLUMN_CONFINED_RANDOM = 'col_confined'




@dataclasses.dataclass(frozen=True)
class HybridQaRcConfig:
  """Data class for HybridQA RC dataset creation configuration.

  Attributes:
    single_cell_examples: (member) If True, an example can have multiple
      interactions each having single-cell table.
    use_original_coordinates: (member) If True, uses cell coordinates present in
      the `answer-node` field of the dataset example as against finding an
      exhaustive list of coordinates.
    skip_ambiguous_examples: (member) If True, skips the examples while creating
      training interactions that have answer-texts in multiple cell coordinates.
    noisy_cell_strategy: (member) Strategy for including some noisy
      cells---cells that doesn't contain answer-text---along with the answer
      cells when creating interaction tables.
    num_predictions_to_keep: (member) Number of cell predictions to use.
    interactions_dir: (member) Directory of interactions files.
    hybridqa_prediction_filepath: (member) The filepaths of the output
      prediction files.
    interactions_dirs: (property) Directory of interactions files for each run.
    hybridqa_prediction_filepaths: (property) The filepaths of the output
      prediction files for each run.
  """

  single_cell_examples: bool = False
  use_original_coordinates: bool = False
  skip_ambiguous_examples: bool = False
  noisy_cell_strategy: NoisyCellStrategy = NoisyCellStrategy.NONE
  num_predictions_to_keep: int = 1
  interactions_dir: Optional[Text] = None
  hybridqa_prediction_filepath: Optional[Mapping[Text, Text]] = None

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a config from a Python dictionary of parameters."""
    json_object = dict(json_object)
    if 'noisy_cell_strategy' in json_object:
      strategy_name = json_object['noisy_cell_strategy']
      if strategy_name is None:
        json_object['noisy_cell_strategy'] = NoisyCellStrategy.NONE
      if isinstance(strategy_name, str):
        json_object['noisy_cell_strategy'] = NoisyCellStrategy(strategy_name)
      elif not isinstance(strategy_name, NoisyCellStrategy):
        raise TypeError(
            "'noisy_cell_strategy' field in input dict 'json_object'"
            'must be an instance of str or NoisyCellStrategy.')
    return HybridQaRcConfig(**json_object)

  @classmethod
  def from_json(cls, json_str):
    """Constructs a config from a json dict string."""
    return cls.from_dict(json.loads(json_str))

  @property
  def hybridqa_prediction_filepaths(self):
    """Returns a dict of {(split, run_id): prediction_filename}."""
    return {(split, 0): prediction_file_path for split, prediction_file_path
            in self.hybridqa_prediction_filepath.items()}

  @property
  def interactions_dirs(self):
    """Returns interactions directory."""
    return {0: self.interactions_dir}


def _parse_answer_cell(cell,
                       text,
                       links,
                       descriptions,
                       *,
                       url_unquote = True):
  """Parse text, links and its description in a table cell.

  Args:
    cell: Input table cell
    text: Cell text in the table
    links: An iterable of links of form '/wiki/<identifier>'
    descriptions: A string map of link -> description
    url_unquote: A boolean argument to signal whether to unquote the input
      links. This argument must be set to False if the input links are already
      url-unquoted. [default = True]
  """
  cell.text = text
  for link in collections.OrderedDict.fromkeys(links):
    cell_annotation = cell.Extensions[_annotated_cell].annotations.add()
    cell_annotation.identifier = (
        urllib.parse.unquote(link) if url_unquote else link)
    if cell_annotation.identifier not in descriptions:
      raise ValueError(f'{link} not found in requests')
    description = descriptions[cell_annotation.identifier]
    cell.text += f' : {description}'


def _init_answer_table(idx, title,
                       url):
  """Initializes an empty Table proto with index and title."""
  return interaction_pb2.Table(
      table_id=str(idx), document_title=title, document_url=url)


def get_texts_and_links(table_dict, coord):
  row_index, column_index = coord
  return table_dict['data'][row_index][column_index]


def _parse_answer_table(
    json_dict,
    descriptions,
    coordinates,
):
  """Converts a Table in JSON format to an expanded Table proto."""
  answer_table = _init_answer_table(json_dict['uid'], json_dict['title'],
                                    json_dict['url'])
  row = answer_table.rows.add()
  for coords in coordinates:
    text, links = get_texts_and_links(json_dict, coords)
    column = answer_table.columns.add()
    column.text = ''
    _parse_answer_cell(row.cells.add(), text, links, descriptions)

  return answer_table


def _parse_answer_tables(
    json_dict,
    descriptions,
    coordinates,
):
  """Converts a Table in JSON format to an iterable of expanded Table protos."""
  for coords in coordinates:
    answer_table = _init_answer_table(json_dict['uid'], json_dict['title'],
                                      json_dict['url'])
    row = answer_table.rows.add()
    column = answer_table.columns.add()
    column.text = ''
    text, links = get_texts_and_links(json_dict, coords)
    _parse_answer_cell(row.cells.add(), text, links, descriptions)
    yield answer_table


def _create_answer_table(
    original_table,
    descriptions,
    answer_coordinates,
):
  """Converts a HybridQA Table to an expanded HybridQA RC Table."""
  answer_table = interaction_pb2.Table(
      table_id=original_table.table_id,
      document_title=original_table.document_title,
      document_url=original_table.document_url)
  row = answer_table.rows.add()
  for coords in answer_coordinates:
    row_index, col_index = coords
    answer_table.columns.add().text = ''
    original_cell = original_table.rows[row_index].cells[col_index]
    links = [
        annotation.identifier
        for annotation in original_cell.Extensions[_annotated_cell].annotations
    ]
    _parse_answer_cell(
        row.cells.add(),
        original_cell.text,
        links,
        descriptions,
        url_unquote=False)
  return answer_table


def generate_sample_space(
    strategy, n_rows, n_cols,
    true_coordinates):
  """Generates a sample space of cell coordinates for a noisy cell selection strategy.

  Args:
    strategy: Noisy-cells selection strategy.
    n_rows: Number of rows in the table.
    n_cols: Number of columns in the table.
    true_coordinates: True cell coordinates in the table.

  Returns:
    All possible cell coordinates iterable for the input strategy.
  """
  true_coordinates_set = frozenset(true_coordinates)

  is_noisy = lambda c: c not in true_coordinates_set

  if strategy == NoisyCellStrategy.FULL_RANDOM:
    filter_func = is_noisy

  elif strategy == NoisyCellStrategy.SELECTIVE_RANDOM:
    row_set = {i for i, j in true_coordinates_set}
    col_set = {j for i, j in true_coordinates_set}

    filter_func = lambda c: (c[0] in row_set or c[1] in col_set) and is_noisy(c)

  elif strategy == NoisyCellStrategy.COLUMN_CONFINED_RANDOM:
    col_set = {j for i, j in true_coordinates_set}
    filter_func = lambda c: c[1] in col_set and is_noisy(c)

  elif strategy == NoisyCellStrategy.NONE:
    filter_func = lambda c: False

  else:
    raise ValueError(f'Invalid NoisyCellStrategy: {strategy}')

  return filter(filter_func, itertools.product(range(n_rows), range(n_cols)))


def sample_noisy_coordinates(
    n_rows, n_cols, answer_coordinates,
    strategy, n_samples,
    random_state):
  """Returns a random sampling of cell coordinates of the table.

  The function first tries to sample without replacement from a generated sample
  space of noisy coordinates. However, if the number of desired samples is
  higher than the size of the sample space, it falls back to sampling
  with-replacement. Finally, the function returns a tuple: a list of sampled
  noisy coordinates and a boolean variable signaling whether the sampling was
  done with (True) or without (False) replacement.

  Args:
    n_rows: Number of rows in the table.
    n_cols: Number of columns in the table.
    answer_coordinates: Iterable of cell coordinates which contains the
      answer-text, either in cell text or in the linked description.
    strategy: Strategy used for sampling coordinates.
    n_samples: Number of cell coordinates sampled.
    random_state: a RandomState object to randomly sample from a sample space.
  """

  if n_rows == 0:
    raise ValueError('n_rows must be positive integer.')

  if n_cols == 0:
    raise ValueError('n_cols must be positive integer.')

  noisy_sample_space = list(
      generate_sample_space(strategy, n_rows, n_cols, answer_coordinates))

  # Sample with replacement if sample space not large enough
  replace = n_samples > len(noisy_sample_space)

  if strategy == NoisyCellStrategy.NONE or not noisy_sample_space:
    return [], replace

  chosen_indices = random_state.choice(
      range(len(noisy_sample_space)), n_samples, replace=replace)

  noisy_coordinates = [noisy_sample_space[i] for i in chosen_indices]

  return [*noisy_coordinates], replace


def _parse_answer_interactions(
    table_json,
    descriptions,
    example,
    *,
    single_cell_examples = False,
    use_original_coordinates = True,
    noisy_cell_strategy = NoisyCellStrategy.NONE
):
  """Converts a single example to an interaction with a single question.

  Args:
    table_json: Table in JSON-format mapping.
    descriptions: The Wikipedia intro for each entity in the Table annotations.
    example: Question parsed from input JSON file.
    single_cell_examples: Generate multiple single-celled table interactions for
      each example (interaction) if this is set True.
    use_original_coordinates: Only use the coordinates of answer-text present in
      the dataset if this argument is set True, else it finds a more exhaustive
      set of coordinates where the answer-text is present --- either in the cell
      text, or in the description.
    noisy_cell_strategy: Determines the strategy for using noisy cells, which
      doesn't contain the answer-text along with the ground truth cells. If
      argument is set None, noisy cells are not included in the table.

  Yields:
    Interaction proto.
  """

  interaction = interaction_pb2.Interaction()

  # We append -0 that corresponds to position annotator field
  interaction.id = f"{example['question_id']}/{0}-0"
  desc_map = interaction.Extensions[_annotation_descriptions].descriptions
  for key, value in descriptions.items():
    desc_map[key] = value

  question = interaction.questions.add()
  # We append _0 that corresponds to SQA position field
  question.id = f'{interaction.id}_0'
  question.original_text = example['question']

  # Reference answer for the question. The test set answers are hidden.
  if 'answer-text' in example:
    question.answer.answer_texts.append(example['answer-text'])

    if use_original_coordinates:
      coordinates = hybridqa_utils.find_dataset_coordinates(example)
    else:
      original_table = hybridqa_utils.parse_table(table_json, descriptions)
      coordinates, _, _ = hybridqa_utils.find_answer_coordinates(
          example['answer-text'], original_table, desc_map)

    answer_coordinates = [*coordinates]

    random_seed = pretrain_utils.to_numpy_seed(interaction.id)
    random_state = np.random.RandomState(random_seed)
    n_noisy_samples = len(answer_coordinates)

    n_rows, n_columns = get_table_dimensions(table_json)
    noisy_coordinates, with_replacement = sample_noisy_coordinates(
        n_rows, n_columns, answer_coordinates, noisy_cell_strategy,
        n_noisy_samples, random_state)

    if with_replacement:
      logging.warning(
          "n_samples=%d higher than %s sample space for example_id %s. Sampled 'with-replacement' here.",
          n_noisy_samples, noisy_cell_strategy, example['question_id'])

    selected_coordinates = answer_coordinates + noisy_coordinates
    random_state.shuffle(selected_coordinates)

    if single_cell_examples:
      answer_tables = _parse_answer_tables(table_json, descriptions,
                                           selected_coordinates)
      for table_idx, answer_table in enumerate(answer_tables):
        new_interaction = interaction_pb2.Interaction()
        new_interaction.CopyFrom(interaction)
        new_interaction.table.CopyFrom(answer_table)
        new_interaction.id = f"{example['question_id']}/{table_idx}-0"
        new_interaction.questions[0].id = f'{new_interaction.id}_0'
        remove_unreferred_annotation_descriptions(new_interaction)
        yield new_interaction

    else:
      answer_table = _parse_answer_table(table_json, descriptions,
                                         selected_coordinates)
      interaction.table.CopyFrom(answer_table)
      remove_unreferred_annotation_descriptions(interaction)
      yield interaction


def get_referred_annotation_keys(table):
  """Returns a set of annotation keys referred across all the table cells."""
  annotation_keys = set()

  for cell in table.columns:
    for annotation in cell.Extensions[_annotated_cell].annotations:
      annotation_keys.add(annotation.identifier)

  for row in table.rows:
    for cell in row.cells:
      for annotation in cell.Extensions[_annotated_cell].annotations:
        annotation_keys.add(annotation.identifier)

  return annotation_keys


def remove_unreferred_annotation_descriptions(
    interaction):
  """Removes the unreferred annotation descriptions from the interaction."""
  required_keys = get_referred_annotation_keys(interaction.table)

  annotation_extension = interaction.Extensions[_annotation_descriptions]

  required_descriptions = {
      key: value
      for key, value in annotation_extension.descriptions.items()
      if key in required_keys
  }

  annotation_extension.ClearField('descriptions')
  for key, value in required_descriptions.items():
    annotation_extension.descriptions[key] = value


def _create_eval_answer_interaction(
    interaction,
    answer_table,
    i_table = 0):
  """Returns an answer interaction from a base HybridQA interaction.

  Args:
    interaction: HybridQA interaction
    answer_table: The table to-be interested in the new interaction for HybridQA
      RC task.
    i_table: The enumeration index of the answer table formed from the original
      table.
  """
  example_id = text_utils.get_example_id(interaction.id)

  new_interaction = interaction_pb2.Interaction()
  new_interaction.CopyFrom(interaction)
  new_interaction.table.CopyFrom(answer_table)

  new_interaction.id = text_utils.create_interaction_id(example_id, i_table)
  new_interaction.questions[0].id = f'{new_interaction.id}_0'
  new_interaction.questions[0].answer.ClearField('answer_coordinates')

  remove_unreferred_annotation_descriptions(new_interaction)
  return new_interaction


def _create_eval_answer_interactions(
    interaction,
    predicted_coordinates,
    *,
    single_cell_examples = False
):
  """Converts a HybridQA interaction to HybridQA RC interaction.

  This function uses the predicted coordinates of HybridQA task to create eval
  interactions for HybridQA RC Task——ones with smaller expanded
  Tables——from the HybridQA dev interactions.

  Args:
    interaction: HybridQA Cell Selection Task Interaction proto object.
    predicted_coordinates: Iterable of answer-coordinates predicted by HybridQA
      Cell Selection model.
    single_cell_examples: Generate multiple single-celled table interactions for
      each example (interaction) if this is set True.

  Yields:
    Interaction proto.
  """

  # Reference answer for the question. The test set answers are hidden.
  if interaction.questions:
    descriptions = interaction.Extensions[_annotation_descriptions].descriptions

    if single_cell_examples:
      for i_table, coordinate in enumerate(predicted_coordinates):
        answer_table = _create_answer_table(interaction.table, descriptions,
                                            [coordinate])
        new_interaction = _create_eval_answer_interaction(
            interaction, answer_table, i_table)
        yield new_interaction

    else:
      answer_table = _create_answer_table(interaction.table, descriptions,
                                          predicted_coordinates)
      new_interaction = _create_eval_answer_interaction(
          interaction, answer_table, i_table=0)
      yield new_interaction


def _convert_split(
    split,
    main_path,
    table_jsons,
    descriptions_dict,
    config,
):
  """Convert a single JSON file to Interaction protos for a given split."""
  filename = f'{split.value}.json'
  if split in {_Split.TRAIN, _Split.DEV}:
    filename = filename.replace('.json', '.traced.json')
  file_path = os.path.join(main_path, filename)
  with tf.io.gfile.GFile(file_path, 'r') as input_file:
    examples = json.load(input_file)
  multi_answer_count = 0
  for example in examples:
    table_id = str(example['table_id'])
    noisy_cell_strategy = (
        config.noisy_cell_strategy
        if split == _Split.TRAIN else NoisyCellStrategy.NONE)
    interactions = _parse_answer_interactions(
        table_json=table_jsons[table_id],
        descriptions=descriptions_dict[table_id],
        example=example,
        single_cell_examples=config.single_cell_examples,
        use_original_coordinates=config.use_original_coordinates,
        noisy_cell_strategy=noisy_cell_strategy)
    for interaction in interactions:
      coords = interaction.questions[0].answer.answer_coordinates
      if len(coords) > 1:
        multi_answer_count += 1
        if split == _Split.TRAIN and config.skip_ambiguous_examples:
          continue
      yield interaction

  logging.info('Found %d/%d ambiguous examples in %s', multi_answer_count,
               len(examples), split.value)


def create_interactions_from_hybridqa_predictions(
    output_dir,
):
  """Create Interactions for HybridQA RC Task from HybridQA interactions and predictions file.

  Args:
    output_dir: Location where HybridQaRcConfig can be found

  Returns:
    A mapping of dataset split: iterable of interactions
  """
  config_path = os.path.join(output_dir, _CONFIG_NAME)
  with tf.io.gfile.GFile(config_path, 'r') as input_file:
    config = HybridQaRcConfig.from_json(input_file)
  e2e_interactions = {}

  splits_dict = config.hybridqa_prediction_filepaths
  interaction_dirs = config.interactions_dirs

  for ((split, run_id), predictions_file) in splits_dict.items():

    answer_coordinates: MutableMapping[Text, Sequence[_CoordinateType]] = {}
    for p in get_predictions(predictions_file):
      if p.token_probabilities is not None:
        ranked_cells = hybridqa_utils.get_best_cells(p.token_probabilities)
      else:
        ranked_cells = p.answer_coordinates
      example_id = text_utils.get_example_id(p.interaction_id)
      answer_coordinates[example_id] = ranked_cells[:config
                                                    .num_predictions_to_keep]
    original_interactions_file = os.path.join(interaction_dirs[run_id],
                                              f'{split.lower()}.tfrecord')
    original_interactions = prediction_utils.iterate_interactions(
        original_interactions_file)
    interaction_iterables = []
    for original_interaction in original_interactions:
      example_id = text_utils.get_example_id(original_interaction.id)
      interaction_iterables.append(
          _create_eval_answer_interactions(
              original_interaction,
              answer_coordinates[example_id],
              single_cell_examples=config.single_cell_examples))

    if len(splits_dict) == 1:
      # Drop name for backwards compatibility.
      name = split
    else:
      name = f'{split}_{run_id:02}'
    e2e_interactions[name] = itertools.chain(*interaction_iterables)

  return e2e_interactions


def convert(
    input_dir,
    output_dir,
):
  """Reads examples and table in JSON format and converts to interactions."""
  tmpdir = hybridqa_utils.extract_zip_archive(
      os.path.join(input_dir, f'{_ZIP_NAME}.zip'))

  # For each table, the Wikipedia intro of each entity linked to the table.
  descriptions_dir = os.path.join(tmpdir, _ZIP_NAME, 'request_tok')
  descriptions = hybridqa_utils.read_json_directory(descriptions_dir)

  table_dir = os.path.join(tmpdir, _ZIP_NAME, 'tables_tok')
  json_tables = {
      key: json_map for key, json_map in hybridqa_utils.read_json_directory(
          table_dir).items()
  }

  main_path = os.path.join(input_dir, 'released_data')

  config_path = os.path.join(output_dir, _CONFIG_NAME)
  if tf.io.gfile.exists(config_path):
    with tf.io.gfile.GFile(config_path, 'r') as input_file:
      config = HybridQaRcConfig.from_json(input_file)
  else:
    config = HybridQaRcConfig()

  return {
      split.value: _convert_split(split, main_path, json_tables, descriptions,
                                  config) for split in _Split
  }
