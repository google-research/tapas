# Pre-Training Data

The pre-training data consists of 6.2 million table-text examples extracted from
the [English Wikipedia](https://en.wikipedia.org/wiki/Wikipedia) on December 2019.
The associated text of a table is the page title and description, table caption as well as the section title and section text.


## Example

This is an example in proto text format extracted from [this](https://en.wikipedia.org/wiki/ARY_Film_Award_for_Best_Dialogue) page.

```
 table: {
    columns: { text: "Year" }
    columns: { text: "Film" }
    columns: { text: "Dialogue-writer(s)" }
    rows: {
      cells: { text: "2013\n(1st)" }
      cells: { text: "" }
      cells: { text: "" }
    }
    rows: {
      cells: { text: "2013\n(1st)" }
      cells: { text: "Main Hoon Shahid Afridi" }
      cells: { text: "Vasay Chaudhry" }
    }
    table_id: "http://en.wikipedia.org/wiki/ARY_Film_Award_for_Best_Dialogue_1"
  }
  questions: {
    id: "TITLE"
    original_text: "ARY Film Award for Best Dialogue"
  }
  questions: {
    id: "DESCRIPTION"
    original_text: "The ARY Film Award for Best Dialogue is the ARY Film Award for the best dialogues of the year in film. It is one of three writing awards in the Technical Awarding category."
  }
  questions: {
    id: "SEGMENT_TITLE"
    original_text: "2010s"
  }
```

## Data

You can find the latest version of the data [here](https://storage.googleapis.com/tapas_models/2020_05_11/interactions.txtpb.gz).
We also provide a small [snapshot](https://storage.googleapis.com/tapas_models/2020_05_11/interactions_sample.txtpb.gz) of the first 100 interactions.

## Conversion to TF Examples

`create_pretrain_examples_main.py` converts the data to TF examples.
It can be run locally (that will take a long time on a single machine) or as a [Dataflow](https://cloud.google.com/dataflow) on Google Cloud.
You can find command line snippets [here](https://github.com/google-research/tapas#pre-training).

## Parsing Protobuffers in Text Format

In case you want to work with the data in ways we didn't anticipate you can
simple parse them into proto objects line-by-line.

Here is a simple example:

```python
from google.protobuf import text_format
from tapas.protos import interaction_pb2

for line in input_file:
  interaction = text_format.Parse(line, interaction_pb2.Interaction())
```

## Licence

This data is licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).\
See also the Wikipedia [Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) page.

## How to cite this data?

You can cite the [paper to appear at ACL](https://arxiv.org/abs/2004.02349):

```
@inproceedings{49053,
title = {Tapas: Weakly Supervised Table Parsing via Pre-training},
author = {Jonathan Herzig and Paweł Krzysztof Nowak and Thomas Müller and Francesco Piccinno and Julian Martin Eisenschlos},
year = {2020},
URL = {https://arxiv.org/abs/2004.02349},
note = {to appear},
booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
address = {Seattle, Washington, United States}
}
```
