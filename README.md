# TAble PArSing (TAPAS)

<p>
<a href="https://console.tiyaro.ai/explore?q=google/tapas-&pub=google"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>
</p>


Code and checkpoints for training the transformer-based Table QA models introduced
in the paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](#how-to-cite-tapas).

## News

#### 2021/09/15
* Released code for sparse table attention from [MATE: Multi-view Attention for Table Transformer Efficiency](https://arxiv.org/abs/2109.04312). For more info check [here](https://github.com/google-research/tapas/blob/master/MATE.md).

#### 2021/08/24
* Added a [colab](http://tiny.cc/tapas-retrieval-colab) to try predictions on open domain question answering.

#### 2021/08/20
* New models and code for [DoT: An efficient Double Transformer for NLP tasks with tables](https://arxiv.org/abs/2106.00479) released
[here](https://github.com/google-research/tapas/blob/master/DOT.md).

#### 2021/07/23
* New release of NQ with tables data used in [Open Domain Question Answering over Tables via Dense Retrieval](https://arxiv.org/abs/2103.12011). The use of the data is detailed [here](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md).

#### 2021/05/13
 * New models and code for [Open Domain Question Answering over Tables via Dense Retrieval](https://arxiv.org/abs/2103.12011) released
[here](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md).

#### 2021/03/23

 * The upcoming NAACL 2021 short paper [Open Domain Question Answering over Tables via Dense Retrieval](https://arxiv.org/abs/2103.12011) extends the TAPAS capabilities to table retrieval and open-domain QA. We are planning to release the new models and code soon.

#### 2020/12/17

 * TAPAS is added to [huggingface/transformers](https://github.com/huggingface/transformers) in version 4.1.1. 28 checkpoints are added to the [huggingface model hub](https://huggingface.co/models?filter=tapas) and can be played with using a [custom table question answering widget](https://huggingface.co/google/tapas-base-finetuned-wtq).

#### 2020/10/19
 * Small change to WTQ training example creation
   * Questions with ambiguous cell matches will now be discarded
   * This improves denotation accuracy by ~1 point
   * For more details see [this issue](https://github.com/google-research/tapas/issues/73).
 * Added option to filter table columns by textual overlap with question
   * Based on the **HEM** method described in section 3.3 of
 [Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/).

#### 2020/10/09
 * Released code & models to run TAPAS on [TabFact](https://tabfact.github.io/) for table entailment, companion for the EMNLP 2020 Findings paper [Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/).
 * Added a [colab](http://tiny.cc/tapas-tabfact-colab) to try predictions on TabFact
 * Added [new page](https://github.com/google-research/tapas/blob/master/INTERMEDIATE_PRETRAIN_DATA.md) describing the intermediate pre-training process.

#### 2020/08/26
 * Added a [colab](http://tiny.cc/tapas-wtq-colab) to try predictions on WTQ

#### 2020/08/05
 * New pre-trained models (see Data section below)
 * `reset_position_index_per_cell`: New option that allows to train models that instead of using absolute position indices reset the position index when a new cell starts.

#### 2020/06/10
 * Bump TensorFlow to v2.2

#### 2020/06/08
 * Released the [pre-training data](https://github.com/google-research/tapas/blob/master/PRETRAIN_DATA.md).

#### 2020/05/07
 * Added a [colab](http://tiny.cc/tapas-colab) to try predictions on SQA

## Installation

The easiest way to try out TAPAS with free GPU/TPU is in our
[Colab](http://tiny.cc/tapas-colab), which shows how to do predictions on [SQA](http://aka.ms/sqa).

The repository uses protocol buffers, and requires the `protoc` compiler to run.
You can download the latest binary for your OS [here](https://github.com/protocolbuffers/protobuf/releases).
On Ubuntu/Debian, it can be installed with:

```bash
sudo apt-get install protobuf-compiler
```

Afterwards, clone and install the git repository:

```bash
git clone https://github.com/google-research/tapas
cd tapas
pip install -e .
```

To run the test suite we use the [tox](https://tox.readthedocs.io/en/latest/) library which can be run by calling:
```bash
pip install tox
tox
```

## Models

We provide pre-trained models for different model sizes.

The metrics are computed by our tool and not the official metrics of the
respective tasks. We provide them so one can verify whether one's own runs
are in the right ballpark. They are medians over three individual runs.


### Models with intermediate pre-training (2020/10/07).

New models based on the ideas discussed in [Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/). Learn more about the methods use [here](https://github.com/google-research/tapas/blob/master/INTERMEDIATE_PRETRAIN_DATA.md).

#### WTQ

Trained from Mask LM, intermediate data, SQA, WikiSQL.

Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.5062 | [tapas_wtq_wikisql_sqa_inter_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_large.zip)
LARGE | reset | 0.5097 | [tapas_wtq_wikisql_sqa_inter_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_large_reset.zip)
BASE | noreset | 0.4525 | [tapas_wtq_wikisql_sqa_inter_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_base.zip)
BASE | reset | 0.4638 | [tapas_wtq_wikisql_sqa_inter_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_base_reset.zip)
MEDIUM | noreset | 0.4324 | [tapas_wtq_wikisql_sqa_inter_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_medium.zip)
MEDIUM | reset | 0.4324 | [tapas_wtq_wikisql_sqa_inter_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_medium_reset.zip)
SMALL | noreset | 0.3681 | [tapas_wtq_wikisql_sqa_inter_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_small.zip)
SMALL | reset | 0.3762 | [tapas_wtq_wikisql_sqa_inter_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_small_reset.zip)
MINI | noreset | 0.2783 | [tapas_wtq_wikisql_sqa_inter_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_mini.zip)
MINI | reset | 0.2854 | [tapas_wtq_wikisql_sqa_inter_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_mini_reset.zip)
TINY | noreset | 0.0823 | [tapas_wtq_wikisql_sqa_inter_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_tiny.zip)
TINY | reset | 0.1039 | [tapas_wtq_wikisql_sqa_inter_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wtq_wikisql_sqa_inter_masklm_tiny_reset.zip)

#### WIKISQL

Trained from Mask LM, intermediate data, SQA.

Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.8948 | [tapas_wikisql_sqa_inter_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_large.zip)
LARGE | reset | 0.8979 | [tapas_wikisql_sqa_inter_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_large_reset.zip)
BASE | noreset | 0.8859 | [tapas_wikisql_sqa_inter_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_base.zip)
BASE | reset | 0.8855 | [tapas_wikisql_sqa_inter_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_base_reset.zip)
MEDIUM | noreset | 0.8766 | [tapas_wikisql_sqa_inter_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_medium.zip)
MEDIUM | reset | 0.8773 | [tapas_wikisql_sqa_inter_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_medium_reset.zip)
SMALL | noreset | 0.8552 | [tapas_wikisql_sqa_inter_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_small.zip)
SMALL | reset | 0.8615 | [tapas_wikisql_sqa_inter_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_small_reset.zip)
MINI | noreset | 0.8063 | [tapas_wikisql_sqa_inter_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_mini.zip)
MINI | reset | 0.82 | [tapas_wikisql_sqa_inter_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_mini_reset.zip)
TINY | noreset | 0.3198 | [tapas_wikisql_sqa_inter_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_tiny.zip)
TINY | reset | 0.6046 | [tapas_wikisql_sqa_inter_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_wikisql_sqa_inter_masklm_tiny_reset.zip)

#### TABFACT

Trained from Mask LM, intermediate data.

Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.8101 | [tapas_tabfact_inter_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_large.zip)
LARGE | reset | 0.8159 | [tapas_tabfact_inter_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_large_reset.zip)
BASE | noreset | 0.7856 | [tapas_tabfact_inter_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_base.zip)
BASE | reset | 0.7918 | [tapas_tabfact_inter_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_base_reset.zip)
MEDIUM | noreset | 0.7585 | [tapas_tabfact_inter_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_medium.zip)
MEDIUM | reset | 0.7587 | [tapas_tabfact_inter_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_medium_reset.zip)
SMALL | noreset | 0.7321 | [tapas_tabfact_inter_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_small.zip)
SMALL | reset | 0.7346 | [tapas_tabfact_inter_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_small_reset.zip)
MINI | noreset | 0.6166 | [tapas_tabfact_inter_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_mini.zip)
MINI | reset | 0.6845 | [tapas_tabfact_inter_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_mini_reset.zip)
TINY | noreset | 0.5425 | [tapas_tabfact_inter_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_tiny.zip)
TINY | reset | 0.5528 | [tapas_tabfact_inter_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_tabfact_inter_masklm_tiny_reset.zip)

#### SQA

Trained from Mask LM, intermediate data.

Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.7223 | [tapas_sqa_inter_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_large.zip)
LARGE | reset | 0.7289 | [tapas_sqa_inter_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_large_reset.zip)
BASE | noreset | 0.6737 | [tapas_sqa_inter_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_base.zip)
BASE | reset | 0.6874 | [tapas_sqa_inter_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_base_reset.zip)
MEDIUM | noreset | 0.6464 | [tapas_sqa_inter_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_medium.zip)
MEDIUM | reset | 0.6561 | [tapas_sqa_inter_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_medium_reset.zip)
SMALL | noreset | 0.5876 | [tapas_sqa_inter_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_small.zip)
SMALL | reset | 0.6155 | [tapas_sqa_inter_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_small_reset.zip)
MINI | noreset | 0.4574 | [tapas_sqa_inter_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_mini.zip)
MINI | reset | 0.5148 | [tapas_sqa_inter_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_mini_reset.zip)
TINY | noreset | 0.2004 | [tapas_sqa_inter_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_tiny.zip)
TINY | reset | 0.2375 | [tapas_sqa_inter_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_sqa_inter_masklm_tiny_reset.zip)

#### INTERMEDIATE

Trained from Mask LM.

Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.9309 | [tapas_inter_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_large.zip)
LARGE | reset | 0.9317 | [tapas_inter_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_large_reset.zip)
BASE | noreset | 0.9134 | [tapas_inter_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_base.zip)
BASE | reset | 0.9163 | [tapas_inter_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_base_reset.zip)
MEDIUM | noreset | 0.8988 | [tapas_inter_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_medium.zip)
MEDIUM | reset | 0.9005 | [tapas_inter_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_medium_reset.zip)
SMALL | noreset | 0.8788 | [tapas_inter_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_small.zip)
SMALL | reset | 0.8798 | [tapas_inter_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_small_reset.zip)
MINI | noreset | 0.8218 | [tapas_inter_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_mini.zip)
MINI | reset | 0.8333 | [tapas_inter_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_mini_reset.zip)
TINY | noreset | 0.6359 | [tapas_inter_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_tiny.zip)
TINY | reset | 0.6615 | [tapas_inter_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_tiny_reset.zip)


### Small Models & position index reset (2020/08/08)

Based on the pre-trained checkpoints available at the [BERT github page](https://github.com/google-research/bert/blob/master/README.md).
See the page or the [paper](https://arxiv.org/abs/1908.08962) for detailed information on the model dimensions.

**Reset** refers to whether the parameter `reset_position_index_per_cell` was
set to true or false during training. In general it's recommended to set it to true.

The accuracy depends on the respective task. It's denotation accuracy for
WTQ and WIKISQL, average position accuracy with gold labels for the previous answers for SQA and Mask-LM accuracy for Mask-LM.

The models were trained in a chain as indicated by the model name.
For example, *sqa_masklm* means the model was first trained on the Mask-LM task and then on SQA. No destillation was performed.

#### WTQ
Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.4822 | [tapas_wtq_wikisql_sqa_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_large.zip)
LARGE | reset | 0.4952 | [tapas_wtq_wikisql_sqa_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_large_reset.zip)
BASE | noreset | 0.4288 | [tapas_wtq_wikisql_sqa_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_base.zip)
BASE | reset | 0.4433 | [tapas_wtq_wikisql_sqa_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_base_reset.zip)
MEDIUM | noreset | 0.4158 | [tapas_wtq_wikisql_sqa_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_medium.zip)
MEDIUM | reset | 0.4097 | [tapas_wtq_wikisql_sqa_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_medium_reset.zip)
SMALL | noreset | 0.3267 | [tapas_wtq_wikisql_sqa_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_small.zip)
SMALL | reset | 0.3670 | [tapas_wtq_wikisql_sqa_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_small_reset.zip)
MINI | noreset | 0.2275 | [tapas_wtq_wikisql_sqa_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_mini.zip)
MINI | reset | 0.2409 | [tapas_wtq_wikisql_sqa_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_mini_reset.zip)
TINY | noreset | 0.0901 | [tapas_wtq_wikisql_sqa_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_tiny.zip)
TINY | reset | 0.0947 | [tapas_wtq_wikisql_sqa_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wtq_wikisql_sqa_masklm_tiny_reset.zip)

#### WIKISQL
Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.8862 | [tapas_wikisql_sqa_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_large.zip)
LARGE | reset | 0.8917 | [tapas_wikisql_sqa_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_large_reset.zip)
BASE | noreset | 0.8772 | [tapas_wikisql_sqa_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_base.zip)
BASE | reset | 0.8809 | [tapas_wikisql_sqa_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_base_reset.zip)
MEDIUM | noreset | 0.8687 | [tapas_wikisql_sqa_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_medium.zip)
MEDIUM | reset | 0.8736 | [tapas_wikisql_sqa_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_medium_reset.zip)
SMALL | noreset | 0.8285 | [tapas_wikisql_sqa_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_small.zip)
SMALL | reset | 0.8550 | [tapas_wikisql_sqa_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_small_reset.zip)
MINI | noreset | 0.7672 | [tapas_wikisql_sqa_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_mini.zip)
MINI | reset | 0.7944 | [tapas_wikisql_sqa_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_mini_reset.zip)
TINY | noreset | 0.3237 | [tapas_wikisql_sqa_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_tiny.zip)
TINY | reset | 0.3608 | [tapas_wikisql_sqa_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_wikisql_sqa_masklm_tiny_reset.zip)

#### SQA
Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.7002 | [tapas_sqa_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_large.zip)
LARGE | reset | 0.7130 | [tapas_sqa_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_large_reset.zip)
BASE | noreset | 0.6393 | [tapas_sqa_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_base.zip)
BASE | reset | 0.6689 | [tapas_sqa_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_base_reset.zip)
MEDIUM | noreset | 0.6026 | [tapas_sqa_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_medium.zip)
MEDIUM | reset | 0.6141 | [tapas_sqa_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_medium_reset.zip)
SMALL | noreset | 0.4976 | [tapas_sqa_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_small.zip)
SMALL | reset | 0.5589 | [tapas_sqa_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_small_reset.zip)
MINI | noreset | 0.3779 | [tapas_sqa_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_mini.zip)
MINI | reset | 0.3687 | [tapas_sqa_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_mini_reset.zip)
TINY | noreset | 0.2013 | [tapas_sqa_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_tiny.zip)
TINY | reset | 0.2194 | [tapas_sqa_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_sqa_masklm_tiny_reset.zip)

#### MASKLM
Size     |  Reset  | Dev Accuracy | Link
-------- | --------| -------- | ----
LARGE | noreset | 0.7513 | [tapas_masklm_large.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_large.zip)
LARGE | reset | 0.7528 | [tapas_masklm_large_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_large_reset.zip)
BASE | noreset | 0.7323 | [tapas_masklm_base.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_base.zip)
BASE | reset | 0.7335 | [tapas_masklm_base_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_base_reset.zip)
MEDIUM | noreset | 0.7059 | [tapas_masklm_medium.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_medium.zip)
MEDIUM | reset | 0.7054 | [tapas_masklm_medium_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_medium_reset.zip)
SMALL | noreset | 0.6818 | [tapas_masklm_small.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_small.zip)
SMALL | reset | 0.6856 | [tapas_masklm_small_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_small_reset.zip)
MINI | noreset | 0.6382 | [tapas_masklm_mini.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_mini.zip)
MINI | reset | 0.6425 | [tapas_masklm_mini_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_mini_reset.zip)
TINY | noreset | 0.4826 | [tapas_masklm_tiny.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_tiny.zip)
TINY | reset | 0.5282 | [tapas_masklm_tiny_reset.zip](https://storage.googleapis.com/tapas_models/2020_08_05/tapas_masklm_tiny_reset.zip)

### Original Models

The pre-trained TAPAS checkpoints can be downloaded here:

* [MASKLM base](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_base.zip)
* [MASKLM large](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_large.zip)
* [SQA base](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_sqa_base.zip)
* [SQA large](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_sqa_large.zip)

The first two models are pre-trained on the Mask-LM task and the last two
on the Mask-LM task first and SQA second.

## Fine-Tuning Data

You also need to download the task data for the fine-tuning tasks:

* [SQA](http://aka.ms/sqa)
* [WikiSQL](https://github.com/salesforce/WikiSQL)
* [WTQ 1.0](https://github.com/ppasupat/WikiTableQuestions)
* [TabFact](https://github.com/wenhuchen/Table-Fact-Checking)

## Pre-Training

Note that you can skip pre-training and just use one of the pre-trained checkpoints provided above.

Information about the pre-taining data can be found [here](https://github.com/google-research/tapas/blob/master/PRETRAIN_DATA.md).

The TF examples for pre-training can be created using [Google Dataflow](https://cloud.google.com/dataflow):

```bash
python3 setup.py sdist
python3 tapas/create_pretrain_examples_main.py \
  --input_file="gs://tapas_models/2020_05_11/interactions.txtpb.gz" \
  --vocab_file="gs://tapas_models/2020_05_11/vocab.txt" \
  --output_dir="gs://your_bucket/output" \
  --runner_type="DATAFLOW" \
  --gc_project="you-project" \
  --gc_region="us-west1" \
  --gc_job_name="create-pretrain" \
  --gc_staging_location="gs://your_bucket/staging" \
  --gc_temp_location="gs://your_bucket/tmp" \
  --extra_packages=dist/tapas-0.0.1.dev0.tar.gz
```

You can also run the pipeline locally but that will take a long time:

```bash
python3 tapas/create_pretrain_examples_main.py \
  --input_file="$data/interactions.txtpb.gz" \
  --output_dir="$data/" \
  --vocab_file="$data/vocab.txt" \
  --runner_type="DIRECT"
```

This will create two tfrecord files for training and testing.
The pre-training can then be started with the command below.
The init checkpoint should be a standard BERT checkpoint.

```bash
python3 tapas/experiments/tapas_pretraining_experiment.py \
  --eval_batch_size=32 \
  --train_batch_size=512 \
  --tpu_iterations_per_loop=5000 \
  --num_eval_steps=100 \
  --save_checkpoints_steps=5000 \
  --num_train_examples=512000000 \
  --max_seq_length=128 \
  --input_file_train="${data}/train.tfrecord" \
  --input_file_eval="${data}/test.tfrecord" \
  --init_checkpoint="${tapas_data_dir}/model.ckpt" \
  --bert_config_file="${tapas_data_dir}/bert_config.json" \
  --model_dir="..." \
  --compression_type="" \
  --do_train
```

Where **compression_type** should be set to **GZIP** if the tfrecords are compressed.
You can start a separate eval job by setting `--nodo_train --doeval`.

## Running a fine-tuning task

We need to create the TF examples before starting the training.
For example, for SQA that would look like:

```bash
python3 tapas/run_task_main.py \
  --task="SQA" \
  --input_dir="${sqa_data_dir}" \
  --output_dir="${output_dir}" \
  --bert_vocab_file="${tapas_data_dir}/vocab.txt" \
  --mode="create_data"
```

Optionally, to handle big tables, we can add a `--prune_columns` flag to
apply the **HEM** method described section 3.3 of our
[paper](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) to discard some columns based on
textual overlap with the sentence.

Afterwards, training can be started by running:

```bash
python3 tapas/run_task_main.py \
  --task="SQA" \
  --output_dir="${output_dir}" \
  --init_checkpoint="${tapas_data_dir}/model.ckpt" \
  --bert_config_file="${tapas_data_dir}/bert_config.json" \
  --mode="train" \
  --use_tpu
```

This will use the preset hyper-parameters set in `hparam_utils.py`.

It's recommended to start a separate eval job to continuously produce predictions
for the checkpoints created by the training job. Alternatively, you can run
the eval job after training to only get the final results.

```bash
python3 tapas/run_task_main.py \
  --task="SQA" \
  --output_dir="${output_dir}" \
  --init_checkpoint="${tapas_data_dir}/model.ckpt" \
  --bert_config_file="${tapas_data_dir}/bert_config.json" \
  --mode="predict_and_evaluate"
```

Another tool to run experiments is ```tapas_classifier_experiment.py```. It's more
flexible than ```run_task_main.py``` but also requires setting all the hyper-parameters
(via the respective command line flags).


## Evaluation

Here we explain some details about different tasks.

### SQA

By default, SQA will evaluate using the reference answers of the previous
questions. The number in [the paper](#how-to-cite-tapas) (Table 5) are computed
using the more realistic setup
where the previous answer are model predictions. `run_task_main.py` will output
additional prediction files for this setup as well if run on GPU.

### WTQ

For the official evaluation results one should convert the TAPAS predictions to
the WTQ format and run the official evaluation script. This can be done using
`convert_predictions.py`.


### WikiSQL

As discussed in [the paper](#how-to-cite-tapas) our code will compute evaluation
metrics that deviate from the official evaluation script (Table 3 and 10).


## Hardware Requirements

TAPAS is essentialy a BERT model and thus has the same [requirements](https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues).
This means that training the large model with 512 sequence length will
require a TPU.
You can use the option `max_seq_length` to create shorter sequences. This will
reduce accuracy but also make the model trainable on GPUs.
Another option is to reduce the batch size (`train_batch_size`),
but this will likely also affect accuracy.
We added an options `gradient_accumulation_steps` that allows you to split the
gradient over multiple batches.
Evaluation with the default test batch size (32) should be possible on GPU.

## <a name="how-to-cite-tapas"></a>How to cite TAPAS?

You can cite the [ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-main.398/)
and the [EMNLP 2020 Findings paper](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) for the laters work on pre-training objectives.

## Disclaimer

This is not an official Google product.

## Contact information

For help or issues, please submit a GitHub issue.
