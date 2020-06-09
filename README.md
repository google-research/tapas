# TAble PArSing (TAPAS)

Code and checkpoints for training the transformer-based Table QA models introduced
in the paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](#how-to-cite-tapas).

## News

2020/06/10
 - Bump TensorFlow to v2.2

2020/06/08
 - Released the [pre-training data](https://github.com/google-research/tapas/blob/master/PRETRAIN_DATA.md).

2020/05/07
 - Added a [colab](http://tiny.cc/tapas-colab) to try predictions on SQA

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

## Data

See the section below for the pre-training data.

The pre-trained TAPAS checkpoints can be downloaded here:

* [MASKLM base](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_base.zip)
* [MASKLM large](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_large.zip)
* [SQA base](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_sqa_base.zip)
* [SQA large](https://storage.googleapis.com/tapas_models/2020_04_21/tapas_sqa_large.zip)

The first two models are pre-trained on the Mask-LM task and the last two
on the Mask-LM task first and SQA second.

You also need to download the task data for the fine-tuning tasks:

* [SQA](http://aka.ms/sqa)
* [WikiSQL](https://github.com/salesforce/WikiSQL)
* [WTQ 1.0](https://github.com/ppasupat/WikiTableQuestions)


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
  --save_checkpoints_steps=600 \
  --num_train_examples=512000000 \
  --max_seq_length=128 \
  --input_file_train="${data}/train.tfrecord" \
  --input_file_eval="${data}/test.tfrecord" \
  --init_checkpoint="${tapas_data_dir}/model.ckpt" \
  --bert_config_file="${tapas_data_dir}/bert_config.json" \
  --model_dir="..." \
  --do_train
```

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

Here we explain some details about different tasks and give some rough numbers.
These numbers are ```denotation accuracy``` as computed by our tool and not
the official metrics of the respective tasks. The numbers are here to
to verify whether one's own runs are in the right ballpark. These numbers are
not medians but individual runs.

### SQA

By default, SQA will evaluate using the reference answers of the previous
questions. The number in [the paper](#how-to-cite-tapas) (Table 5) are computed
using the more realistic setup
where the previous answer are model predictions. `run_task_main.py` will output
additional prediction files for this setup as well if run on GPU.

Model | Dev Denotation | Dev Seq Denotation
----- | -------------- | ------------------
Large |        0.68298 | 0.65038
Base  |        0.63854 | 0.57837


### WTQ

For the official evaluation results one should convert the TAPAS predictions to
the WTQ format and run the official evaluation script. This can be done using
`convert_predictions.py`.

Model     |  Dev Denotation
--------- | ---------------
SQA Large | 0.49288
Large     | 0.41637
SQA Base  | 0.41210
Base      | 0.26085


### WikiSQL

As discussed in [the paper](#how-to-cite-tapas) our code will compute evaluation
metrics that deviate from the official evaluation script (Table 3 and 10).

Model     |  Dev Denotation
--------- | ---------------
LARGE     | 0.85239
BASE      | 0.81166


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


## Disclaimer

This is not an official Google product.

## Contact information

For help or issues, please submit a GitHub issue.
