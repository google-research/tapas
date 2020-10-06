# TAble PArSing (TAPAS)

Code and checkpoints for training the transformer-based Table QA models introduced
in the paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](#how-to-cite-tapas).

## News

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

### Docker 
You can also directly use TAPAS via Docker. NVIDIA driver and Docker 19.03+ must be installed beforehand. There is no need to install the CUDA toolkit on the host, but the driver needs to be installed. More details can be found on [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

#### Build Image
To build a docker image you can call the following command in the same folder the `Dockerfile` is:

```
docker build --tag tapas:latest .
```

Then, you can simply call the following command to run image:

```  
docker run -it --gpus all -v /absolute/path/to/my/data/directory:/home/data tapas:latest
```

## Data

See the section below for the pre-training data.

### New Models

Based on the pre-trained checkpoints available at the [BERT github page](https://github.com/google-research/bert/blob/master/README.md).
See the page or the [paper](https://arxiv.org/abs/1908.08962) for detailed information on the model dimensions.

**Reset** refers to whether the parameter `reset_position_index_per_cell` was
set to true or false during training. In general it's recommended to set it to true.

The accuracy depends on the respective task. It's denotation accuracy for
WTQ and WIKISQL, average position accuracy with gold labels for the previous answers for SQA and Mask-LM accuracy for Mask-LM.

The models were trained in a chain as indicated by the model name.
For example, *sqa_masklm* means the model was first trained on the Mask-LM task and then on SQA. No destillation was performed.

#### WTQ
Size     |  Reset  | Accuracy | Link
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
Size     |  Reset  | Accuracy | Link
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
Size     |  Reset  | Accuracy | Link
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
Size     |  Reset  | Accuracy | Link
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

### Fine-Tuning Data

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

You can cite the [ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-main.398/).

## Disclaimer

This is not an official Google product.

## Contact information

For help or issues, please submit a GitHub issue.
