# Open Domain Question Answering over Tables via Dense Retrieval

This document contains models and steps to reproduce the results of [Open Domain Question Answering over Tables via Dense Retrieval](https://arxiv.org/abs/2103.12011) published at NAACL2021.

## Retrieval Models

Size  |  Type           | Hard Negatives | Down Project | Recall@1 | Recall@10 | Recall@50 | Link
----- | --------------- | -------------- | ------------ | -------- | --------- | --------- | ----
LARGE | Pretrained      | No             | No           | | | | [tapas_dual_encoder_proj_0_large.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_dual_encoder_proj_0_large.zip)
LARGE | Pretrained      | No             | 256          | | | | [tapas_dual_encoder_proj_256_large.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_dual_encoder_proj_256_large.zip)
MEDIUM | Pretrained      | No             | 256          | | | | [tapas_dual_encoder_proj_256_medium.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_dual_encoder_proj_256_medium.zip)
SMALL | Pretrained      | No             | 256          | | | | [tapas_dual_encoder_proj_256_small.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_dual_encoder_proj_256_small.zip)
TINY  | Pretrained      | No             | 256          | | | | [tapas_dual_encoder_proj_256_tiny.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_dual_encoder_proj_256_tiny.zip)
LARGE | Finetuned on NQ | No             | 256          | 35.9 | 75.9 | 91.4 | [tapas_nq_retriever_large.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_retriever_large.zip)
LARGE | Finetuned on NQ | Yes            | 256          | 44.2 | 81.8 | 92.3 | [tapas_nq_hn_retriever_large.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_hn_retriever_large.zip)
MEDIUM | Finetuned on NQ | No             | 256          | 37.1 | 74.5 | 88.0 | [tapas_nq_retriever_medium.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_retriever_medium.zip)
MEDIUM| Finetuned on NQ | Yes            | 256          | 44.9 | 79.8 | 91.1 | [tapas_nq_hn_retriever_medium.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_hn_retriever_medium.zip)
SMALL | Finetuned on NQ | No             | 256          | 37.6 | 72.8 | 87.4 | [tapas_nq_retriever_small.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_retriever_small.zip)
SMALL | Finetuned on NQ | Yes            | 256          | 41.8 | 77.1 | 89.9 | [tapas_nq_hn_retriever_small.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_hn_retriever_small.zip)
TINY | Finetuned on NQ | No             | 256          | 17.3 | 54.1 | 76.3 | [tapas_nq_retriever_tiny.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_retriever_tiny.zip)
TINY | Finetuned on NQ | Yes            | 256          | 22.2 | 61.3 | 78.9 | [tapas_nq_hn_retriever_tiny.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_hn_retriever_tiny.zip)

## Reader models

Size  | Hard Negatives | Link
----- | -------------- | --------------
LARGE | No             | [tapas_nq_reader_large.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_reader_large.zip)
LARGE | Yes            | [tapas_nq_hn_reader_large.zip](https://storage.googleapis.com/tapas_models/2021_04_27/tapas_nq_hn_reader_large.zip)

## Load directly the released data
```bash
mkdir -p "${nq_data_dir}"
gsutil -m cp -R gs://tapas_models/2021_07_22/nq_tables/* "${nq_data_dir}"
```

## Or generate the data

The following pipeline will generate the subset of Natural Questions where
the answers are part of tables.


```bash
nq_data_dir=~/data/nq

# Set GCP_PROJECT and GCP_BUCKET variables
gcloud config set project "${GCP_PROJECT}"
gcloud auth application-default login
python3 setup.py sdist
python3 tapas/scripts/preprocess_nq.py \
  --input_path="gs://natural_questions/v1.0" \
  --output_path="gs://${GCP_BUCKET}/nq_tables" \
  --runner_type="DATAFLOW" \
  --save_main_session \
  --gc_project="${GCP_PROJECT}" \
  --gc_region="us-west1" \
  --gc_job_name="create-intermediate" \
  --gc_staging_location="gs://${GCP_BUCKET}/staging" \
  --gc_temp_location="gs://${GCP_BUCKET}/tmp" \
  --extra_packages=dist/tapas-table-parsing-0.0.1.dev0.tar.gz
mkdir -p "${nq_data_dir}"
gsutil -m cp -R "gs://${GCP_BUCKET}/nq_tables/*" "${nq_data_dir}"
```

Or you can also run the pipeline locally but that will take a long time and memory:

```bash
mkdir -p "${nq_data_dir}/raw"
gsutil -m cp -R gs://natural_questions/v1.0/* "${nq_data_dir}/raw"
python3 tapas/scripts/preprocess_nq.py \
  --input_path="gs://natural_questions/v1.0" \
  --output_path="${nq_data_dir}" \
  --runner_type="DIRECT"
```

## Retrieval Flow

The full-fledged retrieval process is composed of the following steps. Each step
is described in details below.

1.  Pre-train the model.
2.  Fine-tune the model.
3.  Select the best checkpoint w.r.t to some retrieval metric (e.g.,
    `eval_precision_at_1`) in the local setting (which considers all tables that
    appear in the dev set as the corpus). These metrics are printed to XM.
4.  Produce global predictions for the selected best checkpoint - these consist
    of representations for all tables in the corpus.
5.  Generate retrieval metrics w.r.t to the global setting, and write KNN tables
    ids and scores for each query to a JSON file (to be used for negatives
    mining or E2E QA).

### Fine-Tuning a retrieval model

Download a pretrained checkpoint:

```bash
retrieval_model_name=tapas_dual_encoder_proj_256_large
gsutil cp "gs://tapas_models/2021_04_27/${retrieval_model_name}.zip" . && unzip "${retrieval_model_name}.zip"
```

Then we can create the data for the retrieval model

```bash
max_seq_length=512
python3 tapas/retrieval/create_retrieval_data_main.py \
  --input_interactions_dir="${nq_data_dir}/interactions" \
  --input_tables_dir=${nq_data_dir}/tables \
  --output_dir="${nq_data_dir}/tf_examples" \
  --vocab_file="${retrieval_model_name}/vocab.txt" \
  --max_seq_length="${max_seq_length}" \
  --max_column_id="${max_seq_length}" \
  --max_row_id="${max_seq_length}" \
  --use_document_title
```

 and train a dual encoder model

```bash
model_dir=~/models/nq
python3 tapas/experiments/table_retriever_experiment.py \
   --do_train \
   --use_tpu \
   --keep_checkpoint_max=40 \
   --model_dir="${model_dir}" \
   --input_file_train="${nq_data_dir}/tf_examples/train.tfrecord" \
   --bert_config_file="${retrieval_model_name}/bert_config.json" \
   --init_checkpoint="${retrieval_model_name}/model.ckpt" \
   --init_from_single_encoder=false \
   --down_projection_dim=256 \
   --num_train_examples=5120000 \
   --learning_rate=1.25e-5 \
   --train_batch_size=256 \
   --warmup_ratio=0.01 \
   --max_seq_length="${max_seq_length}"
```

It's recommended to start a separate eval job to continuously produce predictions
for the checkpoints created by the training job. This will also create json
files with compputed metrics that will allow doing early stopping.

```bash
python3 tapas/experiments/table_retriever_experiment.py \
   --do_predict \
   --model_dir="${model_dir}" \
   --input_file_eval="${nq_data_dir}/tf_examples/dev.tfrecord" \
   --bert_config_file="${retrieval_model_name}/bert_config.json" \
   --init_from_single_encoder=false \
   --down_projection_dim=256 \
   --eval_batch_size=32 \
   --num_train_examples=5120000 \
   --max_seq_length="${max_seq_length}"
```

### Predict using the best dev checkpoint

Once training is done, we use the best checkpoint to gerenerate embeddings for
all the tables and all of the training data queries.
This will be necessary to train the reader model as well as realistic evaluation
using all table candidates.

```bash
for mode in train tables test
  python3 tapas/experiments/table_retriever_experiment.py \
     --do_predict \
     --model_dir="${model_dir}" \
     --prediction_output_dir="${model_dir}/${mode}" \
     --evaluated_checkpoint_metric=precision_at_1 \  # This actually represents recall@1
     --input_file_predict="${nq_data_dir}/tf_examples/${mode}.tfrecord" \
     --bert_config_file="${retrieval_model_name}/bert_config.json" \
     --init_from_single_encoder=false \
     --down_projection_dim=256 \
     --eval_batch_size=32 \
     --max_seq_length="${max_seq_length}"
```

### Generate Retrieval Results

Run evaluation to print recall@k scores in the global setting given the best
model (e.g., 5K checkpoint in this case). Also, generate all KNN most similar
tables per query and their similarity scores to a jsonl file.

*   Set `prediction_files_local` to the best model output. This file holds the
    query ids, their representations, and the ids for the gold table.
*   Set `prediction_files_global` to the output path of the last step.


```bash
step=<SET_STEPS>  # Set this value according to the best dev results. The train and tables predictions generated in the previous step will only exist for this step.

# Computes train results
python tapas/scripts/eval_table_retriever.py \
 --prediction_files_local=${model_dir}/train/predict_results_${step}.tsv \
 --prediction_files_global=${model_dir}/tables/predict_results_${step}.tsv \
 --retrieval_results_file_path="${model_dir}/train_knn.jsonl"

# Computes test results
python tapas/scripts/eval_table_retriever.py \
 --prediction_files_local=${model_dir}/test/predict_results_${step}.tsv \
 --prediction_files_global=${model_dir}/tables/predict_results_${steps}.tsv \
 --retrieval_results_file_path="${model_dir}/test_knn.jsonl"

# Computes dev results
python tapas/scripts/eval_table_retriever.py \
 --prediction_files_local=${model_dir}/eval_results_${step}.tsv \
 --prediction_files_global=${model_dir}/tables/predict_results_${steps}.tsv \
 --retrieval_results_file_path="${model_dir}/dev_knn.jsonl"
```

### Create training data for reader model

First we create the training data
```bash
python3 tapas/retrieval/create_e2e_interactions.py \
  --input_dir="${nq_data_dir}/interactions" \
  --table_file="${nq_data_dir}/tables/tables.tfrecord" \
  --index_files_pattern="${FLAGS_model_dir}/*_knn.jsonl" \
  --output_dir="${nq_data_dir}/e2e/nq_retrieval/interactions"

reader_model_name=tapas_masklm_large
gsutil cp "gs://tapas_models/2020_08_05/${reader_model_name}.zip" . && unzip "${reader_model_name}.zip"
python3 tapas/run_task_main.py \
  --task="NQ_RETRIEVAL" \
  --verbosity=-1 \
  --input_dir="${nq_data_dir}/e2e" \
  --output_dir="${nq_data_dir}/e2e" \
  --bert_vocab_file="${reader_model_name}/vocab.txt" \
  --mode="create_data" \
  --use_document_title \
  --update_answer_coordinates \
  --drop_rows_to_fit
```

### Fine-tune reader model

```bash
model_dir=~/models/nq_e2e
python3 tapas/run_task_main.py \
  --task="NQ_RETRIEVAL" \
  --output_dir="${nq_data_dir}/e2e" \
  --model_dir="${model_dir}" \
  --init_checkpoint="${reader_model_name}/model.ckpt" \
  --bert_config_file="${reader_model_name}/bert_config.json" \
  --mode="train" \
  --use_tpu
```

This will use the preset hyper-parameters set in `hparam_utils.py`.

It's recommended to start a separate eval job to continuously produce predictions
for the checkpoints created by the training job. Alternatively, you can run
the eval job after training to only get the final results.

```bash
python3 tapas/run_task_main.py \
  --task="NQ_RETRIEVAL" \
  --output_dir="${nq_data_dir}/e2e" \
  --model_dir="${model_dir}" \
  --init_checkpoint="${reader_model_name}/model.ckpt" \
  --bert_config_file="${reader_model_name}/bert_config.json" \
  --bert_vocab_file="${reader_model_name}/vocab.txt" \
  --mode="predict_and_evaluate"
```

## Licence

This code and data derived from Natural Questions are licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
The pretraining data is licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).\
See also the Wikipedia [Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) page.

## How to cite this data and code?

You can cite the [paper](https://arxiv.org/abs/2103.12011) and the released data
to appear in NAACL 2021.
