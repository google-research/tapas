# DoT: An efficient Double Transformer for NLP tasks with tables
This document contains models and steps to reproduce the results of [DoT: An efficient Double Transformer for NLP tasks with tables](https://arxiv.org/abs/2106.00479) published at ACL Findings 2021.

## DoT Models
DoT is a double transformer model composed of a first pruning transformer that selects 256 tokens and passes them to a second transformer, a task specific transformer that solves the main task.

Best models' results presented in the paper are reported in the following table.

Dataset  | Model                     | Accuracy   | Best  | NPE/s | Link
-------- | ------------------------- | ---------- | ----- | ----- | ------
WikiSQL  | HEM-1024->DoT(s-256->l)   | 85.3±0.4   | 85.76 | 1250  | [wikisql_hem_1024_dot_small_256_large.zip](https://storage.googleapis.com/tapas_models/2021_08_20/wikisql_hem_1024_dot_small_256_large.zip)
TABFACT  | HEM-1024->DoT(s-256->l)   | 81.6±0.3   | 81.74 | 1300  |[tabfact_hem_1024_dot_small_256_large.zip](https://storage.googleapis.com/tapas_models/2021_08_20/tabfact_hem_1024_dot_small_256_large.zip)
WikiTQ   | CC-1024->C-DoT(m-256->l)  | 50.1±0.5   | 50.14 | 950   |[wtq_cc_1024_column_dot_medium_256_large.zip](https://storage.googleapis.com/tapas_models/2021_08_20/wtq_cc_1024_column_dot_medium_256_large.zip)

## Learning DoT model
The different steps to learn DoT models:
1. Pre-train the pruning and task-specific transformers.
2. Create the data for the fine-tuning.
3. Create the pruning config.
4. Fine-tune DoT: Jointly learn the two transformers.

### Pre-training
All DoT models are initialized from models pre-trained with a Mask LM, intermediate data and SQA. Both pruning and task-specific transformers are pre-trained.

### Generating HEM or CC data
To reduce the input size of the interactions we can use the heuristic exact match (HEM) or the cell concatenation (CC). Then the code extracts the corresponding tf_examples.
```bash
HEURISTIC="hem"

python -m tapas/run_task_main \
  --task=${TASK} \
  --input_dir="${task_data_dir}" \
  --output_dir="${output_dir}/${HEURISTIC}" \
  --max_seq_length="1024" \
  --prune_columns="true" \
  --bert_vocab_file="${tapas_data_dir}/vocab.txt" \
  --mode="create_data"
```
For CC use HEURISTIC="CC" and prune_columns=false.
We use max_seq_length=1024 as all the DoT models use a heuristic to reduce the input length to 1024 (.-1024->DoT).

### Create the pruning transformer config
The pruning config follows tapas/protos/table_pruning.proto
For example to create a config for -DoT(s-256->.) use:

```bash
# proto-file: tapas/google/table_pruning.proto
# proto-message: tapas.TablePruningModel
max_num_tokens: 256
tapas{
  selection: TOKENS
  loss: {
    unsupervised: {regularization:NONE}
    train: {selection_fn: TOP_K}
    eval: {selection_fn: TOP_K}
  }
  reset_position_index_per_cell: true
  bert_config_file: "${TAPAS_S_CHECK_POINT}/bert_config.json"
  bert_init_checkpoint: "${TAPAS_S_CHECK_POINT}/model.ckpt"
}
```
Then use the path to the created file:
```bash
CONGIG_FILE="<path_to_the_config>.textproto"
```

### Fine-tuning DoT
DoT has been used for three datasets. To select the dataset TASK can be set to WIKISQL, TABFACT, or WTQ.

```bash
python -m tapas.run_task_main \
  --task="${TASK}" \
  --max_seq_length=1024 \
  --output_dir="${output_dir}/${HEURISTIC}" \
  --init_checkpoint="${TAPAS_L_CHECK_POINT}/model.ckpt" \
  --bert_config_file="${TAPAS_L_CHECK_POINT}/bert_config.json" \
  --table_pruning_config_file="${CONGIG_FILE}" \
  --reset_position_index_per_cell=true \
  --mode="train"
```

## Licence
This code and data are licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).\
See also the Wikipedia [Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) page.

## How to cite this data and code?
You can cite the [paper](https://arxiv.org/abs/2106.00479) to appear in ACL Findings 2021.
