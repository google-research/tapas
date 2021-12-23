# Intermediate Pre-Training Data

In our latest models, two training objectives are used after the standard mask
language modeling and before fine-tuning, hence the name intermediate pre-training. Both objectives are binary classification tasks on a sentence table pair, where the tables are real Wikipedia tables based on the ones released [here](https://github.com/google-research/tapas/blob/master/PRETRAIN_DATA.md). To explain each of the two tasks let us use the following example table:

| Rank | Player          | Country       | Earnings  | Events | Wins |
|------|-----------------|---------------|-----------|--------|------|
| 1    | Greg Norman     | Australia     | 1,654,959 | 16     | 3    |
| 2    | Billy Mayfair   | United States | 1,543,192 | 28     | 2    |
| 3    | Lee Janzen      | United States | 1,378,966 | 28     | 3    |
| 4    | Corey Pavin     | United States | 1,340,079 | 22     | 2    |
| 5    | Steve Elkington | Australia     | 1,254,352 | 21     | 2    |

## Synthetic Examples

We generate synthetic examples using a simple grammar.
The grammar randomly generates two SQL-like phrases that can use aggregations or constants and compares one against each other using equality or numeric comparisons.
We assign a binary label according to the truth value of the generated statement, and the algorithm is adjusted to get the same number of positives and negatives.
The total amount of examples obtained is 3.7 million.

Below are some examples, and we recommend looking at the section 3.2 in the [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) to see the full details.

1. **2** is less than **wins when Player is Lee Janzen**.
The right hand side corresponds to the query

    SELECT wins FROM table WHERE player = "Lee Janzen"

2. **The sum of Earnings when Country is Australia** is **2,909,311**.
The right hand side corresponds to the query

    SELECT SUM(earnings) FROM table WHERE country = "Australia"

Although the language is artificial, these examples can improve the model numerical reasoning skills.

## Counterfactual Examples

Counterfactual examples are created by randomly replacing an entity in a statement for a different but plausible entity.
The original sentences are also obtained from Wikipedia by selecting text in the vicinity of the table or that has a hyperlink to the table from a different page.
For example, if we get a text that reads

> Greg Norman has the highest earnings

we replace "Greg Norman" for another entity appearing in the same column of the table, obtaining

> ~~Greg Norman~~ **Steve Elkington** has the highest earnings.

The total amount of examples obtained is 4.1 million and the model is then asked to detect whether the sentence has been corrupted or not.
The sentences obtained in this manner sound more natural, but the type of logical inference
that the model has to perform on the table is typically simpler, since often looking at a single row is enough. You may also check section 3.1 in the [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) to see the full details.

## Model release

The models trained on these datasets jointly are released with and without the fine-tuning on the end tasks (TabFact, SQA, etc.). All the links with the expected results are listed in the main [readme](https://github.com/google-research/tapas/blob/master/README.md#models).

## Code Release

Data generation code can be run with:

```bash
python3 setup.py sdist
python3 tapas/create_intermediate_pretrain_examples_main.py \
  --input_file="gs://tapas_models/2021_07_22/interactions.txtpb.gz" \
  --vocab_file="gs://tapas_models/2021_07_22/vocab.txt" \
  --output_dir="gs://your_bucket/output" \
  --runner_type="DATAFLOW" \
  --gc_project="you-project" \
  --gc_region="us-west1" \
  --gc_job_name="create-intermediate" \
  --gc_staging_location="gs://your_bucket/staging" \
  --gc_temp_location="gs://your_bucket/tmp" \
  --extra_packages=dist/tapas-0.0.1.dev0.tar.gz
```

You can also run the pipeline locally but that will take a long time:

```bash
python3 tapas/create_intermediate_pretrain_examples_main.py \
  --input_file="$data/interactions.txtpb.gz" \
  --output_dir="$data/" \
  --vocab_file="$data/vocab.txt" \
  --runner_type="DIRECT"
```

## Licence

This code and data are licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).\
See also the Wikipedia [Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) page.

## How to cite this data and code?

You can cite the [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) to appear in
EMNLP 2020 Findings.
