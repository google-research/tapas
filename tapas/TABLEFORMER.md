# TableFormer: Robust Transformer Modeling for Table-Text Encoding
This document contains models and steps to reproduce the results of [TableFormer: Robust Transformer Modeling for Table-Text Encoding](https://arxiv.org/abs/2203.00274) published at ACL 2022.

## TableFormer Model

TableFormer encodes the general table structure along with the associated text
by introducing task-independent relative attention biases for table-text
encoding to facilitate the following:

 * structural inductive bias for better table understanding and table-text
alignment,
 * robustness to table row/column perturbation.

TableFormer is:
 * strictly invariant to row and column orders, and,
 * could understand tables better due to its tabular inductive biases.

Our evaluations show that TableFormer outperforms strong baselines in all
settings on SQA, WTQ and TABFACT table reasoning datasets, and achieves
state-of-the-art performance on SQA, especially when facing answer-invariant
row and column order perturbations (6% improvement over the best baseline),
because previous SOTA models’ performance drops by 4% - 6% when facing such
perturbations while TableFormer is not affected.

## Using TableFormer

Using TableFormer for pre-training and fine-tuning can be acomplished through
the following configuration flags in `tapas_pretraining_experiment.py` and
`tapas_classifier_experiment.py`, respectively:

 * `--restrict_attention_mode=table_attention` Uses the 13 relative relational
 ids introduced in TableFormer.
 * `--attention_bias_use_relative_scalar_only` Whether to just use a scalar bias
 or an embedding per relative id per head per layer.
 * `--attention_bias_disabled` Which relational id to be disabled. This should
 only be used for ablation studies, otherwise defaults to 0.


## Licence

This code and data are licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).\
See also the Wikipedia [Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) page.

## How to cite this data and code?

You can cite the [paper](https://arxiv.org/abs/2203.00274) to appear in
ACL 2022.

```
@inproceedings{yang-etal-2022-tableformer,
    title="{TableFormer: Robust Transformer Modeling for Table-Text Encoding}",
    author="Jingfeng Yang and Aditya Gupta and Shyam Upadhyay and Luheng He and Rahul Goel and Shachi Paul",
    booktitle = "Proc. of ACL",
    year = "2022"
}
```
