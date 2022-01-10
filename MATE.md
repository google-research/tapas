# MATE: Multi-view Attention for Table Transformer Efficiency
This document contains models and steps to reproduce the results of [MATE: Multi-view Attention for Table Transformer Efficiency](https://arxiv.org/abs/2109.04312) published at EMNLP 2021.

## MATE Model

Based on the intuition that attention across tokens in different columns and
rows is not needed, MATE uses two types of attention heads that can either only
attend within the same column or within the same row.

MATE can be (approximately) implemented linearly by adapting an idea from
Reformer  (Kitaev et al., 2020): having column heads sort the input according to
a column order and row heads according to the row order.
Then the input is bucketed and attention restricted to adjacent buckets.

## Using MATE

Using for pre-training or fine-tuning a model can be accomplished through the
following configuration flags in `tapas_classifier_experiment.py`:

 * `--restrict_attention_mode=same_colum_or_row` Attention from tokens in
 different columns and rows is masked out.
 * `--restrict_attention_mode=headwise_same_colum_or_row` Row heads mask
 attention between different rows, and columns heads between columns.
 The `bucket_size` and `header_size` arguments define below can be optionally
 applied to mimic the efficient implementation.
 * `--restrict_attention_mode=headwise_efficient` Similar to
 `headwise_same_colum_or_row` but uses an log linear implementation by sorting
 the input tokens by column or row order depending on the type of attention head.
 * `--restrict_attention_bucket_size=<int>` For sparse attention modes, further
 restricts attention to consecutive buckets of uniform size. Two tokens may only
 attend each other if the fall in consecutive buckets of this size.
 Only required for `restrict_attention_mode=headwise_efficient`.
 * `--restrict_attention_header_size=<int>` For sparse attention modes, size of
 the first section that will attend to/from everything else. Only required for
 `restrict_attention_mode=headwise_efficient`.
 * `--restrict_attention_row_heads_ratio=<float> For sparse attention modes,
 proportion of heads that should focus on rows vs columns. Default is 0.5.

## Licence

This code and data are licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).\
See also the Wikipedia [Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) page.

## How to cite this data and code?

You can cite the [paper](https://arxiv.org/abs/2109.04312) to appear in
EMNLP 2021.
