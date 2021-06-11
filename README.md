# This is the official code repository for LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning

There are two dirs under this file:
 - reason/generate_data.py generates all but one task introduced in LIME (see
Appendix B): induct, deduct, abduct, induct_v2, induct_v3, induct_rewrite,
rewrite.
 - rewrite_multi/generate_data.py generates two variants of rewrite_multi_step
introduced in LIME, Appendix B.1. The version "rewrite_multistep_hard" is the
one described in the paper, which is also recommended to use.

Arguments of generate_data.py:

 - To generate synthetic data, we recommend generating 5 to 10 M examples,
 specified by the arg `num_train`. 

 - To generate a mixed of tasks, one simply put multiple task names after arg `mode`. 
 For example: python reason/generate_data.py --mode induct deduct abduct generates 
 examples that's a mixed of three tasks, which is called task "mixed" in the paper.

 - The arg `vocab_size` is the vocab size of the synthetic task. In LIME, we had a 
 discussion about the effect of this in Appendix C.1. It seems matter somewhat.
 Empirically we observed that the closer to the downstream task's vocab size, the
 better. But for larger vocab size it becomes harder to train. So we recommend
 vocab_size 1000, and it seems to work well for most of our tasks. Usually, it
 takes 3-5M steps to converge for vocab_size 1000, with batch size 4*4*4096.



