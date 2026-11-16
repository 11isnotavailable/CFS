We added seven randomly initialized BERT-style MLM encoders with shared2B-input-token training, separating width, depth, and parameter-identical head decomposition. We evaluate the same256 held-out documents and fixed masked positions using the original CFS intervention and alpha grid. The table reports observed directions under matched candidate counts and fixed graph sizes, together with input-paired intervals and1B/2B functional stability. This provides a controlled bidirectional-encoder test beyond the original causal-decoder family; it does not establish universal scaling or causal performance benefits. All confidence intervals condition on one common training seed.

| axis   | metric             | direction     |
|:-------|:-------------------|:--------------|
| width  | raw                | increasing    |
| width  | cover_full         | decreasing    |
| width  | rank_full          | decreasing    |
| width  | valid_source_ratio | increasing    |
| width  | mean_deletion_kl   | decreasing    |
| width  | cover_4            | non-monotonic |
| width  | rank_4             | decreasing    |
| width  | matched_2          | increasing    |
| width  | matched_4          | increasing    |
| width  | matched_5          | increasing    |
| width  | validation_ce      | decreasing    |
| depth  | raw                | non-monotonic |
| depth  | cover_full         | non-monotonic |
| depth  | rank_full          | non-monotonic |
| depth  | valid_source_ratio | increasing    |
| depth  | mean_deletion_kl   | decreasing    |
| depth  | cover_4            | non-monotonic |
| depth  | rank_4             | non-monotonic |
| depth  | matched_2          | non-monotonic |
| depth  | matched_4          | non-monotonic |
| depth  | matched_5          | non-monotonic |
| depth  | validation_ce      | decreasing    |
| heads  | raw                | non-monotonic |
| heads  | cover_full         | non-monotonic |
| heads  | rank_full          | decreasing    |
| heads  | valid_source_ratio | increasing    |
| heads  | mean_deletion_kl   | decreasing    |
| heads  | cover_4            | increasing    |
| heads  | rank_4             | increasing    |
| heads  | matched_1          | decreasing    |
| heads  | matched_2          | decreasing    |
| heads  | matched_3          | decreasing    |
| heads  | validation_ce      | non-monotonic |
