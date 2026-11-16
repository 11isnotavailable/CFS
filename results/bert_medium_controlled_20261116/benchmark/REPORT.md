# New BERT-Medium family preflight throughput

Fresh random weights, actual WordPiece cache pipeline, batch256, sequence128, BF16 and Flash Attention;50 warmup +200 timed updates per case. The I/O confirmation adds host next-batch wait / asynchronous-copy enqueue timing; device copy completion is not separately isolated.

| stage           | model   |   parameters |   input_tokens_per_second |   seconds_per_update |   gpu_utilization_mean |   peak_allocated_gib |   mean_host_input_wait_seconds |   host_input_wait_fraction |
|:----------------|:--------|-------------:|--------------------------:|---------------------:|-----------------------:|---------------------:|-------------------------------:|---------------------------:|
| measurements    | B1      |     26293434 |             967794.047265 |             0.033858 |              93.923077 |             6.085268 |                     nan        |                 nan        |
| measurements    | B2      |     41404730 |             725495.634285 |             0.045166 |              96.647059 |             7.614766 |                     nan        |                 nan        |
| measurements    | B3      |     59694522 |             509803.306469 |             0.064276 |              97.043478 |             9.226090 |                     nan        |                 nan        |
| measurements    | B5      |     54014266 |             514785.833356 |             0.063654 |              96.521739 |            10.451550 |                     nan        |                 nan        |
| io_confirmation | B1      |     26293434 |             980072.383261 |             0.033434 |              95.461538 |             6.085268 |                       0.000313 |                   0.009360 |
| io_confirmation | B2      |     41404730 |             728006.361268 |             0.045011 |              96.882353 |             7.614766 |                       0.000301 |                   0.006694 |
| io_confirmation | B3      |     59694522 |             510167.657166 |             0.064230 |              97.291667 |             9.226090 |                       0.000359 |                   0.005594 |
| io_confirmation | B5      |     54014266 |             518255.248383 |             0.063228 |              97.000000 |            10.451550 |                       0.000383 |                   0.006058 |

| model   |   tokens_per_second |   gpu_hours_2B | basis                               |
|:--------|--------------------:|---------------:|:------------------------------------|
| B1      |         967794.0473 |         0.5740 | measured                            |
| B2      |         725495.6343 |         0.7658 | measured                            |
| B3      |         509803.3065 |         1.0897 | measured                            |
| B4      |         912180.4062 |         0.6090 | depth extrapolation                 |
| B5      |         514785.8334 |         1.0792 | measured                            |
| B6      |         725495.6343 |         0.7658 | B2 proxy, head dimension unmeasured |
| B7      |         725495.6343 |         0.7658 | B2 proxy, head dimension unmeasured |

{
  "total_training_gpu_hours": 5.649306904725254,
  "optimal_two_gpu_hours": 2.8713232396171406,
  "optimal_three_gpu_hours": 1.948844504775751,
  "limitations": "Short windows; exclude preparation, clipping overhead, evaluation, checkpoint IO and CFS. B4/B6/B7 are estimates; third GPU hypothetical."
}

The original small W256/D12 benchmark is not used for these estimates. Production also clips gradients and evaluates/saves milestones. Short cached input windows are not a guarantee of full-corpus IO or long-run throughput; actual logs record input wait and GPU-hours.
