# Pre-test diagnostic amendment

Recorded after validation lock but before any new residual old-test or
confirmation outputs were computed. The original finite fit is unchanged.

All three variants selected gamma=(0,0,0); the primary remains A3. In addition
to these locked policies, evaluate each variant at fixed gamma=1 for every
budget. Label these `plain_raw`, `cfs_raw`, `shuffled_raw` and report them all
on both cohorts. These are diagnostics of actual residual reranking and cannot
replace the locked policy. No strength/epoch is selected using these results.
This amendment responds to validation fallback, not to a test result, and
does not add fitting runs, model architectures or calibration candidates.
