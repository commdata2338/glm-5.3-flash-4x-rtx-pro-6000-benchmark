# Data

`scripts/build_data.py --results raw` makes the CSV files of this directory from [`../raw/`](../raw/). Three files
are not made by the script. `scripts/make_tables.py` makes the tables of the paper from this directory, and
`scripts/make_figures.py` draws the figures. The body of the paper has short tables, and the appendixes have the full
tables. The two types come from the same files.

## Files that the script makes

| File | Contents |
|---|---|
| `decode_sparkdash.csv` | Decode test of sparkDash: each server, output type, and number of streams |
| `prefill_sparkdash.csv` | Prefill test of sparkDash: each server and prompt size |
| `concurrent_waves.csv` | `longctx.py`: the 1K greedy test and the long-context test, one row for each wave. The probe `long_context_sampled_system_message` is the long-context test with the context in a system message. A row with the status `no_overlap` has no speed: no time interval of more than one second contained the output of all its requests. |
| `prefix_reuse_check.csv` | `prefix_reuse_check.py`: the seconds to the full reply of each step, for each server and for the two forms of the check |
| `chat_reuse_ttft.csv` | `warm_chat.py`: the time to the first token for each case and round |
| `integrity.csv`, `integrity_recheck.csv` | `integrity.py`: the passphrase test, and the test with a limit of 300 tokens |
| `thinking_off_forms.csv` | `thinkoff_probe.py`: the chat form and the raw form on each server |
| `tensorfold_sampling_sensitivity.csv` | The eight sampler tests of Appendix C.5 |
| `tensorfold_round_times.csv` | The time of a decode round, from the request logs of TensorFold and of the TensorFold fork |
| `longctx_estimate_check.csv` | The check of the token estimate of `longctx.py`: the total speed from the probe and the sum of the request speeds from the log, for each wave |
| `gpu_to_gpu_copy.csv` | `p2p_test.py`: the copy test for each GPU pair |
| `b12x_allreduce_limits.csv` | The size limits of the B12X PCIe all-reduce, from the start log of Jovian Judgement r24 (Appendix G) |
| `fork_arms.csv` | The TensorFold fork: the waves of `longctx.py` for each group of settings |
| `fork_cold_burst.csv` | `burst_reuse.py`: four cold requests with a shared prefix, on each server |
| `fork_prefill_direct.csv` | `prefill_direct.py`: two prefill chunk sizes |
| `fork_round_profile.csv` | The report lines of the profiler |
| `quality_pi_tasks.csv` | The quality run: one row for each task and server |
| `quality_pi_summary.csv` | The pass counts and the 95% Wilson intervals |
| `quality_pi_paired.csv` | The paired counts, the difference of the pass rates with its interval, and the exact McNemar test |
| `quality_pi_calls.csv` | The model calls of each run, from the relay log |
| `quality_pi_exceptions.csv` | The tasks that did not complete or have no solution file |

## Files that the script does not make

| File | Source |
|---|---|
| `tensorfold_published_decode.csv`, `tensorfold_published_prefill.csv` | The speed tables in the README of the TensorFold recipe, release 1.0.1 (`bdf4f18`). We typed these values. |
| `gpu_copy_paths.json` | The output of `probes/gpu_copy_paths.py` on the test host |

## The names of the runs

The column `run` of the CSV files has these values.

| Run | Server and configuration | Session |
|---|---|---|
| `tensorfold` | TensorFold | 1 |
| `tensorfold_session3` | TensorFold, the long-context test with top_k 20 and with no top_k | 3 |
| `tensorfold_fork_best` | TensorFold fork, the configuration of Appendix A.3 | 4 |
| `vllm_links_on_16` | Jovian Judgement r24, 16 slots, direct GPU links on, template with the thinking switch | 1 |
| `vllm_links_on_16_session3` | The same configuration, the long-context test with top_k 20 and with no top_k | 3 |
| `vllm_links_off_8`, `vllm_links_on_8` | Jovian Judgement r24, 8 slots, standard template, direct GPU links off and on. The quality run used `vllm_links_on_8`. | 1 and 4 |
| `vllm_links_off_16_before` | Jovian Judgement r24, 16 slots, direct GPU links off, standard template, before the IOMMU change. The paper uses its prefill values in Appendix C.7. | before session 1 |
| `official_default_16`, `official_default_16_pass2` | Official vLLM, default configuration: all tests, and the second run of sparkDash | 2 |
| `official_pcie_16`, `official_pcie_16_pass2` | Official vLLM, tuned configuration: all tests, and the second run of sparkDash | 2 |
| `official_default_16_session5`, `official_pcie_16_session5` | Official vLLM, the two configurations: the long-context test with top_k 20 and with no top_k | 5 |
| `jovian_r281_16` | Jovian Judgement r28.1, 16 slots, the settings of release r24: the test suite without its two long-context steps | 5 |
| `jovian_r281_16_topk` | The same configuration: the long-context test with top_k 20 and with no top_k. These rows have the status `no_overlap`. | 5 |
| `jovian_r281_16_aligned`, `jovian_r281_16_aligned_topk` | Jovian Judgement r28.1, 16 slots, with `--recurrent-checkpoint-policy aligned`: the test suite, and the long-context test with top_k 20 and with no top_k | 5 |
| `jovian_r281_8` | Jovian Judgement r28.1, 8 slots, standard template: the quality run, and the long-context test with the context in a system message | 5 |
| `vllm_links_on_8_run2` | Jovian Judgement r24, 8 slots: the second quality run | 5 |
| `vllm_links_on_8_session5` | Jovian Judgement r24, 8 slots: the long-context test with the context in a system message | 5 |

In the quality files, the run of the official vLLM has the name `official_default_16`.

In `fork_arms.csv`, the column `arm` gives the directory of one start of the TensorFold fork, and the column
`settings` gives the settings that were on. `fork-best` is the configuration of Appendix A.3.

Some values of the paper are not in this directory or in `raw/`. These are the start times of the servers, the GPU
load in Appendix C.5, and the host data in Appendix A.4. They come from our notes of the sessions.
