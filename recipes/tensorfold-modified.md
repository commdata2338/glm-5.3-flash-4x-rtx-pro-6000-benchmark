# TensorFold modified: changes, build, and tests

TensorFold modified is TensorFold v0.6.2 (`56e2e3ec55bc`) with the 87 patches of recipe release 1.0.1 and with our
changes. The file [`patches/tensorfold-modified-engine.patch`](../patches/tensorfold-modified-engine.patch) contains our
changes: 17 files, 1,640 added lines, and 34 removed lines. Seven of the files are tests. The weights, the drafter,
and all other parts of the recipe are the same as in [`tensorfold.md`](tensorfold.md).

Each change has a setting. With each setting at its default value, TensorFold modified does the same steps as the
release.

## The settings

| Setting | Default | Value in our tests | Function |
|---|---|---|---|
| `TENSORFOLD_GPU_SAMPLE_FULL` | 0 | 1 | For a row with no top_k, top_p 1.0, and no min_p. Each rank finds its two best scores on the GPU and sends 48 bytes for the row. The sampler accepts the best token if its score is sufficiently far from the second score. If not, the row uses the CPU rule of the release. |
| `TENSORFOLD_GPU_SAMPLE_NUCLEUS` | 0 | 1 | For a row with no top_k and a top_p below 1.0. The candidates stay on the GPU. The GPU calculates the integer masses, the sequence, and the top_p limit. A row that is too near to the limit uses the CPU rule. |
| `TENSORFOLD_SAMPLE_SKIP_FUTILE` | 0 | 1 | For top_p 1.0 with no top_k and no min_p. The sampler does not do the step with 16,384 candidates. That step cannot give a sufficient set of candidates. |
| `TF_GLM_BURST_PREFIX` | 0 | 1 | For requests that arrive together and start with the same tokens. One request does the prefill of the shared part. The other requests then copy the engine state at a point on the 64-token grid. |
| `TF_GLM_CACHE_REASONS` | 0 | 1 | The server adds the cause of the cache decision to the log line of each request (`cache_reason=`). |
| `TF_GLM_COMM` | `nccl` | `ipc` | The release engine contains this setting (patches 0058 and 0070), and the release recipe sets it to `nccl`. With `ipc`, the all-gathers between the ranks use CUDA IPC. |
| `TF_GLM_FILL_CREDIT` | `steps` | `steps` | With `time`, the scheduler counts the measured time of prefill steps and decode steps. We did no GPU test of the value `time`. |

The values of `cache_reason` are: `hit`, `burst_hit`, `token_mismatch:<index>`, `off_grid`, `drafter_missing`,
`head_missing`, `evicted`, `no_room`, `admitted_before_state`, `cold`, `reuse_disabled`, and `vision`.

**Protection.** A process does a self-check when it uses one of the two GPU sampler paths for the first time. Each
rank compares random rows with the CPU rule of the release, and the four ranks vote. If one rank finds a difference,
the process sets the two paths to off and writes a line to the log. All ranks must get the same sampler settings. The
start of the server stops if they are different.

**One correction to the release.** With `TF_GLM_SEGPROF` on, release 1.0.1 stopped each request with a Python error
(`timed() got an unexpected keyword argument 'events_only'`). The patch adds the argument to the function. We used
this profiler for the parts of a round in Appendix E.2 of the paper. We made this correction during the session. The eight
groups of settings before the profile used an image without it. The correction changes only code that operates with
the profiler on.

## How the sampler changes keep the replies equal

- The GPU paths use the same keyed noise as the release (splitmix64 and a 64-bit score).
- For top_p 1.0, each rank sends its best token, its second score, and the largest logit of its part. The largest
  two scores of the four ranks then contain each possible winner.
- The sampler accepts a winner only if the score difference is more than 2^-39 of the scores. This is the rule of
  patch 0065 for top_k rows.
- For a top_p below 1.0, the GPU uses the same integer masses as the release (the exponent of the logit difference,
  multiplied by 2^40). It puts the candidates in the same sequence (value down, then token id up).
- The GPU compares the top_p limit with each cumulative mass. If a mass is too near to the limit for 64-bit
  arithmetic, the row goes to the CPU rule, which uses exact fractions.
- Rows that are not certain go to the CPU in one group for each round.

## How to build TensorFold modified

1. In the directory of the recipe, run `scripts/apply-patches.sh`. It clones TensorFold v0.6.2 into `TensorFold/`
   and applies the 87 patches.
2. In `TensorFold/`, apply our patch: `git apply <this repository>/patches/tensorfold-modified-engine.patch`.
3. Build the image of the release recipe (`tensorfold-glm53:1.0.0`).
4. Make a child image that replaces the installed `tensorfold` package with `src/tensorfold` of the patched tree.
5. Set `IMAGE` in `scripts/config.sh` of the recipe to the child image.
6. In the same file, change `export TF_GLM_COMM=nccl` to `export TF_GLM_COMM="${TF_GLM_COMM:-nccl}"`.

The 87 patches change only files below `src/`. Our patch also adds tests in `tests/` and two tools in `tools/`.

For step 4, put this Dockerfile into `TensorFold/` with the name `Dockerfile.modified`:

    FROM tensorfold-glm53:1.0.0
    COPY src/tensorfold /usr/local/lib/python3.12/dist-packages/tensorfold

Then build the child image in `TensorFold/`:

    docker build -f Dockerfile.modified -t tensorfold-glm53:modified .

Our child image had no other change. We compared all 503 files of the installed package with the patched tree, and
they were equal (`raw/tensorfold-modified/package-compare.log`). The kernels, the weights, and the launch table of the
release stay in use.

Start the server with the settings in the environment. The launcher of the recipe gives each variable that starts
with `TENSORFOLD_` or `TF_GLM_` to the container:

    TENSORFOLD_GPU_SAMPLE_FULL=1 TENSORFOLD_GPU_SAMPLE_NUCLEUS=1 TENSORFOLD_SAMPLE_SKIP_FUTILE=1 \
    TF_GLM_BURST_PREFIX=1 TF_GLM_CACHE_REASONS=1 TF_GLM_COMM=ipc ./start.sh

On our host, the time for a start was 182 s, with the kernels and the launch table of the release in the cache.

## Tests that the replies are equal

We did three tests on the release and then on TensorFold modified. `$URL` is the base URL of the server.

**1. The gates of the recipe.** The tool of the recipe stores reference hashes from the release. Then it compares TensorFold
modified with these hashes, at 4, 8, and 16 concurrent requests:

    python3 tools/bench/tf_bench.py --base $URL/v1 --id baseline --ref-tag=-fp8 --reference-dir refs \
      --results-dir results --gate-levels 4,8,16 --no-telemetry --gate-write-ref gate gatelong
    python3 tools/bench/tf_bench.py --base $URL/v1 --id candidate --ref-tag=-fp8 --reference-dir refs \
      --results-dir results --gate-levels 4,8,16 --no-telemetry gate gatelong

**2. A set of 36 requests with a seed.** The requests use greedy rows, top_k rows, top_p rows with no top_k, top_p
1.0 with no top_k, min_p, and three temperatures. The probe compares the token ids of each reply:

    python3 probes/reply_equality.py --base-url $URL --requests probes/reply_equality_requests.jsonl \
      --output release-c8.jsonl --concurrency 8
    python3 probes/reply_equality.py --base-url $URL --requests probes/reply_equality_requests.jsonl \
      --output modified-c8.jsonl --concurrency 8 --reference release-c8.jsonl

**3. Four cold requests with a shared prefix of 42K tokens.**

    python3 probes/burst_reuse.py make --workload burst --lines 2800 --streams 4 --output-tokens 128 \
      --temperature 1 --top-k 20 --top-p .95 --thinking-on max --output burst.json
    python3 probes/burst_reuse.py run --fixtures burst.json --base-url $URL --concurrency 4 --output release.jsonl
    python3 probes/burst_reuse.py run --fixtures burst.json --base-url $URL --concurrency 4 --output modified.jsonl
    python3 probes/burst_reuse.py compare release.jsonl modified.jsonl

Before the GPU tests, a synthetic check on the four GPUs compared 256 rows of each new sampler path with the CPU
rule. The file `tools/sampler_device_check.py` in the patch is this check. It does not load the model.

| Test | Result |
|---|---|
| Synthetic check of the two GPU sampler paths, four ranks | 256 of 256 rows equal for each path |
| Gates of the recipe, all settings at the default | pass |
| Gates of the recipe, the three sampler settings on | pass |
| Gates of the recipe, the settings of the last column of the table above | pass |
| 36 requests, 8 at a time, for nine groups of settings. For three of the groups, also 1 at a time. | 36 of 36 replies equal in each run |
| Four cold requests with a shared prefix, burst reuse on | the reply tokens of the 4 requests equal |
| Tests with no GPU (sampler, cache, and scheduler) | 121 pass (`raw/tensorfold-modified/tests-with-no-gpu.log`) |

The gates contain five requests with tool calls, one request with an image, and prompts of 8K, 32K, and 100K
tokens. We did no equality test with a grammar or with more than 16 concurrent requests.

## Speed tests

The speed tests are the tests of [`method.md`](method.md). For the directory of one group of settings,
`scripts/build_data.py` reads these file names:

| File | Test |
|---|---|
| `settings.json` | the settings of the server |
| `sparkdash-decode-<type>.json`, `sparkdash-prefill.json`, `warm-chat.jsonl` | as in `method.md` |
| `longctx-topk20.log`, `longctx-topk20-run2.log` | long context, top_p 0.95, top_k 20 (two runs) |
| `longctx-topkoff.log` | long context, top_p 0.95, no top_k |
| `longctx-p1.log` | long context, top_p 1.0, no top_k, at 4, 8, 12, and 16 requests |
| `longctx-p1short.log` | 1K context, four streams, top_p 1.0, no top_k, thinking on |
| `longctx-greedy.log` | step 4 of `method.md` |
| `prefill-direct.jsonl` | one cold prompt for each size, with `probes/prefill_direct.py` |
| `rank0.log` | the request log of the server |

The commands for the long-context test with a sampler are:

    python3 probes/longctx.py --base-url $URL --concurrency 4,8,16 --output-tokens 2000 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --thinking-on max --raw longctx-topk20-raw.jsonl > longctx-topk20.log
    python3 probes/longctx.py --base-url $URL --concurrency 4,8,16 --output-tokens 2000 \
      --temperature 1.0 --top-p 0.95 --top-k -1 --thinking-on max --raw longctx-topkoff-raw.jsonl > longctx-topkoff.log
    python3 probes/longctx.py --base-url $URL --concurrency 4,8,12,16 --output-tokens 2000 \
      --temperature 1.0 --top-p 1.0 --top-k -1 --thinking-on max --raw longctx-p1-raw.jsonl > longctx-p1.log
    python3 probes/longctx.py --base-url $URL --concurrency 4 --context-tokens 1000 --output-tokens 400 \
      --temperature 1.0 --top-p 1.0 --top-k -1 --thinking-on max --raw longctx-p1short-raw.jsonl > longctx-p1short.log
    python3 probes/prefill_direct.py --base-url $URL --out prefill-direct.jsonl

The directory `burst/` of the session has the files `burst-<name>.jsonl` of `probes/burst_reuse.py`.

**The profile of a round.** Start TensorFold modified with `TF_GLM_SEGPROF=4` and `TF_GLM_SEGPROF_REPORT=40`. Then
the log of rank 0 gets one line for each 40 measured rounds. The line gives the GPU time of a round for each part of
the model. The profiler measures one of four rounds a second time, and thus the server is slower with this setting.
