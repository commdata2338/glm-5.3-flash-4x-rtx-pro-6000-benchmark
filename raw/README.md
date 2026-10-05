# Raw output

This directory contains the output of the tools and the probes, as the test host wrote it.

- Each directory is one server configuration. The table in `scripts/collect_raw.py` gives the name of each
  directory on the test host, and `data/README.md` gives the names in the `run` column of `data/*.csv`.
- `python3 scripts/build_data.py --results raw` reads this directory and makes `data/*.csv` again. The result is
  equal to the files in `data/`.
- `scripts/collect_raw.py` made this copy. It replaced the paths of the test host with placeholders such as
  `<tensorfold-dir>` and `<weights-dir>`. It made no other change.
- `tensorfold/rank0.log` is the log of the TensorFold engine. Each `done req-...` line gives the prompt tokens, the
  cached tokens, the output tokens, the speed, the rounds, and the accepted draft tokens of one request.
- The `*-raw.jsonl` files contain the time and the token count of each stream event.
- The directories with `jovian-judgement-r24` in the name are Jovian Judgement r24. The name gives the state of the
  direct GPU links and the number of slots.
- The directories with `official` in the name are the two configurations of the official vLLM. A directory with
  `pass2` in the name contains the second run of the sparkDash tests on the same server.
- `container.log` is the log of the official vLLM until the server was ready. `container-full.log` is the log
  of the full test. `docker-run.json` is the command.
- The files `vllm-official-start-*.log` and `vllm-official-b12x-experts-refused.log` are from the starts that gave
  no server. Each file contains the important lines of one start log.
- `<container-address>` is the placeholder for an address of the container network. `<host>` is the placeholder for
  the name of the test host.
- The directories with `session3-top-k` in the name are from the third session. Each directory contains the
  long-context test with top_k 20, with no top_k, and with top_k 20 a second time.
- `tensorfold-modified/` is from the fourth session. Each directory in it is one start of TensorFold modified with one
  group of settings. `settings.json` gives the settings, and `start.log` is the output of the launcher.
- `tensorfold-modified/sampler-burst-ipc/` is the configuration of Appendix A.3 of the paper. Its `rank0.log` also
  contains the requests of the quality run. With the cache causes on, each `done req-...` line has a `cache_reason`.
- `tensorfold-modified/sampler-profiler/rank0.log` contains the `segment profile` lines of Appendix E.2.
- `tensorfold-modified/reply-equality/`, `gates/`, and `burst-test/` contain the equality tests of section 8.2. A file
  name with `release` is from TensorFold with no changes.
- In the files of `tensorfold-modified/` and of `quality-pi/tensorfold-modified/`, some file names and some lines say
  `fork`. During the tests, `fork` was the name of TensorFold modified. We did not change the output of the tests.
- `session5/` is from the fifth session. Each directory is one server configuration.
- `session5/jovian-judgement-r28.1-16-slots/` has the files of two starts of the server with the same settings.
  The three `longctx-p95-*` tests, `cache-check.jsonl`, and the files with `first-start-` in the name are from the
  first start. The other files are from the second start. The two `longctx-sampled*.log` files show that the test
  suite did not do these two steps on this server.
- `session5/jovian-judgement-r28.1-16-slots-dense-retention/` is the start with
  `--prefix-cache-retention-interval None`. `session5/jovian-judgement-r28.1-16-slots-policy-aligned/` is the start
  with `--recurrent-checkpoint-policy aligned`.
- `cache-check.jsonl` and `cache-check-system.jsonl` are the output of `probes/prefix_reuse_check.py`, with the
  context in the user message and in a system message. In `session5/jovian-judgement-r28.1-8-slots/`,
  `early-look-system.jsonl` is the first check with a system message. The file `cache-check-system.jsonl` of that
  directory is a second check. Before this second check, other tests put the system message into the cache.
- The files with `longctx-system-` in the name are the long-context test with the context in a system message.
- `checkpoint-policy.log` contains the lines of the server log of one start that name the rule for the recurrent
  state. `first-start-prefix-cache-hit-rate.log` contains the status lines of the server log during the three
  long-context tests of the first start of release r28.1.
- `session5/cold-requests/` contains the test with four cold requests on the vLLM servers, and the requests.
  `burst-jovian-r24-live.jsonl` is a first attempt, before the probe had the option `--no-token-ids`. The probe
  recorded an error for each request, and the paper does not use this file. After that attempt, the server had the
  prompts in its cache. Thus the requests of the test got a new first line.
- The address `192.168.254.1` in the `run.json` files is the gateway of the sandbox network that the quality
  harness makes (`quality/sandbox.py`). It is not an address of the network of the test host.
- `quality-pi/` contains the quality runs of section 7. `records.jsonl` has one line for each task.
  `samples.jsonl` contains the files that the agent wrote. `graded.json` contains the grades. `relay.jsonl` contains
  the start time and the duration of each model call. `timing.json` gives the start and the end of the run.
- The event streams of the agent are not in this directory. They have a size of approximately 270 MB.
- `chat-template-check.log` is the output of `probes/template_check.py`. `tensorfold-modified/tests-with-no-gpu.log`
  is the output of the tests of TensorFold modified with no GPU. `tensorfold-modified/package-compare.log` compares the
  installed package of the image with the patched source tree.
- Each directory in `quality-pi/` also has `first_calls.jsonl` and `exceptions.jsonl`. `quality/collect_run.py` made
  them from the event streams.
- `NOT_INCLUDED.txt` gives the files that `scripts/collect_raw.py` did not copy, because they contained a private
  pattern. It contains `none` if the script found no such file. It does not show that the copy is complete.
