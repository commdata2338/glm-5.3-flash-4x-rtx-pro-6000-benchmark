# Jovian Judgement (the vLLM fork by local-inference-lab): how we used it

The image is [`voipmonitor/vllm:jovian-judgement-community-20260904-r24`](https://hub.docker.com/r/voipmonitor/vllm/tags?name=jovian-judgement-community-20260904-r24)
on Docker Hub (release r24). It is the community release
of local-inference-lab for GLM-5.3-Flash on Blackwell GPUs. The section "Release r28.1" gives the later release that
we also measured. The image contains the vLLM fork by local-inference-lab and the
[B12X](https://github.com/local-inference-lab/b12x) kernel package of the same lab. The lab uses the name
Jovian Judgement for the development branch of the fork (`dev/jovian-judgement`) and for its release images. For
the official vLLM, refer
to [`vllm-official.md`](vllm-official.md). The entry point of the image is a launcher. The launcher
reads the settings from the environment and builds the `vllm serve` command.

The weights are [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4) at `46aaae8a`. We downloaded them before the tests and
mounted them as read-only.

## Command

    docker run --rm --init --name vllm-glm53 --gpus all --ipc=host --shm-size 32g \
      -p 127.0.0.1:8801:8000 \
      -v <weights dir>:/models:ro \
      -v <cache dir>:/cache \
      -v <hf cache dir>:/root/.cache/huggingface \
      -e MODEL=/models/GLM-5.3-Flash-NVFP4 -e SERVED_MODEL_NAME=glm-5.3-flash \
      -e HOST=0.0.0.0 -e PORT=8000 \
      -e CACHE_MODE=vram -e KV_CACHE_QUANT=fp8_ds_mla \
      -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      -e "CUDAGRAPH_CAPTURE_SIZES=<see below>" \
      -e TP=4 -e DCP=1 -e MAX_MODEL_LEN=524288 \
      -e MAX_NUM_SEQS=<slots> -e MAX_NUM_BATCHED_TOKENS=8192 \
      -e PREFILL_SCHEDULE_INTERVAL=1 -e FAIRNESS_ENGINE=compute_share -e PREFILL_COMPUTE_SHARE=0.4 \
      -e GPU_MEMORY_UTILIZATION=<fraction> \
      -e SPECULATOR=mtp -e MTP_DEPTH=3 \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e NCCL_P2P_DISABLE=<0 or 1> -e B12X_PCIE_ALLREDUCE=<1 or 0> \
      voipmonitor/vllm:jovian-judgement-community-20260904-r24 \
      --default-chat-template-kwargs '{"reasoning_effort": "max"}'

The JIT cache in `/cache` stays between starts. The time for each of our starts was 5 to 6.5 minutes. The first
start compiles the kernels, and its time is longer.

## The settings that we changed between tests

| Setting | Server with 16 slots (sections 2 to 5) | Server with 8 slots (sections 6 and 7) |
|---|---|---|
| `MAX_NUM_SEQS` | 16 | 8 |
| `CUDAGRAPH_CAPTURE_SIZES` | `1 2 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 96 128 192 256` | `1 2 4 8 12 16 20 24 28 32 40 48 64 96 128 192 256` |
| `GPU_MEMORY_UTILIZATION` | 0.90 | 0.93 |
| `NCCL_P2P_DISABLE`, `B12X_PCIE_ALLREDUCE` | 0, 1 (links on) | 1, 0 (links off), then 0, 1 (links on) |
| Chat template | standard template, with the thinking switch of two lines mounted on it | standard template |
| KV cache | 2,684,584 tokens | 3,388,737 tokens with the links off, 3,465,275 tokens with the links on |

**Capture sizes.** With MTP depth 3, each request in decode adds 4 tokens to a step. Jovian Judgement builds a full CUDA
graph for decode only for capture sizes that are multiples of 4. The largest of these sizes is 4 × the slot count. The
default list of the image (1 2 4 8 16 32 …) has no full graph for some batch sizes. At those concurrencies, decode
uses a path that is much slower. Put each multiple of 4 up to 4 × `MAX_NUM_SEQS` in the list.

**The direct GPU links.** `NCCL_P2P_DISABLE=0` and `B12X_PCIE_ALLREDUCE=1` are the defaults of the image. They are
only serviceable if PCIe peer-to-peer transfers operate correctly. Refer to Appendix A.5 of the paper. With
`B12X_PCIE_ALLREDUCE=0`, the launcher adds `--disable-custom-all-reduce`, and the fork does the all-reduce through NCCL.
The start log gives the path in use:

    Using ['B12X_PCIE', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'

**The thinking switch.** Add one mount to give the test server a correct thinking-off prompt:

    -v <this repo>/recipes/chat_template.thinking-switch.jinja:/models/GLM-5.3-Flash-NVFP4/chat_template.jinja:ro

Apply `glm53-chat-template-thinking-switch.patch` to the `chat_template.jinja` of the checkpoint to make that file.
Then make sure that the template is in effect:

1. Send a chat to `/tokenize` with `enable_thinking: false`.
2. Send the tokens to `/detokenize`.
3. Make sure that the prompt stops with `<|assistant|><think></think>`.
4. Make sure that the prompt has no "Reasoning Effort" line.

## What the engine showed

These are the arguments that were not the defaults, from the start log of the server with 16 slots:

    quantization=modelopt_mixed  load_format=instanttensor  dtype=bfloat16
    attention_backend=B12X  moe_backend=b12x  linear_backend=b12x
    tensor_parallel_size=4  max_model_len=524288  block_size=256
    kv_cache_dtype=fp8  enable_prefix_caching=True  mamba_cache_mode=align
    max_num_seqs=16  max_num_batched_tokens=8192  enable_chunked_prefill=True
    fairness_engine=compute_share  prefill_compute_share=0.4
    speculative_config={method: mtp, num_speculative_tokens: 3, draft_sample_method: probabilistic,
                        rejection_sample_method: standard, moe_backend: marlin, attention_backend: B12X}
    cudagraph_mode=FULL_AND_PIECEWISE
    additional_config={glm53_kda_decode_backend: auto, kda_prefill_backend: flashkda}
    tool_call_parser=glm47  reasoning_parser=glm45

## Release r28.1

The image [`voipmonitor/vllm:jovian-judgement-community-20260908-r28.1`](https://hub.docker.com/r/voipmonitor/vllm/tags?name=jovian-judgement-community-20260908-r28.1)
is a later release of the same fork. It
started with the command and the weights of release r24. Only the image name was different. The first start used
446 s, and the subsequent starts used 333 s to 349 s.

| | Release r24 | Release r28.1 |
|---|---|---|
| Image digest | `sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd` | `sha256:52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f` |
| Source of Jovian Judgement | [`local-inference-lab/vllm@d49385468458`](https://github.com/local-inference-lab/vllm/tree/d49385468458cf97dff0fc8d9c8863f8082abf4f) | [`local-inference-lab/vllm@9ff42d83938e`](https://github.com/local-inference-lab/vllm/tree/9ff42d83938e74018f9c255e8cfa7ca6df6921b0) |
| b12x kernels | [`local-inference-lab/b12x@e3d0ae067f60`](https://github.com/local-inference-lab/b12x/tree/e3d0ae067f607538e3709ac3c30c7042276c6f88) | [`local-inference-lab/b12x@3edbcbce70f4`](https://github.com/local-inference-lab/b12x/tree/3edbcbce70f491741b82f5eab9c1b30b39447228) |
| KV cache with 16 slots | 2,684,584 tokens | 2,578,877 tokens and 2,555,662 tokens in two starts |
| KV cache with 8 slots | 3,465,275 tokens | 3,366,277 tokens |

The source code of release r28.1 has 155 commits that release r24 does not have. Release r24 has 91 commits that
release r28.1 does not have. Release r28.1 has
[284 commits](https://github.com/local-inference-lab/vllm/compare/299ebd094a9c...9ff42d83938e74018f9c255e8cfa7ca6df6921b0) more than
the official `299ebd094a`.

**The checkpoint policy.** Release r28.1 has the argument `--recurrent-checkpoint-policy`, with the values `auto`
(the default), `request_boundaries`, and `aligned`. Release r24 does not have this argument. With our settings,
`auto` selected `request_boundaries`. The start log then contains this line:

    Request-boundary recurrent checkpoint caching is enabled. Supported chat requests retain leading
    instructions, the complete prompt, and the response endpoint.

With this policy, a request can use the cache from three points only. The first point is the end of a system
message at the start of the prompt. The other points are the end of an earlier prompt and the end of an earlier
reply. A request that has a known long context in its user message and a new question does the prefill of the
context again. Section 5 of the paper gives the
measurements.

To get block-aligned states as in release r24, add this argument at the end of the command:

    --recurrent-checkpoint-policy aligned

With this argument, the launcher of the image also sets `--prefix-cache-retention-interval None`, and the KV cache
had 2,495,985 tokens with 16 slots. The argument `--prefix-cache-retention-interval None` with the default policy
did not change the reuse of a prompt.

**The scheduler setting.** The start log of release r28.1 shows `prefill_compute_share=0.4` and no
`fairness_engine` argument. The launcher of release r28.1 reads `FAIRNESS_ENGINE` for compatibility only.

## Versions in the image

| Component | Version |
|---|---|
| vLLM (release r24) | 0.26.1rc0 fork, [`local-inference-lab/vllm@d49385468458`](https://github.com/local-inference-lab/vllm/tree/d49385468458cf97dff0fc8d9c8863f8082abf4f), [220 commits](https://github.com/local-inference-lab/vllm/compare/299ebd094a9c...d49385468458cf97dff0fc8d9c8863f8082abf4f) more than the official `299ebd094a` (25 August 2026) |
| b12x kernels | [`local-inference-lab/b12x@e3d0ae067f60`](https://github.com/local-inference-lab/b12x/tree/e3d0ae067f607538e3709ac3c30c7042276c6f88) |
| FlashInfer | 0.6.18+cu133, [`voipmonitor/flashinfer@1ac6942776b3`](https://github.com/voipmonitor/flashinfer/tree/1ac6942776b383c6b03c7a5805a22e72a3e3349f) |
| FlashKDA | [`vllm-project/FlashKDA@3b225bf26bb8`](https://github.com/vllm-project/FlashKDA/tree/3b225bf26bb8e218928a1fe14751cb48cf31d11b) |
| NCCL | 2.31.2, [`local-inference-lab/nccl-canonical@fb6f40999a2a`](https://github.com/local-inference-lab/nccl-canonical/tree/fb6f40999a2a9e63104d4ae4a84118bce61528f8) |
| PyTorch, CUDA | 2.13.0, 13.3 |

**Where the versions come from.** The image has labels that give the repository and the commit of each component.
Use `docker image inspect` to read them. A tag on Docker Hub can change, and thus the table of the section
"Release r28.1" gives the digest of each image that we used. To get the image of release r24 that we used, pull it
with its digest:

    docker pull voipmonitor/vllm@sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd

The label `org.opencontainers.image.source` gives the build repository of the image:
[`local-inference-lab/blackwell-llm-docker`](https://github.com/local-inference-lab/blackwell-llm-docker). The image
of release r24 also gives a build commit, `10901bcc31e7`. On 4 October 2026, we did not find this commit in that
repository.

The development branch of the fork is [`dev/jovian-judgement`](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement).
On 4 October 2026, the commits of the two releases were not on this branch. The branch had 306 commits that release
r24 does not have, and release r24 had 84 commits that the branch does not have. To read the source code of a
release, use the commit of the table and not the branch.
