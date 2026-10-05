# Official vLLM 0.31.0: how we used it

The image is [`vllm/vllm-openai:v0.31.0`](https://hub.docker.com/r/vllm/vllm-openai/tags?name=v0.31.0) on Docker Hub.
It is release 0.31.0 of the vLLM project. We did the tests on the day of the release.

A tag on Docker Hub can change. To get the image that we used, pull it with its digest:

    docker pull vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3

The weights are `local-inference-lab/GLM-5.3-Flash-NVFP4` at `46aaae8a`. They are the same weights as for Jovian
Judgement. We mounted them as read-only.

## Command for the default configuration

    docker run -d --name vllm-official --gpus all --ipc=host --shm-size 32g \
      -p 127.0.0.1:8801:8000 \
      -v <weights dir>:/models:ro \
      -v <cache dir>:/root/.cache \
      -v <template with the thinking switch>:/models/GLM-5.3-Flash-NVFP4/chat_template.jinja:ro \
      -v <config with the added key>:/models/GLM-5.3-Flash-NVFP4/config.json:ro \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      vllm/vllm-openai:v0.31.0 \
      /models/GLM-5.3-Flash-NVFP4 --served-model-name glm-5.3-flash \
      --tensor-parallel-size 4 --max-model-len 524288 \
      --max-num-seqs 16 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90 --no-enable-flashinfer-autotune \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
      --default-chat-template-kwargs '{"reasoning_effort": "max"}'

The time for a start was 617 s. The image loads the weights two times: 163 s for the model and 170 s for the MTP
head. The compile of the kernels was 148 s.

## Command for the configuration with the PCIe all-reduce

Use the same command with these three changes:

1. Add `-e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`.
2. Add `--cudagraph-capture-sizes 1 2 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 96 128 192 256`.
3. Use this value for `--speculative-config`:

       {"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}

The PCIe all-reduce is only serviceable if PCIe peer-to-peer transfers operate correctly. Refer to Appendix A.5 of
the paper.

## Three changes that were necessary for a start

The official vLLM did not start with the settings of Jovian Judgement. We found three problems. Each problem is from
one attempt on one host.

**1. The InstantTensor loader.** With `--load-format instanttensor`, the model loaded in 177 s. Then the load of the
MTP head stopped with this error:

    RuntimeError: buffer_size (1268776960 B) exceeds device memory budget (440532992 B)

InstantTensor 0.2.0 uses half of the free GPU memory as its limit. After the first load, one GPU had only 0.88 GB
of free memory. The default loader has no such limit. Thus we did not use `--load-format`.

**2. The quantization map and the MTP head.** With the default loader, the load of the MTP head stopped with this
error:

    KeyError: 'model.layers.45.mtp_block.mlp.experts.routed_experts.w2_weight_scale'

The quantization map of the checkpoint gives the MTP experts the name `model.language_model.layers.45.mlp.experts`.
The MTP head of the official vLLM uses the name `model.layers.45.mlp.experts` for its search. It finds no entry and
builds experts that are not quantized. Then it gets the MXFP8 scale tensors of the checkpoint and stops. The MTP head of Jovian
Judgement contains a map between the two names.

The correction is a copy of `config.json` with one more key
([`glm53-nvfp4-config-mtp-quant-key.patch`](glm53-nvfp4-config-mtp-quant-key.patch)). Mount the copy on the
`config.json` of the checkpoint. This correction does not change the checkpoint.

**3. The FlashInfer autotune.** The official vLLM tunes the FlashInfer kernels during the start. In our attempt, this
step did not continue. For 11 minutes, the log showed no new line. One worker used 100% of a CPU core in a kernel
launch. The other three workers were idle. We stopped the server. With `--no-enable-flashinfer-autotune`, the
server started. The launcher of Jovian Judgement uses the same option.

## One option that the official vLLM refused

`--moe-backend flashinfer_b12x` selects the B12X expert kernels of FlashInfer. The official vLLM stopped with this
error:

    ValueError: Model sets swiglu_limit=10.0, but the explicitly requested moe_backend='flashinfer_b12x' does not
    apply the SwiGLU clamp.

The `b12x` package is necessary for the option `--moe-backend b12x`. The official image does not contain this
package.

## What the engine selected

These lines are from the start log of the default configuration:

| Function | Selection |
|---|---|
| Attention | `FLASHINFER_MLA_SPARSE_SM120`, KV cache format `fp8_ds_mla` |
| Index of the sparse attention | DeepGEMM, from the copy in the vLLM package |
| NVFP4 experts | `FLASHINFER_CUTLASS` |
| MXFP8 experts of the MTP head | `MARLIN` |
| All-reduce | `PYNCCL` only. The log gives the cause: "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs." |
| Sampler | FlashInfer for top-p and top-k |
| Cache pages | Attention pages of 2,304 tokens. The state page of the linear attention gets 5.11% more memory. |
| Memory | 47.74 GiB for the model on each GPU, 4.79 GiB for the CUDA graphs, 27.93 GiB for the KV cache |
| KV cache | 3,382,087 tokens |

With `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, the log shows this sequence for the all-reduce:

    Using ['FLASHINFER_PCIE_IPC', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'
    Initialized FlashInfer PCIe IPC all-reduce for TP4, hidden_dim=4096, dtype=torch.bfloat16, max_tokens=256.

The PCIe all-reduce of the official vLLM operates on a maximum of 256 tokens. A prefill step has 8,192 tokens. Thus
prefill continues to use NCCL.

## Versions in the image

| Component | Version |
|---|---|
| vLLM | 0.31.0 |
| FlashInfer | 0.7.0.post1 |
| NCCL | 2.30.7 |
| InstantTensor | 0.2.0 |
| Triton, TileLang | 3.7.1, 0.1.12 |
| PyTorch, CUDA | 2.13.0, 13.0 |
| b12x kernels | not in the image |
