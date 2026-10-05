# TensorFold recipe: how we used it

The recipe is [Aevonix GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold),
release 1.0.1 (`bdf4f18`). One command, `./start.sh`, does these steps:

1. It builds the image.
2. It downloads the weights.
3. It compiles and tunes the kernels.
4. It starts four ranks in one container.
5. It does a short test of the API.

## What the recipe builds and serves

| | |
|---|---|
| Engine | TensorFold v0.6.2 (`56e2e3ec55bc`) with 87 patches: 55 from Mia's AI Lab, 32 from Aevonix |
| Base image | `nvcr.io/nvidia/pytorch` 26.07 (PyTorch 2.13) |
| Weights | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` at `78353f1f6eb2` (164 GiB on disk) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` at `bf582e4eacc1` (CC BY-NC-ND 4.0), and copy drafts |
| Dense layers, KV cache | FP8, FP8 |
| Slots, context | 40, 1,048,576 tokens |
| KV pool | Setting of 60 GiB. The server showed 5,097,472 tokens. |
| Default sampler settings | temperature 1.0, top_p 0.95, top_k 20. A request with no `top_k` gets top_k 20. |
| Default thinking mode | off (`THINKING=0`). If a request sets thinking to on, the reasoning effort is high. |
| Draft policy | `fnc7:0.3` |
| GPU transfers | NCCL with PCIe peer-to-peer for collectives, a second NCCL communicator with peer-to-peer off for paired sends, and CUDA IPC exchanges with the copy engines |

All items are the release defaults, but `scripts/local.sh` has these two lines:

    HOST=127.0.0.1            # the API is only on the loopback address
    GPU_MAX_USED_MIB=2000     # refer to the next section

## Our one change to the recipe

`scripts/prepare.sh` stops if a GPU has 1,000 MiB or more in use. One GPU on our host supplies the
desktop display (0.4 GB to 1.2 GB). Thus we replaced the constant with a setting:

    -    (( used < 1000 )) || fail E_GPU_BUSY ...
    +    (( used < ${GPU_MAX_USED_MIB:-1000} )) || fail E_GPU_BUSY ...

The recipe calculates the image label from the contents of `prepare.sh`. Thus this change gives a different label
(`2ac85f7a683e18bb` on our host). The contents of the image do not change.

## What release 1.0.1 changed

The image of release 1.0.0 did not contain the C/CUDA headers of TensorFold. Thus the CUDA IPC extension did not
compile, and the gathers of the index split used NCCL. Release 1.0.1 copies the headers into the image. On our host,
the start log then shows:

    building CUDA extension tensorfold_ipc_v3
    the index split's pools over CUDA IPC (copy engines, up to 2048 KiB a rank)

We have no measurement of release 1.0.0. It did not start on our host before the IOMMU change (Appendix A.5 of the
paper).

## Times on this host

| Step | Time |
|---|---|
| `scripts/prepare.sh`: compile the kernels and tune the launch table (15 entries for 5 kernels) | 212 s |
| `./start.sh` until the API is ready: load the weights and build the other extensions | 372 s |
| First request after the start | 22 s for a 2K prompt (compile that occurs one time). Subsequent requests have the usual speed. |

## Checks that the recipe does, and one that it cannot do

The launcher makes sure that these conditions are correct:

- four idle GPUs, each with 90 GB free
- a power limit of 250 W for each GPU
- sufficient disk space
- CUDA peer access between each pair of GPUs

The peer access check asks the driver if it has the peer access capability. On our host, the result of this check was satisfactory. But at
the same time, the IOMMU prevented each peer write. The ranks then stopped after the NCCL start, and they gave no
error. Run `probes/p2p_test.py` before you start the recipe.
