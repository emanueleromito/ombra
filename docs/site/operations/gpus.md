# Supported GPUs

The server needs an NVIDIA GPU of the Ampere generation. The image contains code for compute capability 8.0 and 8.6 (`sm_80` and `sm_86`) and nothing else.

| Supported | Compute capability |
| --- | --- |
| A100, A30 | 8.0 |
| A10, RTX 3060 to RTX 3090, RTX A-series workstation cards | 8.6 |

Other GPUs are not supported. That includes Turing (RTX 20xx, T4), Volta (V100), Ada (RTX 40xx, L4, L40), Hopper (H100) and Blackwell. The kernels are not compiled for them, so the server cannot run on them. Support for other generations has not been built or tested, and there is no date for it.

To see what you have:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

The compute capability should read 8.0 or 8.6.

## Driver and toolkit

The image uses the CUDA 12.6 runtime, so the host driver must support CUDA 12.6 or newer. The host also needs the NVIDIA container toolkit so that Docker can pass the GPU in. The compose file asks for one GPU.

## Memory

The server keeps a cache of ciphertext blocks in GPU memory and another in host memory. The defaults are 4096 MiB on the GPU and 2048 MiB on the host. Raise them with `OMBRA_GPU_MB` and `OMBRA_HOST_MB` (or `--gpu-budget-mb` and `--host-budget-mb`) when your collections are larger and the card has room. A block that is not in a cache has to be loaded first, which is slower. `GET /metrics` shows the GPU memory in use and the cache hit rate, see [Monitoring](monitoring.md).

Without a GPU there is no server. The client has a [local mode](../getting-started/local-or-server.md) that runs on the CPU, for development and small collections.
