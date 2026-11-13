# Supported GPUs

The server runs on NVIDIA GPUs of five architectures. The code is compiled for compute capabilities 7.5, 8.0, 8.6, 8.9 and 9.0. Every kernel is tested against OpenFHE on each of them.

| Card | Compute capability | Support |
| --- | --- | --- |
| A10 (development) and A100 (benchmarks) | 8.6 and 8.0 | Reference. The published numbers come from these. |
| L4, L40S, RTX 40xx, H100, H200 | 8.9 and 9.0 | Supported. |
| T4 | 7.5 | Works, with no latency target. It has 16 GB of memory and expansion is about five times slower than on an A100. |
| A30, RTX 30xx, RTX A-series workstation cards | 8.0 and 8.6 | Supported, like the A10 and A100. |

Other GPUs are not supported. That includes Volta (V100) and Blackwell. The kernels are not compiled for them, so the server cannot run on them.

To see what you have:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

The compute capability should read 7.5, 8.0, 8.6, 8.9 or 9.0.

## Driver and toolkit

The image uses the CUDA 12.6 runtime, so the host driver must support CUDA 12.6 or newer. The host also needs the NVIDIA container toolkit so that Docker can pass the GPU in. The compose file asks for one GPU.

## Memory

The server keeps a cache of ciphertext blocks in GPU memory and another in host memory. The defaults are 4096 MiB on the GPU and 2048 MiB on the host. Raise them with `OMBRA_GPU_MB` and `OMBRA_HOST_MB` (or `--gpu-budget-mb` and `--host-budget-mb`) when your collections are larger and the card has room. A block that is not in a cache has to be loaded first, which is slower. `GET /metrics` shows the GPU memory in use and the cache hit rate, see [Monitoring](monitoring.md).

Without a GPU there is no server. The client has a [local mode](../getting-started/local-or-server.md) that runs on the CPU, for development and small collections.
