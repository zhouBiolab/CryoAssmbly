# Tested low-memory inference

This branch preserves the tested implementation based on a141ef6e553cd1ff433121f4e4c765be7880acaa. It applies query tiling (1024 queries) in PAREConv model construction and enables hypothesis scoring in blocks of 64 through an explicit runtime configuration. Both are required to reproduce the tested configuration. This is inference-only; model training is rejected by the tiled forward wrapper.

## Run

Run on the Linux server in the existing point environment, from this checkout:

```bash
/root/miniconda3/envs/point/bin/python main.py /path/to/map.mrc /path/to/structures RESOLUTION CONTOUR /path/to/new-output --runtime-config configs/low_memory_inference.json --homo-chain-refine --log
```

Replace the positional placeholders with real inputs. Use a fresh output directory. The original full-pipeline comparison used default num-processes=8 and batch-size=8, with homo-chain refinement enabled; it did not include complex-domain-opt.

To align the four flags reported for production job DPA063001028, use the following instead (this exact production-parameter combination was not covered by that comparison):

```bash
/root/miniconda3/envs/point/bin/python main.py /path/to/map.mrc /path/to/structures RESOLUTION CONTOUR /path/to/new-output --runtime-config configs/low_memory_inference.json --complex-domain-opt --homo-chain-refine --num-processes 10 --batch-size 10 --log
```

The external job launcher must pass --runtime-config explicitly. Pushing this branch does not update the production launcher or deploy the code. Check the GPU server log for INFRA_QUERY_TILING and hypothesis_chunk=64. Omitting the JSON leaves hypothesis scoring at its original default and does not reproduce the tested low-memory configuration.

## Full-pipeline validation reported by the user

All eight runs exited successfully with the required products present. GPU memory is the sampled MIG Memory-Usage from nvidia-smi, not PyTorch allocator statistics.

| Case | Original MiB | Optimized MiB | Reduction | Wall seconds, original / optimized |
|---|---:|---:|---:|---:|
| 1 | 6305 | 2853 | 54.8% | 173 / 178 |
| 2 | 6839 | 2749 | 59.8% | 1106 / 1474 |
| 3 | 6453 | 3107 | 51.9% | 2315 / 2989 |
| 4 | 5597 | 3107 | 44.5% | 111 / 110 |

Accepted components/domains were unchanged. Final original-frame RMS/max differences in angstroms were 0.0718/0.2434, 0.1454/0.4593, 0.6351/5.0705 and 0.1634/0.3925 respectively. This is not a bitwise-equivalent or proven accuracy-neutral implementation. Case 3 requires particular attention. More candidate exploration accompanied the increased time in cases 2 and 3; the causal contributions of scheduling and numerical changes were not isolated.

The audit identified hypothesis scoring as the main memory-saving mechanism on these inputs; query tiling did not trigger in case 2. This branch nevertheless retains the exact tested query wrapper rather than silently substituting an untested simplified variant. Weight tensors, point counts, neighbor sets and hypothesis counts are unchanged.

Measure memory with nvidia-smi -lms 100 in a separate session and retain its raw output. Sampling can miss short spikes; the measured peak is not a guaranteed minimum GPU capacity. No benchmark outputs, input structures or private server logs are included in this commit.

## Reproduce the original baseline

Use a separate checkout at a141ef6e553cd1ff433121f4e4c765be7880acaa without the low-memory JSON. Setting hypothesis_chunk=0 on this branch disables hypothesis tiling only; it does not remove the query wrapper.
