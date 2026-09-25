# protassem -- Protein Structure Assembly Pipeline

**English** | [简体中文](README.zh-CN.md)

Given an experimental density map and one or more chain structures, protassem
performs voxelisation, point-cloud sampling, PARE-Net registration and fitting,
unified-queue assembly, and refinement. The final output is an assembled
macromolecular complex in CIF format.

```text
density map (.mrc) + chain structures (.pdb/.cif)
    -> voxelisation -> point sampling -> fitting and assembly -> refinement
    -> assembled complex (.cif)
```

## Requirements

### Python environment

```bash
conda create -n point python=3.8
conda activate point
pip install -r requirements.txt
```

### pareconv

PARE-Net inference depends on `pareconv`, including its CUDA extensions. The
source is included under `protassem/fitting/pareconv_src/` and must be compiled
on the target machine:

```bash
cd protassem/fitting/pareconv_src
pip install -e .
cd pareconv/extensions/pointops/
python setup.py install
```

Compilation requires a compatible PyTorch installation (for example,
PyTorch 1.10.0 with CUDA 11.3), `nvcc`, and `gcc/g++`.

Verify the installation with:

```bash
python -c "import torch; import pointops_cuda; from pareconv.modules.ops import index_select; print('OK')"
```

### Bundled executables

The repository includes the executables used by the pipeline. Grant execute
permission once on Linux:

```bash
chmod +x protassem/core/USalign
chmod +x protassem/sampling/Sample
chmod +x protassem/assembly/domain_parser/domainparser2.LINUX
chmod +x protassem/assembly/domain_parser/dssp
```

### Model weights

The PARE-Net checkpoint `epoch-18.pth.tar` is included under
`protassem/fitting/parenet/weights/`.

## Running the pipeline

### Quick test

The repository does not ship with sample data (`example2` is not part of it).
Point the entry point at any data directory containing a density map (`.mrc`),
structure files (`.pdb`/`.cif`), `resolution.txt` and `contour_level.txt`:

```bash
cd /xiangyux/claude_c_work/demo_reg
python main.py <data_dir> --log
```

The output is written to `<data_dir>/output/`:

```text
example2/output/
+-- pipeline_<timestamp>.log
+-- voxelized/
+-- sampled/
+-- sampled_sources/
+-- assembly/
    +-- final_results/
    |   +-- assembled_complex.cif       filtered final complex
    |   +-- assembled_complex_all.cif   complete accepted assembly
    |   +-- refined_complex.cif         Step 4 result, when triggered
    |   +-- assembly_summary.txt
    |   +-- chains/
    |   +-- domain_chains/
    +-- work/                           intermediate files
```

### Automatic mode

Place the following files in a data directory:

- density maps in `.mrc` format;
- chain structures in `.pdb` or `.cif` format;
- `resolution.txt`;
- `contour_level.txt`.

Run:

```bash
python main.py <data_dir> --log
```

The pipeline reads chain identifiers from the structure contents rather than
from filenames. Multichain inputs are preserved as complexes. Duplicate chain
identifiers are automatically remapped to unique identifiers. CIF output is
used when more than 52 chain identifiers are required.

### Manual mode

```bash
python main.py <density.mrc> <struct_dir> <resolution> <contour> [output_dir] --log
```

## Command-line options

### Switches

| Option | Default | Description |
|---|---:|---|
| `--log` | off | Write a timestamped pipeline log to the output directory. |
| `--log-file <path>` | none | Write the log to an explicit path. |
| `--no-improve-accepted` | off | Disable per-domain local refinement after a chain is accepted. |
| `--no-domain-opt` | off | Disable domain-optimisation fallback for chains that do not pass the chain threshold. |
| `--complex-domain-opt` | off | Split complex inputs into chains, fit domains, and compare merged chain results. |
| `--homo-chain-refine` | off | Enable Step 5 homologous-chain refinement. |
| `--no-pre-screen` | off | Disable the pre-assembly parallel screening of the original poses. |
| `--save-all-attempts` | off | Save every fitting attempt for debugging. |
| `--cleanup` | off | Remove temporary `work/` directories after completion. |
| `--no-domain-split <ids>` | none | Keep selected chain IDs intact, for example `--no-domain-split A,B`. |
| `--no-refine` | off | Disable Step 4 homologous-domain refinement. |

### Numeric parameters

| Option | Default | Description |
|---|---:|---|
| `--chain-threshold` | 0.40 | Acceptance threshold for ordinary chains. |
| `--complex-threshold` | 0.35 | Acceptance threshold for complexes. |
| `--domain-threshold` | 0.40 | Initial threshold for domain fitting. |
| `--domain-min-cc` | 0.35 | Absolute lower bound for domain acceptance. |
| `--complex-min-cc` | 0.25 | Final confidence filter for components in the assembled complex. |
| `--similarity-threshold` | 0.85 | TM-score threshold for chain/domain similarity. |
| `--refine-tm` | 0.75 | TM-score threshold for Step 4 homologous-domain grouping. |
| `--num-processes` | 8 | Number of parallel processes for CC calculation and local optimisation. |
| `--batch-size` | 8 | Number of predictions accumulated before monitoring evaluation. |
| `--mask-radius-factor` | 1.35 | PARENet mask radius as a scale of the gyration radius. |
| `--min-point-distance-factor` | 0.32 | Minimum point spacing inside the mask, in mask radii. |

### Runtime configuration

`--runtime-config <path>` takes a JSON object that overrides the runtime
defaults. Unknown keys are rejected, so a typo fails loudly instead of being
silently ignored.

```bash
python main.py <density.mrc> <struct_dir> <resolution> <contour> \
    --runtime-config runtime.json
```

```json
{"inference_mode": "joint", "allow_tf32": true, "hypothesis_chunk": 64, "tail_pipeline": false}
```

| Key | Default | Description |
|---|---:|---|
| `inference_mode` | `joint` | Registration inference mode. |
| `allow_tf32` | `null` | TF32 policy; `null` derives it from `inference_mode`. |
| `hypothesis_chunk` | `64` | Pose hypotheses scored per chunk. Chunking bounds the peak memory of registration; `0` disables it and restores the previous memory profile. |
| `tail_pipeline` | `false` | Enable the tail-pipelined scheduling path. |

**Memory.** `hypothesis_chunk=64` is the default and is the main reason the
registration path now peaks far below its previous footprint (measured on
test/1-4 with `nvidia-smi`: 44.5%-59.8% lower peak, with the same accepted
components). On large inputs the model also tiles its convolution queries at
1024 points, which is output-equivalent but only engages when a stage carries
more than 1024 query points.

## Examples

Run with default settings:

```bash
python main.py example2
```

Recommended production configuration:

```bash
python main.py <data_dir> --log --homo-chain-refine
```

For complex inputs:

```bash
python main.py <data_dir> --log \
  --complex-threshold 0.35 \
  --complex-domain-opt
```

Enable all major refinement options:

```bash
python main.py <data_dir> \
  --log \
  --complex-domain-opt \
  --homo-chain-refine \
  --chain-threshold 0.40 \
  --complex-threshold 0.30 \
  --domain-threshold 0.40 \
  --num-processes 16
```

Run fitting and assembly without Step 4 refinement:

```bash
python main.py <data_dir> --no-refine --log
```

Keep selected chains intact during domain splitting:

```bash
python main.py <data_dir> --no-domain-split A,B --log
```

## Assembly strategy

Chains and domains are processed in a unified queue ordered by radius of
gyration, with larger components considered first:

1. Fit a chain with PARE-Net, perform local optimisation, and accept it when
   the CC threshold is reached.
2. If a chain fails, add its domains immediately to the same queue.
3. Fit domains using independent thresholds and round management for each
   chain.
4. Merge fitted domains back into chains according to residue order.
5. Build both the complete accepted assembly and a component-filtered final
   complex.

See [ALGORITHM.md](ALGORITHM.md) for the algorithm description.

## Repository structure

```text
demo_reg/
|-- main.py                         command-line entry point
|-- compute_cc_mask.py              standalone CC-mask calculation
|-- check_clash.py                  CA-overlap detection
|-- geo_sym_refine.py               homologous-chain refinement CLI
|-- requirements.txt
|-- ALGORITHM.md
|-- protassem/
    |-- pipeline.py                 top-level pipeline orchestration
    |-- core/                       shared structure and scoring utilities
    |-- voxelize/                   density-map voxelisation
    |-- sampling/                   point-cloud sampling
    |-- fitting/                    PARE-Net fitting and local optimisation
    |-- assembly/                   queue scheduling, domain fitting and refinement
```

## External dependencies

| Dependency | Type | Handling |
|---|---|---|
| `pareconv` | CUDA extension | Source included; compile on the target machine. |
| PARE-Net checkpoint | Model weight | Included under `protassem/fitting/parenet/weights/`. |
| USalign | Executable | Included under `protassem/core/`. |
| Sample (VoxEM) | Executable | Included under `protassem/sampling/`. |
| DomainParser and DSSP | Executables | Included under `protassem/assembly/domain_parser/`. |

Except for `pareconv`, which must be compiled for the target CUDA environment,
the required model code and executables are included in the repository.

## Standalone utilities

Calculate CC-mask:

```bash
python compute_cc_mask.py <structure.pdb/cif> <density.mrc> <resolution> [contour]
```

Run homologous refinement independently:

```bash
python geo_sym_refine.py <case_dir>
python geo_sym_refine.py --complex a.cif --density b.mrc --resolution 3.5
```

Check CA overlap:

```bash
python check_clash.py <pdb_directory> [clash_distance] [overlap_ratio_threshold]
```

