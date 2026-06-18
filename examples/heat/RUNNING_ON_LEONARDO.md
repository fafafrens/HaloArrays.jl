# Running the multi-GPU example on Leonardo (CINECA)

A **tested** recipe for running [`multigpu_mpi_2d.jl`](multigpu_mpi_2d.jl) — the
one-MPI-rank-per-GPU heat solver — on the Leonardo Booster partition (4× NVIDIA
A100 per node, SLURM, CUDA-aware OpenMPI). Verified end-to-end: 1 and 4 GPUs, with
the global L2 norm bit-identical to the CPU reference.

The setup has four parts that all have to line up: **Julia/depot location**,
**system MPI**, **system HDF5**, and **CUDA runtime** — plus the right **SLURM
launcher**. Each is below, with the gotchas that bite.

> **Why it's fiddly:** compute nodes have **no internet** (downloads must happen on
> the login node), and Julia's bundled `MPICH_jll`/`HDF5_jll`/`CUDA` artifacts must
> be replaced by the *system* CUDA-aware OpenMPI + matching parallel HDF5 + the
> local CUDA toolkit, or things conflict (e.g. `HDF5_jll`→`OpenMPI_jll` 5.x vs the
> system OpenMPI 4.1.6 → `undefined symbol: ompi_instance_count`).

---

## 1. Depot + Julia on a healthy, roomy filesystem

`$HOME` is 50 GB (fine for a depot) and on the *home* storage cluster, separate
from `$WORK`. Put the depot there (or `$WORK` when it's healthy — it's 1 TB):

```bash
echo 'export JULIA_DEPOT_PATH=$HOME/julia_depot' >> ~/.bashrc   # no spaces around '='!
source ~/.bashrc && mkdir -p "$JULIA_DEPOT_PATH"
```

Install the latest Julia (official binary — not the `julia/1.8.2` module, too old;
not Spack):
```bash
curl -fsSL https://install.julialang.org | sh    # juliaup; pick a $WORK/$HOME install path
juliaup add release && juliaup default release    # ≥1.10; CI covers 1.10 & 1.12
```

Clone the repo on the same filesystem:
```bash
cd $HOME && git clone https://github.com/fafafrens/HaloArrays.jl && cd HaloArrays.jl
```

## 2. Modules (CUDA-aware OpenMPI + matching parallel HDF5)

```bash
module load cuda/12.2 \
            openmpi/4.1.6--gcc--12.2.0-cuda-12.2 \
            hdf5/1.14.3--openmpi--4.1.6--gcc--12.2.0-spack0.22
```
Both OpenMPI builds are CUDA-aware (`-cuda-12.2`); the HDF5 is the parallel build
against the same OpenMPI 4.1.6 (ABI-compatible). They coexist — no module conflict.

## 3. Configure the env — **on the LOGIN node** (compute nodes have no internet)

```bash
# find the system parallel libhdf5 the module exposes:
H=$(dirname "$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | while read d; do [ -e "$d/libhdf5.so" ] && echo "$d/libhdf5.so" && break; done)")
echo "HDF5 lib dir = $H"

export JULIA_PKG_PRECOMPILE_AUTO=0     # don't precompile until all prefs are set
julia --project=examples -e '
  using Pkg
  Pkg.develop(path=".")                              # HaloArrays is local, not registered
  Pkg.add(["CUDA","MPIPreferences","Preferences"])
  using MPIPreferences; MPIPreferences.use_system_binary()        # MPI → system OpenMPI 4.1.6
  using CUDA; CUDA.set_runtime_version!(v"12.2"; local_toolkit=true)  # use the cuda/12.2 module'
# point HDF5.jl at the SYSTEM parallel HDF5 (bypasses HDF5_jll/OpenMPI_jll clash):
cat >> examples/LocalPreferences.toml <<EOF

[HDF5]
libhdf5 = "$H/libhdf5.so"
libhdf5_hl = "$H/libhdf5_hl.so"
EOF
unset JULIA_PKG_PRECOMPILE_AUTO

julia --project=examples -e 'using Pkg; Pkg.precompile()'   # downloads + builds, into the depot
```

`examples/LocalPreferences.toml` should now hold **both** `[MPIPreferences]`
(`binary="system"`) and `[HDF5]`. (A `✗ HDF5_jll` during precompile is **harmless**
— HDF5.jl uses the system lib and never loads that JLL.)

## 4. Verify (on a GPU node)

```bash
srun -A <ACCOUNT> -p boost_usr_prod --qos=boost_qos_dbg \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --time=00:10:00 --pty bash
module load cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 hdf5/1.14.3--openmpi--4.1.6--gcc--12.2.0-spack0.22
export OMPI_MCA_opal_cuda_support=1
julia --project=examples -e 'using MPI, CUDA, HDF5;
  println("MPI=",MPI.MPI_LIBRARY," ",MPI.MPI_LIBRARY_VERSION," CUDA-aware=",MPI.has_cuda());
  println("CUDA.functional=",CUDA.functional(),"  HDF5=",HDF5.API.h5_get_libversion())'
```
Want: `MPI=OpenMPI 4.1.6 … CUDA-aware=true`, `CUDA.functional=true`, `HDF5=1.14.3`.
(If `MPI_LIBRARY` says **MPICH**, the system-MPI preference didn't stick — re-run
`use_system_binary()` + `Pkg.precompile()`. The `libcublasLt`/`libnvJitLink`
"loaded from a system path" warnings are harmless.)

## 5. Run

The launcher is the last site-specific knob. On Leonardo (OpenMPI built for PMIx,
but `pmix`/`pmix_v5` won't load) the working one is **`srun --mpi=pmix_v3`**:

```bash
# 1 GPU (smoke test):
srun -A <ACCOUNT> -p boost_usr_prod --qos=boost_qos_dbg \
     --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=8 --time=00:10:00 \
     --mpi=pmix_v3 julia --project=examples examples/heat/multigpu_mpi_2d.jl

# 4 GPUs (real multi-GPU, GPU-to-GPU exchange):
srun -A <ACCOUNT> -p boost_usr_prod --qos=boost_qos_dbg \
     --nodes=1 --ntasks=4 --gpus-per-node=4 --cpus-per-task=8 --time=00:10:00 \
     --mpi=pmix_v3 julia --project=examples examples/heat/multigpu_mpi_2d.jl
```
(Have `cuda`/`openmpi`/`hdf5` modules loaded and `OMPI_MCA_opal_cuda_support=1`
exported in the submitting shell — `srun` passes the environment to the job.)

**Verified output:**
```
backend=CUDA (rank 0 → gpu 0)   ranks=1   topo=(1, 1)   global=256x256
done: …   global ‖u‖₂ = 127.691943    OK         # 1 GPU
backend=CUDA (rank 0 → gpu 0)   ranks=4   topo=(2, 2)   global=512x512
done: …   global ‖u‖₂ = 255.845833    OK         # 4 GPUs — matches the CPU reference
```

For an `sbatch` script, put the same `module load` + `export` + `srun --mpi=pmix_v3
… julia …` in a batch file with `#SBATCH` for `--account`, `--nodes`,
`--ntasks-per-node=4`, `--gpus-per-node=4`, `--time`.

---

## Gotchas (each one cost real time)

- **`$WORK` storage incidents** cause random I/O failures during precompile (writes
  thousands of cache files). If a *different* package fails each retry → it's the
  filesystem, not you: use a healthy area (`$HOME`/`$PUBLIC`), or wait it out.
- **Compute nodes have no internet** → do **all** `Pkg.add`/`precompile`/downloads on
  the **login node**; only *run* on compute nodes (everything must be precached).
- **System MPI** is required for CUDA-aware exchange (`MPIPreferences.use_system_binary()`),
  and it **forces system HDF5** — the bundled `HDF5_jll` pulls an `OpenMPI_jll` (5.x)
  that clashes with the system OpenMPI 4.1.6. Point HDF5.jl at the system lib (step 3).
- **CUDA precompiled on a login node** has no driver → set the runtime:
  `CUDA.set_runtime_version!(v"12.2"; local_toolkit=true)`.
- **No spaces around `=`** in `~/.bashrc` exports (`export X=Y`, not `export X =Y`).
- **Launcher:** `srun --mpi=pmix` fails here; use **`--mpi=pmix_v3`** (`srun --mpi=list`
  to see options). `pmi2` fails because this OpenMPI is PMIx-only.

These steps (system OpenMPI + system HDF5 + CUDA local toolkit + `--mpi=pmix_v3`) are
the full, working configuration as run on Leonardo.
