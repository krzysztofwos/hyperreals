# Exact finite-period Laurent benchmarks

The benchmark compares the proved Lean residue checker, an independent sparse `Fraction` implementation, and SymPy 1.14.0 exact symbolic arithmetic. Workloads cover coprime and shared-factor period refinement, rational Laurent growth and cancellation, support-relative limits, and explicit unknown results from differing or divergent active branches.

Build and run from the repository root:

```bash
lake build residue_checker
uv run --extra benchmark python scripts/benchmark_residues.py --repeat 20 --output benchmarks/residues
```

See the [measurement report](residues.md), [CSV table](residues.csv), and [full JSON results](residues.json). JSON includes every timing sample, exact accepted decisions and final supports, returned rational values, dependency versions, structural metrics, and the native executable hash.

Each backend must match the independent exact reference before timings are saved. Every workload starts from universe support. The checker does not choose a branch to make standard-part extraction succeed. Direct source evaluations also check each emitted cutoff at selected indices in every residue class. These finite checks test executable agreement. Lean supplies the general correctness theorems for its own core.

Native measurements use a persistent JSON-lines session and report process launch/readiness separately. The public `LeanResidueSystem` wrapper currently starts a checker process per request, so the persistent timings do not represent public API end-to-end latency. Python/SymPy use direct calls, and every backend computes an exact sufficient cutoff. The cutoff values can differ.

Peak memory is measured with `resource.getrusage` as a process-lifetime RSS high-water mark for a fresh worker per backend/workload. Lean's native child and its Python transport worker are reported separately. The report documents included imports, validation, warmup, operating-system units, and unsupported measurement platforms. These scopes differ and should not be compared as per-operation allocations.

Timing samples can vary with machine load, caches, and scheduling. The report preserves ranges and repeats and makes no general speed ranking or claim about the performance cost of formal verification.
