# An observation establishes the division domain

The increment `h = periodic([0, 1]) * epsilon` is infinitesimal before any observation. It is zero in completions selecting even indices and nonzero in completions selecting odd indices. The program asks whether `h = 0`. A true answer selects the nonzero fallback `epsilon**2`. A false answer establishes that `periodic([0, 1]) * n` is an inverse of `h` in every remaining completion. Both branches compute a literal cubic difference quotient at 2 with standard part 12.

The multiplication expression `((2 + h)**3 - 8) * periodic([0, 1]) * n` has no shared standard part before the observation. Its even branch has limit 0 and its odd branch has limit 12. It represents division by `h` only on the odd branch. The residue language supports division by explicit rational monomials. Its periodic reciprocal here is a multiplication expression whose inverse property is proved on the selected support.

The universal proofs are in [DomainInfinitesimal.lean](../../Hyperreals/DomainInfinitesimal.lean). They establish fixed-completion execution, nonzeroness and infinitesimality of the selected step, equality with the literal quotient, and standard part 12 for every accepted run. They also prove that the original zero step has no inverse on the even branch. The generic polynomial compiler and derivative theorem are separate results in [PolynomialDifferentiation.lean](../../Hyperreals/PolynomialDifferentiation.lean).

To reproduce both branches and their kernel-checked snapshots:

```bash
uv sync --locked --extra dev
lake build
uv run --locked --extra dev python scripts/domain_infinitesimal_case.py --timeout 180
```

To verify the saved instances independently:

```bash
uv run --locked --extra dev python scripts/verify_replay.py examples/domain-infinitesimal/snapshots/domain-fallback --timeout 180
uv run --locked --extra dev python scripts/verify_replay.py examples/domain-infinitesimal/snapshots/domain-nonzero --timeout 180
uv run --locked --extra dev python scripts/verify_replay.py examples/domain-infinitesimal/snapshots/domain-unrefined --timeout 180
```

Each snapshot captures one actual accepted trace, its retained support, the expression tree, and the claimed standard part. Replay checks these mathematical data by Lean kernel reduction. The separate universal program theorem proves the domain obligation. Tests check Python capture against independent rational evaluations, but the correspondence between this Python branch program and the Lean program is tested rather than formally verified.

The unrefined snapshot additionally certifies that no finite real value works across every initial completion and that the computed failure is divergence or disagreement. Its separate native diagnostic reports disagreement between residues 0 and 1 with respective limits 0 and 12. Snapshot replay does not certify the native diagnostic's particular witness fields.
