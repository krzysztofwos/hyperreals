#!/usr/bin/env python3

"""Evaluation suite for the hyperreal number system with metrics generation."""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperreal import HyperrealSystem

TASK_NAME_MAP = {
    "near_standard_limits": "Near-Standard Limits",
    "finite_vs_cofinite": "Finite vs. Cofinite",
    "alt_sign_conflict_pack": "Alternating Sign",
    "sin_sign_grid_k0_to_5": "Sine Grid",
    "epsilon_cross_links": "Cross-Links",
    "intersection_growth_pack": "Intersection Growth",
    "equality_finite_filters": "Equality Filters",
}


@dataclass
class TaskDelta:
    """Metrics delta for a single evaluation task."""

    name: str
    # PUF deltas
    puf_contains: int
    unit_true: int
    unit_false: int
    fip_checks: int
    fip_blocks: int
    meets: int
    cofinite_includes: int
    finite_excludes: int
    fastpath_true: int
    fastpath_false: int
    cache_true_hits: int
    cache_false_hits: int
    sat_calls: int
    choice_commits: int
    committed_size_delta: int
    max_committed_true_delta: int
    # SAT deltas
    sat_solves: int
    unit_props: int
    decisions: int

    def display_name(self) -> str:
        return TASK_NAME_MAP.get(self.name, self.name.replace("_", " "))

    def to_row(self) -> List[str]:
        return [
            self.name,
            str(self.puf_contains),
            str(self.unit_true),
            str(self.unit_false),
            str(self.fip_checks),
            str(self.fip_blocks),
            str(self.meets),
            str(self.cofinite_includes),
            str(self.finite_excludes),
            str(self.fastpath_true),
            str(self.fastpath_false),
            str(self.cache_true_hits),
            str(self.cache_false_hits),
            str(self.sat_calls),
            str(self.choice_commits),
            str(self.committed_size_delta),
            str(self.max_committed_true_delta),
            str(self.sat_solves),
            str(self.unit_props),
            str(self.decisions),
        ]

    def to_tex_row(self) -> List[str]:
        """Generate a compact row for the LaTeX table."""
        # Metrics selection
        fast_path = self.fastpath_true + self.fastpath_false
        cache = self.cache_true_hits + self.cache_false_hits

        return [
            self.display_name(),
            str(self.puf_contains),
            str(fast_path),
            str(cache),
            str(self.sat_calls),  # Disambiguations
            str(self.choice_commits),
            str(self.committed_size_delta),
            str(self.sat_solves),
            str(self.decisions),
        ]


def _capture_stats(sys: HyperrealSystem) -> Tuple[Dict, Dict]:
    """Capture current PUF and SAT statistics."""
    puf_stats = {
        "puf_contains": sys.puf.stats.puf_contains,
        "unit_true": sys.puf.stats.unit_true,
        "unit_false": sys.puf.stats.unit_false,
        "fip_checks": sys.puf.stats.fip_checks,
        "fip_blocks": sys.puf.stats.fip_blocks,
        "meets": sys.puf.stats.meets,
        "cofinite_includes": sys.puf.stats.cofinite_includes,
        "finite_excludes": sys.puf.stats.finite_excludes,
        "fastpath_true": sys.puf.stats.fastpath_true,
        "fastpath_false": sys.puf.stats.fastpath_false,
        "cache_true_hits": sys.puf.stats.cache_true_hits,
        "cache_false_hits": sys.puf.stats.cache_false_hits,
        "sat_calls": sys.puf.stats.sat_calls,
        "choice_commits": sys.puf.stats.choice_commits,
        "max_committed_true": sys.puf.stats.max_committed_true,
        "committed_size": len(sys.puf._committed_true),
    }
    sat_stats = {
        "sat_solves": sys.puf.sat.solves,
        "unit_props": sys.puf.sat.unit_props,
        "decisions": sys.puf.sat.decisions,
    }
    return puf_stats, sat_stats


def _compute_delta(
    before_puf: Dict, after_puf: Dict, before_sat: Dict, after_sat: Dict, name: str
) -> TaskDelta:
    """Compute metrics delta between before and after states."""
    return TaskDelta(
        name=name,
        puf_contains=after_puf["puf_contains"] - before_puf["puf_contains"],
        unit_true=after_puf["unit_true"] - before_puf["unit_true"],
        unit_false=after_puf["unit_false"] - before_puf["unit_false"],
        fip_checks=after_puf["fip_checks"] - before_puf["fip_checks"],
        fip_blocks=after_puf["fip_blocks"] - before_puf["fip_blocks"],
        meets=after_puf["meets"] - before_puf["meets"],
        cofinite_includes=after_puf["cofinite_includes"]
        - before_puf["cofinite_includes"],
        finite_excludes=after_puf["finite_excludes"] - before_puf["finite_excludes"],
        fastpath_true=after_puf["fastpath_true"] - before_puf["fastpath_true"],
        fastpath_false=after_puf["fastpath_false"] - before_puf["fastpath_false"],
        cache_true_hits=after_puf["cache_true_hits"] - before_puf["cache_true_hits"],
        cache_false_hits=after_puf["cache_false_hits"] - before_puf["cache_false_hits"],
        sat_calls=after_puf["sat_calls"] - before_puf["sat_calls"],
        choice_commits=after_puf["choice_commits"] - before_puf["choice_commits"],
        committed_size_delta=after_puf["committed_size"] - before_puf["committed_size"],
        max_committed_true_delta=after_puf["max_committed_true"]
        - before_puf["max_committed_true"],
        sat_solves=after_sat["sat_solves"] - before_sat["sat_solves"],
        unit_props=after_sat["unit_props"] - before_sat["unit_props"],
        decisions=after_sat["decisions"] - before_sat["decisions"],
    )


# ---- Evaluation Tasks ----


def task_near_standard_limits(sys: HyperrealSystem) -> TaskDelta:
    """Near-standard analytical reasoning (no PUF queries)."""
    before_puf, before_sat = _capture_stats(sys)

    # Derivative of x^3 at 2
    two = sys.constant(2.0)
    eps = sys.infinitesimal()
    f_x = two * two * two
    f_x_plus_eps = (two + eps) * (two + eps) * (two + eps)
    derivative = (f_x_plus_eps - f_x) / eps
    st = derivative.standard_part()
    assert abs(st - 12.0) < 1e-9

    # Limit of sin(x)/x as x→0
    limit_val = (sys.sin(eps) / eps).standard_part()
    assert abs(limit_val - 1.0) < 1e-9

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "near_standard_limits"
    )


def task_finite_vs_cofinite(sys: HyperrealSystem) -> TaskDelta:
    """Fast-paths: finite excludes and cofinite includes."""
    before_puf, before_sat = _capture_stats(sys)

    n = sys.infinite()  # n (omega)
    ten = sys.constant(10.0)
    seven = sys.constant(7.0)
    zero = sys.constant(0.0)
    recip_n = sys.infinitesimal()  # 1/n

    # Finite sets (should be excluded)
    _ = n < ten  # n < 10
    _ = n == seven  # n = 7

    # Cofinite sets (should be included)
    _ = n > ten  # n > 10
    _ = recip_n > zero  # 1/n > 0

    # Empty set (finite exclude)
    _ = recip_n == zero  # 1/n = 0

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "finite_vs_cofinite"
    )


def task_alt_sign_conflict_pack(sys: HyperrealSystem) -> TaskDelta:
    """Oscillation: commit (-1)^n > 0 then ask (-1)^n < 0 and == 0 (contradictions)."""
    before_puf, before_sat = _capture_stats(sys)

    alt = sys.alt()
    zero = sys.constant(0.0)

    # Choose positive branch
    _ = alt > zero

    # Try contradictory queries (should be false)
    _ = alt < zero
    _ = alt == zero

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "alt_sign_conflict_pack"
    )


def task_sin_sign_grid_k0_to_5(sys: HyperrealSystem) -> TaskDelta:
    """For k=0..5 commit sin(n+k)>0 then ask sin(n+k)<0 (UNSAT checks)."""
    before_puf, before_sat = _capture_stats(sys)

    n = sys.infinite()  # n (omega)
    zero = sys.constant(0.0)

    # First commit all as positive
    for k in range(6):
        k_const = sys.constant(float(k))
        sin_n_k = sys.sin(n + k_const)
        _ = sin_n_k > zero  # choose positive

    # Then try contradictory queries
    for k in range(6):
        k_const = sys.constant(float(k))
        sin_n_k = sys.sin(n + k_const)
        _ = sin_n_k < zero  # contradictory

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "sin_sign_grid_k0_to_5"
    )


def task_epsilon_cross_links(sys: HyperrealSystem) -> TaskDelta:
    """ε<0, then 0<ε (implies ε>0 by cross-link); finally ε<0 again (UNSAT)."""
    before_puf, before_sat = _capture_stats(sys)

    eps = sys.infinitesimal()
    zero = sys.constant(0.0)

    # First commit ε < 0
    _ = eps < zero

    # Then 0 < ε (cross-link implies ε > 0)
    _ = zero < eps

    # Finally try ε < 0 again (should be contradictory)
    _ = eps < zero

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "epsilon_cross_links"
    )


def task_intersection_growth_pack(sys: HyperrealSystem) -> TaskDelta:
    """Accumulate several cofinite truths and sine-positives to blow up ∩-closure."""
    before_puf, before_sat = _capture_stats(sys)

    n = sys.infinite()  # n (omega)
    eps = sys.infinitesimal()
    zero = sys.constant(0.0)

    # Cofinite truths
    _ = eps > zero  # 0 < 1/n
    five = sys.constant(5.0)
    ten = sys.constant(10.0)
    twenty = sys.constant(20.0)
    _ = n > five  # 5 < n
    _ = n > ten  # 10 < n
    _ = n > twenty  # 20 < n

    # Several sine positives (choice commitments)
    for k in range(6, 12):
        k_const = sys.constant(float(k))
        sin_n_k = sys.sin(n + k_const)
        _ = sin_n_k > zero

    # Some mixed comparisons (a<b vs b>a cross-links)
    sin_n = sys.sin(n)
    cos_n = sys.cos(n)
    _ = cos_n < sin_n
    _ = sin_n > cos_n  # consistent with cross-link

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "intersection_growth_pack"
    )


def task_equality_finite_filters(sys: HyperrealSystem) -> TaskDelta:
    """Equalities that generate finite-set excludes but avoid SAT work."""
    before_puf, before_sat = _capture_stats(sys)

    n = sys.infinite()  # n (omega)
    eps = sys.infinitesimal()
    zero = sys.constant(0.0)

    # Finite equalities
    for k in [3, 42, 0, 7]:
        k_const = sys.constant(float(k))
        _ = n == k_const

    # 1/n = 0 (empty set)
    _ = eps == zero

    after_puf, after_sat = _capture_stats(sys)
    return _compute_delta(
        before_puf, after_puf, before_sat, after_sat, "equality_finite_filters"
    )


# ---- Evaluation Driver ----

EVAL_COLUMNS = [
    "Task",
    "PUF contains",
    "Unit true",
    "Unit false",
    "FIP checks",
    "FIP blocks",
    "Meets",
    "Cofinite includes",
    "Finite excludes",
    "Fastpath true",
    "Fastpath false",
    "Cache true hits",
    "Cache false hits",
    "SAT calls",
    "Choice commits",
    "Committed Δ",
    "Max committed Δ",
    "SAT solves",
    "Unit propagations",
    "Decisions",
]


def write_csv(results: List[TaskDelta], path: Path) -> None:
    """Write results to CSV file."""
    with path.open("w", encoding="utf-8") as f:
        # Header
        f.write(",".join(EVAL_COLUMNS) + "\n")
        # Data rows
        for td in results:
            f.write(",".join(td.to_row()) + "\n")


def write_tex(results: List[TaskDelta], path: Path) -> None:
    """Write results to LaTeX table."""
    header = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{lrrrrrrrr}
\hline
Task & Queries & Fast & Cache & SAT Calls & Commits & Size $\Delta$ & Solves & Dec. \\ \hline
"""
    footer = r"""\hline
\end{tabular}
\caption{Evaluation metrics for hyperreal operations. Tasks exercise cofinite/finite fast paths, trichotomy conflicts, and intersection-closure growth.}
\label{tab:metrics}
\end{table}
"""
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for td in results:
            row = td.to_tex_row()
            latex_row = " & ".join(row)
            f.write(latex_row + r" \\ " + "\n")
        f.write(footer)


def write_markdown(results: List[TaskDelta], path: Path) -> None:
    """Write results to Markdown table."""
    with path.open("w", encoding="utf-8") as f:
        # Header
        f.write("| " + " | ".join(EVAL_COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(EVAL_COLUMNS)) + " |\n")
        # Data rows
        for td in results:
            f.write("| " + " | ".join(td.to_row()) + " |\n")


def render_markdown(results: List[TaskDelta]) -> str:
    """Render results as a Markdown table (for console output)."""
    lines: List[str] = []
    lines.append("| " + " | ".join(EVAL_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(EVAL_COLUMNS)) + " |")
    for td in results:
        lines.append("| " + " | ".join(td.to_row()) + " |")
    return "\n".join(lines)


def print_summary(results: List[TaskDelta]) -> None:
    """Print summary statistics."""
    total_queries = sum(td.puf_contains for td in results)
    total_cache_hits = sum(td.cache_true_hits + td.cache_false_hits for td in results)
    total_fastpath = sum(td.fastpath_true + td.fastpath_false for td in results)
    total_finite_excludes = sum(td.finite_excludes for td in results)
    total_cofinite_includes = sum(td.cofinite_includes for td in results)
    total_sat_calls = sum(td.sat_calls for td in results)
    total_fip_checks = sum(td.fip_checks for td in results)
    total_choice_commits = sum(td.choice_commits for td in results)
    total_sat = sum(td.sat_solves for td in results)
    total_unit_props = sum(td.unit_props for td in results)
    total_decisions = sum(td.decisions for td in results)
    total_meets = sum(td.meets for td in results)

    resolved_without_sat = total_cache_hits + total_fastpath
    resolved_with_sat = max(0, total_queries - resolved_without_sat)
    sat_calls_per_sat_query = (
        (total_sat_calls / resolved_with_sat) if resolved_with_sat else 0.0
    )

    print("\n" + "=" * 50)
    print("Summary Statistics")
    print("=" * 50)
    print(f"Total membership queries: {total_queries}")
    print(
        f"Resolved without SAT:     {resolved_without_sat} ({(resolved_without_sat / total_queries) if total_queries else 0.0:.1%})"
    )
    print(
        f"Resolved with SAT:        {resolved_with_sat} ({(resolved_with_sat / total_queries) if total_queries else 0.0:.1%})"
    )
    print(f"Cache hits:               {total_cache_hits}")
    print(f"Fast paths (direct):      {total_fastpath}")
    print(f"  Finite excludes:        {total_finite_excludes}")
    print(f"  Cofinite includes:      {total_cofinite_includes}")
    print(
        f"SAT disambiguation calls: {total_sat_calls} ({sat_calls_per_sat_query:.2f}/SAT-query)"
    )
    print(f"FIP checks:               {total_fip_checks}")
    print(f"Choice commits:           {total_choice_commits}")
    print(f"Total SAT solves:        {total_sat}")
    print(f"Total unit propagations: {total_unit_props}")
    print(f"Total decisions:         {total_decisions}")
    print(f"Total meets (∩):         {total_meets}")


def run_evaluation(*, quiet: bool = False) -> Tuple[List[TaskDelta], HyperrealSystem]:
    """Run all evaluation tasks and collect metrics."""
    sys = HyperrealSystem()

    tasks = [
        task_near_standard_limits,
        task_finite_vs_cofinite,
        task_alt_sign_conflict_pack,
        task_sin_sign_grid_k0_to_5,
        task_epsilon_cross_links,
        task_intersection_growth_pack,
        task_equality_finite_filters,
    ]

    results: List[TaskDelta] = []
    for task in tasks:
        td = task(sys)
        results.append(td)
        if not quiet:
            fast = td.fastpath_true + td.fastpath_false
            cache = td.cache_true_hits + td.cache_false_hits
            print(
                f"- {td.display_name():<20}  queries={td.puf_contains:>2}  fast={fast:>2}  cache={cache:>2}  sat_calls={td.sat_calls:>2}"
            )

    return results, sys


def main():
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluation suite for hyperreal number system"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="metrics.csv",
        help="Base output path; writes .csv/.tex/.md (default: metrics.csv)",
    )
    parser.add_argument(
        "--save-ultrafilter",
        type=str,
        metavar="FILE",
        help="Save final ultrafilter state to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-task and summary output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print an additional detailed summary",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("Running evaluation suite...")

    t0 = time.time()
    results, sys = run_evaluation(quiet=args.quiet)
    dt = time.time() - t0

    # Write output files
    out_path = Path(args.out)
    csv_path = out_path.with_suffix(".csv")
    tex_path = out_path.with_suffix(".tex")
    md_path = out_path.with_suffix(".md")

    write_csv(results, csv_path)
    write_tex(results, tex_path)
    write_markdown(results, md_path)

    if not args.quiet:
        print_summary(results)
        print(f"\nWrote: {csv_path}")
        print(f"Wrote: {tex_path}")
        print(f"Wrote: {md_path}")
        print(f"\nEvaluation time: {dt:.3f}s")

    if args.verbose and not args.quiet:
        print("\n" + "=" * 50)
        print("Per-Task Deltas (Markdown)")
        print("=" * 50)
        print(render_markdown(results))

    if args.save_ultrafilter:
        sys.puf.save(args.save_ultrafilter)
        if not args.quiet:
            print(f"\nSaved ultrafilter to: {Path(args.save_ultrafilter).absolute()}")


if __name__ == "__main__":
    main()
