#!/usr/bin/env python3
"""Compare exact finite-period Laurent engines with correctness-gated timings.

Build `lake build residue_checker`, install `uv sync --extra benchmark`, then
run `uv run --extra benchmark python scripts/benchmark_residues.py --repeat 5`.
The native backend stays alive across warmup and repetitions. Each backend/case
gets its own Python worker so process peak-RSS measurements have clear scope.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from math import ceil, lcm
from pathlib import Path
import platform
import select
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".lake/build/bin/residue_checker"


def constant(value):
    value = Fraction(value)
    return ["const", str(value.numerator), str(value.denominator)]


def periodic(values):
    return ["periodic", [[str(Fraction(v).numerator), str(Fraction(v).denominator)] for v in values]]


def binary(tag, left, right):
    return [tag, left, right]


def power(expression, exponent):
    if exponent == 0:
        return constant(1)
    if exponent == 1:
        return expression
    half = power(expression, exponent // 2)
    square = binary("mul", half, half)
    return binary("mul", square, expression) if exponent % 2 else square


def divide(expression, coefficient, exponent):
    coefficient = Fraction(coefficient)
    return ["divMonomial", expression, str(coefficient.numerator), str(coefficient.denominator), str(exponent)]


def query(left, right, op="eq", choice=True):
    return {"left": left, "right": right, "op": op, "choice": choice}


def workloads():
    n, epsilon = ["index"], ["invn"]
    two, three, five, seven = (periodic(range(p)) for p in (2, 3, 5, 7))
    cases = [
        {"name": "infinitesimal_order", "queries": [
            query(epsilon, constant(0), "lt", False),
            query(epsilon, constant(Fraction(1, 10**6)), "lt"),
            query(binary("mul", n, epsilon), constant(1)),
        ], "expression": divide(binary("sub", power(binary("add", constant(2), epsilon), 3), constant(8)), 1, -1),
         "expected": (1, 1, Fraction(12))},
        {"name": "coprime_2_3", "queries": [query(two, constant(1)), query(three, constant(2))],
         "expression": periodic(range(6)), "expected": (6, 1, Fraction(5))},
        {"name": "coprime_3_5_7", "queries": [query(three, constant(2)), query(five, constant(3)), query(seven, constant(4))],
         "expression": binary("add", binary("add", three, five), seven), "expected": (105, 1, Fraction(9))},
        {"name": "shared_factors_4_6", "queries": [
            query(periodic(range(4)), constant(1)), query(periodic(range(6)), constant(2)),
            query(periodic(range(6)), constant(3)),
        ], "expression": periodic(range(12)), "expected": (12, 1, Fraction(9))},
        {"name": "support_relative_limit", "queries": [query(two, constant(1))],
         "expression": binary("add", periodic([1, -1]), binary("mul", three, epsilon)),
         "expected": (2, 1, Fraction(-1))},
        {"name": "unknown_differing_limits", "queries": [query(two, constant(1))],
         "expression": periodic([1, 2, 3]), "expected": (2, 1, None)},
        {"name": "unknown_divergent_branch", "queries": [query(three, constant(0))],
         "expression": binary("add", binary("mul", periodic([0, 1]), n), constant(3)),
         "expected": (3, 1, None)},
    ]
    for degree in (4, 8, 12):
        expression = divide(power(binary("add", n, constant(1)), degree), 1, degree)
        cases.append({"name": f"laurent_growth_{degree}", "queries": [query(constant(1), expression, "lt")],
                      "expression": expression, "expected": (1, 1, Fraction(1))})
    grown = power(binary("add", n, epsilon), 12)
    cancelled = binary("add", binary("sub", grown, grown), constant(Fraction(7, 3)))
    cases.append({"name": "exact_cancellation_12", "queries": [query(cancelled, constant(Fraction(7, 3)))],
                  "expression": cancelled, "expected": (1, 1, Fraction(7, 3))})
    sum_tables = binary("add", binary("add", two, three), binary("add", five, seven))
    cases.append({"name": "dense_lcm_210", "queries": [query(sum_tables, constant(7), "lt")],
                  "expression": binary("mul", sum_tables, epsilon), "expected": (210, None, Fraction(0))})
    return cases


def expression_period(ast):
    if ast[0] == "periodic":
        return len(ast[1])
    if ast[0] in ("const", "index", "invn"):
        return 1
    if ast[0] == "divMonomial":
        return expression_period(ast[1])
    return lcm(expression_period(ast[1]), expression_period(ast[2]))


def expression_nodes(ast):
    if ast[0] in ("const", "index", "invn", "periodic"):
        return 1
    if ast[0] == "divMonomial":
        return 1 + expression_nodes(ast[1])
    return 1 + expression_nodes(ast[1]) + expression_nodes(ast[2])


def evaluate(ast, n):
    """Independent pointwise interpretation of the input expression syntax."""
    tag = ast[0]
    if tag == "const":
        return Fraction(int(ast[1]), int(ast[2]))
    if tag == "periodic":
        numerator, denominator = ast[1][n % len(ast[1])]
        return Fraction(int(numerator), int(denominator))
    if tag == "index":
        return Fraction(n)
    if tag == "invn":
        return Fraction(1, n)
    if tag == "divMonomial":
        return evaluate(ast[1], n) / (Fraction(int(ast[2]), int(ast[3])) * Fraction(n) ** int(ast[4]))
    a, b = evaluate(ast[1], n), evaluate(ast[2], n)
    if tag == "add":
        return a + b
    if tag == "sub":
        return a - b
    return a * b


def trim(poly):
    return {exponent: coefficient for exponent, coefficient in poly.items() if coefficient}


def sparse_form(ast, residue):
    """Sparse integer powers of n, independent of Lean's dense quotient form."""
    tag = ast[0]
    if tag == "const":
        return trim({0: Fraction(int(ast[1]), int(ast[2]))})
    if tag == "periodic":
        a, b = ast[1][residue % len(ast[1])]
        return trim({0: Fraction(int(a), int(b))})
    if tag == "index":
        return {1: Fraction(1)}
    if tag == "invn":
        return {-1: Fraction(1)}
    if tag == "divMonomial":
        coefficient = Fraction(int(ast[2]), int(ast[3]))
        return {k - int(ast[4]): v / coefficient for k, v in sparse_form(ast[1], residue).items()}
    left, right = sparse_form(ast[1], residue), sparse_form(ast[2], residue)
    result = dict(left) if tag in ("add", "sub") else {}
    if tag == "mul":
        for a, x in left.items():
            for b, y in right.items():
                result[a + b] = result.get(a + b, Fraction(0)) + x * y
    else:
        for exponent, coefficient in right.items():
            result[exponent] = result.get(exponent, Fraction(0)) + (coefficient if tag == "add" else -coefficient)
    return trim(result)


def sparse_sign_cutoff(form):
    if not form:
        return 0, 1
    exponent = max(form)
    leading = form[exponent]
    # For n >= 1, every lower power loses at least one factor of n.
    tail = sum((abs(c) for k, c in form.items() if k != exponent), Fraction(0))
    return (1 if leading > 0 else -1), max(1, ceil(tail / abs(leading)) + 1)


class ExactBackend:
    version = "stdlib Fraction"

    def normalize(self, ast, residue):
        return sparse_form(ast, residue)

    def sign_cutoff(self, form):
        return sparse_sign_cutoff(form)

    def limit(self, form):
        if any(exponent > 0 for exponent in form):
            return None
        return form.get(0, Fraction(0))

    def compare(self, support, request):
        period = lcm(expression_period(request["left"]), expression_period(request["right"]))
        mask, cutoff = [], 1
        difference = binary("sub", request["left"], request["right"])
        for residue in range(period):
            sign, bound = self.sign_cutoff(self.normalize(difference, residue))
            mask.append(sign < 0 if request["op"] == "lt" else sign == 0)
            cutoff = max(cutoff, bound)
        common = lcm(len(support), period)
        selected = [support[r % len(support)] and (mask[r % period] == request["choice"]) for r in range(common)]
        return {"predicate": mask, "cutoff": str(cutoff), "accepted": any(selected),
                "support": selected if any(selected) else None}

    def standard_part(self, support, expression):
        period = lcm(len(support), expression_period(expression))
        limits = [self.limit(self.normalize(expression, r)) for r in range(period) if support[r % len(support)]]
        return limits[0] if limits and limits[0] is not None and all(v == limits[0] for v in limits) else None

    def close(self):
        pass


class SympyBackend(ExactBackend):
    def __init__(self):
        import sympy
        self.sympy = sympy
        self.n = sympy.Symbol("n", positive=True)
        self.version = sympy.__version__

    def normalize(self, ast, residue):
        sp, n, tag = self.sympy, self.n, ast[0]
        if tag == "const":
            return sp.Rational(int(ast[1]), int(ast[2]))
        if tag == "periodic":
            a, b = ast[1][residue % len(ast[1])]
            return sp.Rational(int(a), int(b))
        if tag == "index":
            return n
        if tag == "invn":
            return 1 / n
        if tag == "divMonomial":
            return sp.cancel(self.normalize(ast[1], residue) / (sp.Rational(int(ast[2]), int(ast[3])) * n ** int(ast[4])))
        a, b = self.normalize(ast[1], residue), self.normalize(ast[2], residue)
        return sp.expand(a + b if tag == "add" else a - b if tag == "sub" else a * b)

    def sign_cutoff(self, form):
        sp, n = self.sympy, self.n
        numerator, denominator = sp.fraction(sp.cancel(form))
        if numerator == 0:
            return 0, 1
        p, q = sp.Poly(numerator, n), sp.Poly(denominator, n)

        def bound(poly):
            coefficients = poly.all_coeffs()
            return max(1, int(sp.ceiling(sum(abs(c) for c in coefficients[1:]) / abs(coefficients[0]))) + 1)

        return int(sp.sign(p.LC() * q.LC())), max(bound(p), bound(q))

    def limit(self, form):
        result = self.sympy.limit(form, self.n, self.sympy.oo)
        if result.is_Rational:
            return Fraction(int(result.p), int(result.q))
        return None


class NativeBackend:
    version = "Lean native residue_checker"

    def __init__(self):
        self.process = subprocess.Popen(
            [str(CHECKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        # Readiness handshake is charged to startup, not steady-state cases.
        self.exchange({"support": [True], "op": "eq", "left": constant(0), "right": constant(0)})

    def exchange(self, request):
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 60)
        if not ready:
            raise TimeoutError("native checker response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native checker exited before returning a response")
        result = json.loads(line)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def compare(self, support, request):
        return self.exchange(dict(request, support=support))

    def standard_part(self, support, expression):
        result = self.exchange({"support": support, "op": "standardPart", "left": expression})
        if result["support"] != support:
            raise AssertionError("standard-part extraction mutated support")
        value = result["value"]
        return None if value is None else Fraction(int(value[0]), int(value[1]))

    def close(self):
        self.process.stdin.close()
        status = self.process.wait(timeout=10)
        error = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        if status:
            raise RuntimeError(f"native checker failed: {error}")


def run_case(backend, case, *, validate_cutoffs=False):
    support, accepted = [True], []
    for request in case["queries"]:
        response = backend.compare(support, request)
        accepted.append(response["accepted"])
        if validate_cutoffs:
            period = len(response["predicate"])
            cutoff = int(response["cutoff"])
            if cutoff < 1:
                raise AssertionError("nonpositive cutoff")
            for residue in range(period):
                n = cutoff + (residue - cutoff) % period
                for index in (n, n + 3 * period):
                    a, b = evaluate(request["left"], index), evaluate(request["right"], index)
                    truth = a < b if request["op"] == "lt" else a == b
                    if response["predicate"][residue] is not truth:
                        raise AssertionError((case["name"], request, response, index))
        if response["accepted"]:
            support = response["support"]
    value = backend.standard_part(support, case["expression"])
    return {"accepted": accepted, "support": support,
            "standard_part": None if value is None else [str(value.numerator), str(value.denominator)]}


def validate_expected(case, result):
    expected_period, expected_active, expected_value = case["expected"]
    support = result["support"]
    value = result["standard_part"]
    value = None if value is None else Fraction(int(value[0]), int(value[1]))
    if len(support) != expected_period or (expected_active is not None and sum(support) != expected_active) or value != expected_value:
        raise AssertionError((case["name"], support, value, case["expected"]))


def case_metrics(case):
    expressions = [case["expression"]] + [query[key] for query in case["queries"] for key in ("left", "right")]
    max_terms, max_bits = 0, 0
    for expression in expressions:
        for residue in range(expression_period(expression)):
            polynomial = sparse_form(expression, residue)
            max_terms = max(max_terms, len(polynomial))
            max_bits = max(max_bits, max((max(abs(c.numerator).bit_length(), c.denominator.bit_length()) for c in polynomial.values()), default=0))
    return {"expression_nodes": sum(expression_nodes(e) for e in expressions),
            "max_normalized_terms": max_terms, "max_normalized_coefficient_bits": max_bits}


def peak_rss_bytes(children=False):
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    if sys.platform.startswith("linux"):
        return int(value * 1024)
    return None


def worker(backend_name, case_name, repeats):
    case = next(case for case in workloads() if case["name"] == case_name)
    expected = run_case(ExactBackend(), case, validate_cutoffs=True)
    validate_expected(case, expected)
    metrics = case_metrics(case)
    start = time.perf_counter_ns()
    backend = {"fraction": ExactBackend, "sympy": SympyBackend, "lean_native": NativeBackend}[backend_name]()
    startup_ns = time.perf_counter_ns() - start
    try:
        observed = run_case(backend, case, validate_cutoffs=True)
        if observed != expected:
            raise AssertionError((backend_name, case_name, observed, expected))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            observed = run_case(backend, case)
            samples.append(time.perf_counter_ns() - start)
            if observed != expected:
                raise AssertionError("timed repetition disagreed with exact reference")
    finally:
        backend.close()
    own_rss, child_rss = peak_rss_bytes(), peak_rss_bytes(children=True)
    return {"backend": backend_name, "case": case_name, "backend_version": backend.version,
            "samples_ns": samples, "median_ns": statistics.median(samples), "min_ns": min(samples), "max_ns": max(samples),
            "startup_ns": startup_ns, "runtime_peak_rss_bytes": child_rss if backend_name == "lean_native" else own_rss,
            "transport_peak_rss_bytes": own_rss if backend_name == "lean_native" else None,
            "state_period": len(expected["support"]), "active_residues": sum(expected["support"]),
            "result": expected, **metrics}


def emit_results(report, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["case", "backend", "median_us", "min_us", "max_us", "startup_ms", "runtime_peak_mib", "transport_peak_mib", "standard_part",
               "state_period", "active_residues", "expression_nodes", "max_normalized_terms", "max_normalized_coefficient_bits"]
    rows = []
    for record in report["results"]:
        row = {key: record[key] for key in columns if key in record}
        row.update({"median_us": record["median_ns"] / 1000, "min_us": record["min_ns"] / 1000,
                    "max_us": record["max_ns"] / 1000, "startup_ms": record["startup_ns"] / 10**6,
                    "runtime_peak_mib": None if record["runtime_peak_rss_bytes"] is None else record["runtime_peak_rss_bytes"] / 2**20,
                    "transport_peak_mib": None if record["transport_peak_rss_bytes"] is None else record["transport_peak_rss_bytes"] / 2**20})
        value = record["result"]["standard_part"]
        row["standard_part"] = "unknown" if value is None else str(Fraction(int(value[0]), int(value[1])))
        rows.append(row)
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    environment = report["environment"]
    lines = ["# Exact finite-period Laurent benchmark", "",
             f"Measured {report['generated_utc']} on {environment['platform']} ({environment['machine']}) with Python {environment['python']} and {environment['lean_toolchain']}. Repetitions per backend/workload: {report['repeats']}.", "",
             f"Reproduce: `lake build residue_checker && uv run --extra benchmark python scripts/benchmark_residues.py --repeat {report['repeats']} --output benchmarks/residues`.", "",
             "Each row passed exact decision, final-support, and rational-standard-part agreement before timing. Each backend's reported comparison cutoffs were also checked by direct Fraction evaluation on every residue at two later indices. Those finite checks supplement the Lean theorems. They are not a proof of the independent baselines.", "",
             "The Lean process remains alive across an untimed validation/warmup and all repetitions. Its steady timings include Python JSON serialization, pipe transport, native parsing/computation, and JSON decoding. Fraction and SymPy timings are direct Python calls including normalization, period enumeration, cutoff calculation, state refinement, and limit extraction. All cases begin with a fresh logical universe support. Each backend uses its own exact sufficient cutoff. Cutoffs need not be numerically identical.", "",
             "The public LeanResidueSystem wrapper currently launches a checker process for every request. This benchmark's persistent session measures a different transport configuration. Public API end-to-end latency additionally includes repeated process launches. Expected mathematical results and cross-backend agreement are checked outside each timing sample. Native response validation and common result packaging remain inside.", "",
             "Startup is measured separately for native process launch through a readiness request, SymPy import and symbol initialization, or Fraction-backend construction. SymPy's process-local caches remain enabled after warmup. The measured implementations provide different assurance and have different transport costs, so these numbers are not a claim about the performance cost of formal verification.", "",
             "Peak RSS is the operating-system high-water mark for a fresh backend/workload worker, including interpreter/imports, reference validation, warmup, and repetitions. For Lean, runtime RSS instead reports the single native child and transport RSS separately reports its Python worker. macOS reports ru_maxrss in bytes. Linux reports it in KiB. Unsupported platforms emit null. These are process-lifetime peaks, not per-operation allocations or directly interchangeable memory scopes.", "",
             "Expression nodes count serialized AST occurrences across the workload's query operands and extracted expression. Coefficient metrics describe the largest final normalized branch among those expressions, not intermediate allocation peaks. Dense periods are not minimized.", "",
             "A standard-part result of `unknown` means the active residues include unequal finite limits or at least one divergent branch. Extraction leaves the support unchanged. Unknown is not treated as a numeric answer or a benchmark failure.", "",
             "| Workload | Backend | Median µs | Min–max µs | Startup ms | Runtime peak MiB | Transport peak MiB | Period / active | Standard part |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        rss = "n/a" if row["runtime_peak_mib"] is None else f"{row['runtime_peak_mib']:.2f}"
        transport = "—" if row["transport_peak_mib"] is None else f"{row['transport_peak_mib']:.2f}"
        lines.append(f"| {row['case']} | {row['backend']} | {row['median_us']:.2f} | {row['min_us']:.2f}–{row['max_us']:.2f} | {row['startup_ms']:.2f} | {rss} | {transport} | {row['state_period']} / {row['active_residues']} | {row['standard_part']} |")
    lines += ["", f"Native executable SHA-256: `{environment['checker_sha256']}`. Full timing samples, outputs, dependency versions, and structural metrics are in the adjacent JSON/CSV files.", ""]
    if report["skipped_backends"]:
        lines += [f"Unmeasured backends: {', '.join(report['skipped_backends'])}.", ""]
    prefix.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/residues")
    parser.add_argument("--worker", choices=("fraction", "sympy", "lean_native"), help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.worker:
        print(json.dumps(worker(args.worker, args.case, args.repeat)))
        return
    if not CHECKER.is_file():
        parser.error("build the native checker first: lake build residue_checker")
    backends, skipped = ["fraction", "lean_native"], []
    if importlib.util.find_spec("sympy") is not None:
        backends.append("sympy")
    else:
        skipped.append("sympy (install the pinned benchmark extra)")
    results = []
    for case in workloads():
        for backend in backends:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", backend, "--case", case["name"], "--repeat", str(args.repeat)],
                capture_output=True, text=True, check=True, timeout=300,
            )
            results.append(json.loads(completed.stdout))
            print(f"validated and measured {backend}: {case['name']}", flush=True)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "repeats": args.repeat,
              "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
                              "lean_toolchain": (ROOT / "lean-toolchain").read_text().strip(),
                              "checker_sha256": hashlib.sha256(CHECKER.read_bytes()).hexdigest()},
              "skipped_backends": skipped, "results": results}
    emit_results(report, args.output)
    print(f"Saved {args.output.with_suffix('.json')}, .csv, and .md")


if __name__ == "__main__":
    main()
