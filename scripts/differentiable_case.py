"""Compile and kernel-check a vector function with elementary operations."""

from hyperreals import DifferentiableProgram, variables


def main() -> None:
    x, y = variables(2)
    program = DifferentiableProgram(
        2,
        (
            x.sin() * y.exp(),
            (1 + x * x + y * y).log(),
            (1 + x * x).sqrt() / (1 + y * y),
        ),
    )
    compiled = program.jvp((1, 2))
    verified = compiled.verify(point=(1, 0), timeout=180)
    print(
        "Kernel checked the compiler output, derivative, quotient theorem, and domain at (1, 0)."
    )
    print(f"Proof source SHA-256: {verified.source_sha256}")
    print("Symbolic derivative components:")
    for expression in compiled.derivative.outputs:
        print(f"  {expression}")
    print(
        f"Approximate values (not certificates): {compiled.derivative.approximate((1, 0))}"
    )


if __name__ == "__main__":
    main()
