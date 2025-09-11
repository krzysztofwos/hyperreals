# Hyperreals

Hyperreal arithmetic with SAT-backed lazy ultrafilters for non-standard analysis and automatic differentiation.

## Installation

```bash
uv sync
```

## Usage

### As a library

```python
from hyperreal import HyperrealSystem

# Initialize the system
sys = HyperrealSystem()
eps = sys.infinitesimal()
two = sys.constant(2.0)

# Define a function: f(x) = x^3 - 3x
def f(x):
    three = sys.constant(3.0)
    return x * x * x - three * x

# First derivative using f'(x) ≈ [f(x+ε) - f(x)] / ε
f_x = f(two)
f_x_plus_eps = f(two + eps)
first_derivative = (f_x_plus_eps - f_x) / eps
print(f"f'(2) =", first_derivative.standard_part())  # 9.0

# Second derivative using f''(x) ≈ [f(x+ε) - 2f(x) + f(x-ε)] / ε²
f_x_minus_eps = f(two - eps)
second_derivative = (f_x_plus_eps - two * f_x + f_x_minus_eps) / (eps * eps)
print(f"f''(2) =", second_derivative.standard_part())  # 12.0

# Third derivative using f'''(x) ≈ [f(x+2ε) - 2f(x+ε) + 2f(x-ε) - f(x-2ε)] / (2ε³)
two_eps = two * eps
f_x_plus_2eps = f(two + two_eps)
f_x_minus_2eps = f(two - two_eps)
numerator = f_x_plus_2eps - two * f_x_plus_eps + two * f_x_minus_eps - f_x_minus_2eps
third_derivative = numerator / (two * eps * eps * eps)
print(f"f'''(2) =", third_derivative.standard_part())  # 6.0
```

### Run the demo

```bash
# Basic demo
uv run python scripts/demo.py

# Save ultrafilter state to file
uv run python scripts/demo.py --save-ultrafilter ultrafilter_state.json

# Show all clauses instead of just 10
uv run python scripts/demo.py --clause-limit 0

# Show help for all options
uv run python scripts/demo.py --help
```

### Taylor series via ε (all derivatives at once)

```bash
# Compute derivatives of several functions using f(x+ε)
uv run python scripts/taylor.py

# Specific function and point, up to order 5
uv run python scripts/taylor.py --func exp --x 0.7 --order 5

# Multiple functions
uv run python scripts/taylor.py --func sin --func cos --x 0.0 --order 5
```

This reads coefficients of powers of ε (ε=1/n) from the internal series to
recover f(x), f'(x), f''(x), … in one pass.

### Run evaluation suite

```bash
# Run evaluation and generate metrics
uv run python scripts/eval.py

# Generate metrics with verbose output
uv run python scripts/eval.py --verbose

# Save metrics to custom location
uv run python scripts/eval.py --out results.csv

# Also save ultrafilter state after evaluation
uv run python scripts/eval.py --save-ultrafilter final_state.json
```

The evaluation suite generates:

- `metrics.csv` - Raw metrics data
- `metrics.tex` - LaTeX table for papers
- `metrics.md` - Markdown table for documentation

## Project Structure

```text
 scripts      # Standalone scripts
 ├── demo.py  # Demonstration script
 └── eval.py  # Evaluation suite with metrics
 src
 └── hyperreal           # Main library package
     ├── algebra.py      # Set algebra
     ├── dual.py         # Dual numbers for AD
     ├── hyperreal.py    # Hyperreal and HyperrealSystem
     ├── sat.py          # SAT solver
     ├── sequence/       # Sequence DSL module
     ├── series.py       # Laurent series operations
     └── ultrafilter.py  # Partial ultrafilter
```

## Development

```bash
# Run tests
uv run pytest

# Lint code
uv run ruff check src/hyperreal

# Type check
uv run mypy src/hyperreal
```

## License

MIT
