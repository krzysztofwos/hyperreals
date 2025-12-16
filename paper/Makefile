PAPER = paper
TEX = pdflatex
BIB = bibtex
TEXFLAGS = -interaction=nonstopmode

METRICS = metrics.csv metrics.tex metrics.md

# Default target
.PHONY: all
all: $(PAPER).pdf

# Build the PDF with BibTeX
$(PAPER).pdf: $(PAPER).tex references.bib metrics.tex
	@echo "Building PDF with BibTeX..."
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	$(BIB) $(PAPER)
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	@echo "PDF built successfully: $(PAPER).pdf"

# Quick build (single pass, for development)
.PHONY: quick
quick: $(PAPER).tex metrics.tex
	@echo "Quick build..."
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	@echo "Quick build complete: $(PAPER).pdf"

.PHONY: metrics
metrics: ../scripts/eval.py
	@echo "Generating metrics via evaluation suite..."
	python ../scripts/eval.py --out metrics.csv

# Lint LaTeX source files
# Suppressed warnings:
#   -n3: parenthesis enclosure (false positives for \code{})
#   -n13: intersentence spacing (stylistic)
#   -n36: space before parenthesis (false positives for \code{})
#   -n38: punctuation before quotes (standard LaTeX style)
#   -n44: booktabs (auto-generated metrics.tex uses \hline)
#   -n24: spacing before label (auto-generated metrics.tex)
.PHONY: lint
lint:
	@echo "Linting LaTeX source files..."
	@if command -v chktex >/dev/null; then \
		chktex -q -n3 -n13 -n24 -n36 -n38 -n44 $(PAPER).tex || true; \
	else \
		echo "chktex not found. Please install it to lint LaTeX files."; \
	fi

# Format LaTeX source files
.PHONY: fmt
fmt:
	@echo "Formatting LaTeX source files..."
	@if command -v latexindent >/dev/null; then \
		latexindent -y="defaultIndent: '  '" -w $(PAPER).tex; \
		echo "Formatted $(PAPER).tex"; \
	else \
		echo "latexindent not found. Please install it to format LaTeX files."; \
	fi

# Clean temporary files
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log
	rm -f $(PAPER).out $(PAPER).toc $(PAPER).lof $(PAPER).lot
	rm -f $(PAPER).nav $(PAPER).snm $(PAPER).vrb
	rm -f $(PAPER).pdf
	rm -f texput.log
	@echo "Cleanup complete."

# Build and view the PDF
.PHONY: view
view: $(PAPER).pdf
	@echo "Opening PDF..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PAPER).pdf; \
	elif command -v open > /dev/null; then \
		open $(PAPER).pdf; \
	else \
		echo "Please open $(PAPER).pdf manually"; \
	fi

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make         - Build the PDF (two passes for references)"
	@echo "  make quick   - Quick build (single pass)"
	@echo "  make metrics - Regenerate metrics (requires Python)"
	@echo "  make view    - Build and open the PDF"
	@echo "  make lint    - Lint the LaTeX source files (chktex)"
	@echo "  make fmt     - Format the LaTeX source files (latexindent)"
	@echo "  make clean   - Remove all temporary files and PDF"
	@echo "  make help    - Show this help message"
