# Paper Compilation Instructions

## Files Generated

- `main.tex`: Complete LaTeX source for the paper
- `references.bib`: BibTeX bibliography with 25 references
- `figures/`: Directory containing all 16 required figures (PDF format)

## Compilation Steps

To compile the paper to PDF, run the following commands in the `paper/` directory:

```bash
cd C:/Users/ugure/ccode/categorical-tqft-emergence/hopf-layers/paper

# First pass - process main content
pdflatex main.tex

# Process bibliography
bibtex main

# Second pass - resolve citations
pdflatex main.tex

# Third pass - resolve cross-references
pdflatex main.tex
```

This will generate `main.pdf` in the same directory.

## Alternative: Single Command

If you have `latexmk` installed:

```bash
latexmk -pdf main.tex
```

## Cleanup

To remove auxiliary files:

```bash
rm -f *.aux *.log *.out *.bbl *.blg *.toc
```

Or with latexmk:

```bash
latexmk -c
```

## Requirements

- A standard LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (all standard and arXiv-compatible):
  - amsmath, amssymb
  - graphicx
  - booktabs
  - hyperref
  - natbib
  - xcolor
  - microtype
  - algorithm2e
  - geometry

All packages are included in standard LaTeX distributions.

## Paper Structure

The paper follows standard academic format:
- **Document class**: `article` with 10pt, two-column layout
- **Length**: ~8-10 pages (target for TMLR/NeurIPS workshop)
- **Style**: Scientific, concise, well-structured
- **Figures**: 16 figures total, all referenced in text
- **Tables**: 3 tables with numerical results
- **References**: 25 citations (topology, lattice gauge theory, geometric DL, ML)

## Compatibility

The paper is designed to be arXiv-compatible:
- No custom packages or style files
- All figures in PDF format (vector graphics)
- Standard natbib citation style
- Portable paths (relative to paper directory)

## Verification

After compilation, verify:
1. All figures appear correctly (16 figures total)
2. All citations resolve (no "?" marks)
3. All cross-references work (sections, equations, figures, tables)
4. Page count is 8-10 pages
5. No overfull/underfull hbox warnings (minor ones acceptable)

## Content Summary

**Title**: Differentiable Hopf Fibrations for Interpretable Geometric Feature Extraction

**Author**: Ugur Emre

**Key Contributions**:
1. First PyTorch implementation of all 3 Hopf fibrations
2. Differentiable geometric decomposition with gradient stability
3. Mutual information analysis showing clean disentanglement
4. Validation on 3 lattice gauge theory tasks

**Main Results**:
- 100% phase classification accuracy
- R²=0.93 topological charge regression
- NMI=1.0 for transition-topology association
- 23× computational overhead (acceptable for CNN bottleneck)

## Target Venues

The paper is formatted for:
- TMLR (Transactions on Machine Learning Research)
- NeurIPS workshop (geometric DL, physics + ML)
- ICLR (geometric/topological ML track)
- ICML (representation learning)

No venue-specific style file is applied; use standard article format for initial submission, then adapt to venue requirements.
