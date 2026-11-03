# BINDRA manuscript and compact reference release

**BINDRA: pooled fusion and interaction attention for protein-ligand affinity prediction**

This directory contains the current Journal of Cheminformatics draft, its
supplement, selected vector figures, and compact result tables. The main text was
shortened while retaining the primary prediction comparisons, uncertainty,
structural controls, FAK1-P4N case, and complete supporting outcomes in the
supplement. The six main figures and five main tables remain in the article.

- [Main manuscript](manuscript.pdf) and [readable source](manuscript.md)
- [Supplement](additional_files/Additional_file_1_supplement.pdf) and [source](additional_files/Additional_file_1_supplement.md)
- [Reference code](../reference/README.md), including strict three-model pKd aggregation
- [Architecture](figures/Figure_1_architecture.svg) and [FAK1-P4N attention case](figures/Figure_6_FAK1_case.pdf)
- [Tabular summaries](tables/)

The reference code and synthetic tests run from a source checkout. Encoder
weights, trained affinity/generator checkpoints, full feature stores, raw licensed
data, and the large additional-file archives are separate assets. This compact
source release supports inspection of the implementation and reported summaries;
the full reproduction inventory and remaining deposits are described in the
manuscript's data-availability statement. Published external DTA values remain
literature comparisons. Historical test exposure and endpoint-specific supporting
results are retained in the paper.

The PDF figures retain the reviewed source presentation. The full authoring
package separately supplies editable Word/LaTeX documents and 600-dpi PNG/TIFF
exports; those bulk authoring files are omitted from this small Git release.
