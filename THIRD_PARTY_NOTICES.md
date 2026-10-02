# Third-party code and data

BINDRA is MIT licensed (see `LICENSE`). It contains and depends on third-party
work under other terms, listed here.

## Vendored source

### `bindra/encoders/kpgt_descriptors/`

Four files copied from **KPGT** (`lihan97/KPGT`), which is licensed under the
**Apache License 2.0**; the licence text travels with them in
`bindra/encoders/kpgt_descriptors/LICENSE`. KPGT in turn derives this code from
`bp-kelley/descriptastorus` (BSD 3-clause).

Files: `DescriptorGenerator.py`, `rdDescriptors.py`, `rdNormalizedDescriptors.py`,
`dists.py`.

Copied rather than depended on deliberately. These produce the 200 normalized
RDKit descriptors KPGT's second knowledge node consumes, and the normalization
is a CDF against distribution parameters frozen in `dists.py`. The published
KPGT weights were fitted against *this* implementation, so a different version
would feed the pretrained encoder inputs it has never seen — silently, with no
error. Pinning the exact code is the only way to keep that honest.

Three mechanical changes were made, none touching a computed value; they are
documented in `bindra/encoders/kpgt_descriptors/__init__.py` and verified
against the original to 1e-15 by `tests/test_ligand_kpgt.py`.

## Not vendored, but required to reproduce the KPGT verification

### KPGT reference implementation

`tools/kpgt_oracle/make_reference.py` runs the original KPGT code to produce the
tensors that `tests/test_kpgt_port.py` checks the pure-PyTorch port against. That
repository is **not** included here. Clone it separately and point `KPGT_REF` at
it:

```bash
git clone https://github.com/lihan97/KPGT ../KPGT
```

### KPGT pretrained weights

`base.pth`, distributed by the KPGT authors via figshare. Not redistributed
here. See `docs/STATUS.md`.

### ESM-C 600M

`biohub/ESMC-600M`, pinned by revision and weights checksum in
`bindra/encoders/protein_esmc.py`. Downloaded at first use under its own terms;
not redistributed here.

## Data

BindingDB is distributed by its maintainers under their own terms and is **not**
redistributed in this repository. `data/snapshots/**/*.json` records which
release was used, with checksums, so the snapshot can be re-acquired and
verified. See `docs/STATUS.md` for the release and row counts.
