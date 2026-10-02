"""Target constructs: validation, identity, and offline mutation detection.

The governing rule comes straight from the specification: *"Preserve experimental
constructs and mutations; do not automatically replace every construct with the
full canonical UniProt sequence."*

So a target's modelling identity here is the **sha256 of the exact construct
string**, not its UniProt accession. Two constructs of the same protein -- a
kinase domain and the full-length chain, or a wild type and its T790M mutant --
are different targets as far as the model is concerned, because they are
different molecules and were measured as such.

Length is a flag, never a rejection. The measured maximum construct length in
release 202610 is 34,350 residues, so a "proteins are under 5,000 residues"
assumption would silently discard real targets.

**What cannot be done offline.** Mutation calls here come from aligning the
constructs that share a UniProt accession against the longest of that group. That
genuinely finds point differences without any network access, but it cannot
establish which construct is wild type -- that needs the UniProt reference
sequence. Tokens scraped from ``Target Name`` (``"T790M"``) are recorded as
corroboration only and are never authoritative.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from bindra.config.profiles import ProteinPolicy

#: Standard residues plus the ambiguity and non-standard codes that appear in
#: BindingDB target chains. X is unknown; B/Z are ambiguity codes; U is
#: selenocysteine and O is pyrrolysine, both real residues.
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Mutation tokens in free-text target names, e.g. "T790M", "V600E".
_MUTATION_TOKEN = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,5})([ACDEFGHIKLMNPQRSTVWY])\b")


class ConstructStatus:
    OK = "OK"
    EMPTY = "EMPTY"
    INVALID_RESIDUES = "INVALID_RESIDUES"


class ConstructFlag:
    CONTAINS_AMBIGUOUS = "CONTAINS_AMBIGUOUS"
    CONTAINS_NONSTANDARD = "CONTAINS_NONSTANDARD"
    SHORT_CONSTRUCT = "SHORT_CONSTRUCT"
    LONG_CONSTRUCT = "LONG_CONSTRUCT"
    MULTIPLE_UNIPROT_IDS = "MULTIPLE_UNIPROT_IDS"
    NO_UNIPROT = "NO_UNIPROT"
    SUBSTRING_OF_ANOTHER = "SUBSTRING_OF_ANOTHER"


@dataclass(frozen=True)
class Mutation:
    position: int  # 1-based, in the reference construct's numbering
    wild_type: str
    mutant: str

    def __str__(self) -> str:
        return f"{self.wild_type}{self.position}{self.mutant}"


@dataclass
class TargetConstruct:
    """One unique target sequence as it was actually measured."""

    sequence_ref: int
    sequence_sha256: str
    sequence_length: int
    status: str
    construct_id: str | None = None
    uniprot_primary: str | None = None
    uniprot_source: str | None = None
    uniprot_all: str | None = None
    organism: str | None = None
    target_name: str | None = None
    flags: tuple[str, ...] = ()
    invalid_residues: str | None = None
    mutations: tuple[Mutation, ...] = ()
    mutation_source: str | None = None
    mutation_reference_ref: int | None = None
    name_mutation_tokens: tuple[str, ...] = ()
    n_constructs_for_uniprot: int = 1

    @property
    def ok(self) -> bool:
        return self.status == ConstructStatus.OK

    def to_row(self) -> dict[str, Any]:
        return {
            "sequence_ref": self.sequence_ref,
            "sequence_sha256": self.sequence_sha256,
            "sequence_length": self.sequence_length,
            "status": self.status,
            "construct_id": self.construct_id,
            "uniprot_primary": self.uniprot_primary,
            "uniprot_source": self.uniprot_source,
            "uniprot_all": self.uniprot_all,
            "organism": self.organism,
            "target_name": self.target_name,
            "flags": list(self.flags),
            "invalid_residues": self.invalid_residues,
            "mutations": [str(m) for m in self.mutations],
            "mutation_source": self.mutation_source,
            "mutation_reference_ref": self.mutation_reference_ref,
            "name_mutation_tokens": list(self.name_mutation_tokens),
            "n_constructs_for_uniprot": self.n_constructs_for_uniprot,
        }


def canonical_sequence(sequence: str) -> str:
    """Strip and upper-case a construct string.

    Case is not meaningful here -- BindingDB's formatting is simply inconsistent,
    and 33 of the 10,822 constructs in release 202610 are written entirely in
    lower case (none are mixed). Canonicalizing matters because the ESM-C
    tokenizer only recognizes upper-case residues: a lower-case sequence
    tokenizes to ``<unk>`` throughout, and ``<unk>`` is a *special* token, so the
    adapter's special-token stripping removes every position. Verified on
    release 202610: upper-casing merges no distinct sequences, so this changes
    only the spelling and the hash, never the set of targets.
    """
    return (sequence or "").strip().upper()


def sequence_sha256(sequence: str) -> str:
    """Hash of the canonical construct string."""
    return hashlib.sha256(canonical_sequence(sequence).encode("utf-8")).hexdigest()


def validate_sequence(
    sequence: str,
    policy: ProteinPolicy | None = None,
) -> tuple[str, tuple[str, ...], str | None]:
    """Return ``(status, flags, invalid_residues)`` for one construct string.

    Flags are informational and do not stop a construct being used; only
    ``status`` decides that. Length in particular is a flag: the real corpus
    contains a 34,350-residue construct.
    """
    policy = policy or ProteinPolicy()
    allowed = set(policy.allowed_alphabet)
    flags: list[str] = []

    text = (sequence or "").strip().upper()
    if not text:
        return ConstructStatus.EMPTY, (), None

    present = set(text)
    invalid = sorted(present - allowed)
    if invalid:
        return ConstructStatus.INVALID_RESIDUES, (), "".join(invalid)

    if "X" in present or present & {"B", "Z"}:
        flags.append(ConstructFlag.CONTAINS_AMBIGUOUS)
    if present & {"U", "O"}:
        flags.append(ConstructFlag.CONTAINS_NONSTANDARD)
    if len(text) < policy.flag_length_below:
        flags.append(ConstructFlag.SHORT_CONSTRUCT)
    if len(text) > policy.flag_length_above:
        flags.append(ConstructFlag.LONG_CONSTRUCT)

    return ConstructStatus.OK, tuple(flags), None


def mutation_tokens_from_name(name: str | None) -> tuple[str, ...]:
    """Mutation-looking tokens in a free-text target name.

    Corroboration only. ``Target Name`` is uncontrolled free text, and a token
    like ``"T790M"`` may describe the construct, a comparison, or nothing at all.
    """
    if not name:
        return ()
    return tuple(dict.fromkeys(m.group(0) for m in _MUTATION_TOKEN.finditer(name)))


def _align_for_mutations(
    reference: str,
    query: str,
    policy: ProteinPolicy,
) -> tuple[Mutation, ...] | None:
    """Point differences of ``query`` against ``reference``, or None.

    Returns None unless the two align gaplessly at high identity -- i.e. unless
    they really are the same construct differing at a few positions. Anything
    looser (a domain fragment, a splice variant) is not a point-mutation
    relationship and is deliberately not described as one.
    """
    if len(reference) != len(query):
        # Gapless by construction: differing lengths mean an indel or a
        # truncation, which is a different relationship.
        return None
    differences = [
        Mutation(position=index + 1, wild_type=r, mutant=q)
        for index, (r, q) in enumerate(zip(reference, query))
        if r != q
    ]
    if not differences:
        return ()
    identity = 1.0 - (len(differences) / len(reference))
    if identity < policy.mutation_min_identity:
        return None
    return tuple(differences)


def build_target_constructs(
    sequences: Iterable[Mapping[str, Any]],
    annotations: Mapping[int, Mapping[str, Any]] | None = None,
    policy: ProteinPolicy | None = None,
) -> tuple[list[TargetConstruct], dict[str, Any]]:
    """Validate and annotate every unique construct.

    ``sequences`` yields ``{"sequence_ref", "sequence", "sequence_sha256",
    "sequence_length"}``; ``annotations`` maps a ref to the per-row target fields
    observed for it (UniProt, organism, target name).
    """
    policy = policy or ProteinPolicy()
    annotations = annotations or {}

    constructs: list[TargetConstruct] = []
    by_uniprot: dict[str, list[int]] = defaultdict(list)
    sequence_by_ref: dict[int, str] = {}

    report: dict[str, Any] = {
        "n_input": 0,
        "status_counts": defaultdict(int),
        "flag_counts": defaultdict(int),
        "uniprot_sources": defaultdict(int),
        "length": {},
    }
    lengths: list[int] = []

    for record in sequences:
        report["n_input"] += 1
        ref = int(record["sequence_ref"])
        sequence = canonical_sequence(str(record["sequence"]))
        sequence_by_ref[ref] = sequence
        lengths.append(len(sequence))

        status, flags, invalid = validate_sequence(sequence, policy)
        meta = annotations.get(ref, {})
        uniprot_all = meta.get("uniprot_all") or None
        uniprot_primary = meta.get("uniprot_primary") or None
        uniprot_source = meta.get("uniprot_source") or "none"

        flag_list = list(flags)
        if uniprot_all and "," in str(uniprot_all):
            flag_list.append(ConstructFlag.MULTIPLE_UNIPROT_IDS)
        if not uniprot_primary:
            flag_list.append(ConstructFlag.NO_UNIPROT)

        construct = TargetConstruct(
            sequence_ref=ref,
            # Always recomputed from the canonical form: a hash carried over
            # from a non-canonical spelling would not identify this construct.
            sequence_sha256=sequence_sha256(sequence),
            sequence_length=len(sequence),
            status=status,
            construct_id=None,
            uniprot_primary=uniprot_primary,
            uniprot_source=uniprot_source,
            uniprot_all=str(uniprot_all) if uniprot_all else None,
            organism=meta.get("organism") or None,
            target_name=meta.get("target_name") or None,
            flags=tuple(flag_list),
            invalid_residues=invalid,
            name_mutation_tokens=mutation_tokens_from_name(meta.get("target_name")),
        )
        # construct_id names the protein where known but is keyed on the
        # construct hash, so two constructs of one UniProt never collide.
        construct.construct_id = (
            f"{uniprot_primary or 'NOUP'}:{construct.sequence_sha256[:8]}"
        )
        constructs.append(construct)

        report["status_counts"][status] += 1
        for flag in flag_list:
            report["flag_counts"][flag] += 1
        report["uniprot_sources"][uniprot_source] += 1
        if uniprot_primary:
            by_uniprot[uniprot_primary].append(ref)

    index = {c.sequence_ref: c for c in constructs}

    # -- mutation detection, within a UniProt group only -------------------
    n_mutant = 0
    n_groups_multi = 0
    for uniprot, refs in by_uniprot.items():
        for ref in refs:
            index[ref].n_constructs_for_uniprot = len(refs)
        if len(refs) < 2:
            continue
        n_groups_multi += 1
        # The longest construct is the reference. This is a labelling choice, not
        # a claim: it is NOT necessarily the wild type, and nothing offline can
        # establish which one is.
        reference_ref = max(refs, key=lambda r: len(sequence_by_ref[r]))
        reference = sequence_by_ref[reference_ref]
        for ref in refs:
            if ref == reference_ref:
                continue
            mutations = _align_for_mutations(reference, sequence_by_ref[ref], policy)
            if mutations:
                index[ref].mutations = mutations
                index[ref].mutation_source = "alignment_vs_longest_construct"
                index[ref].mutation_reference_ref = reference_ref
                n_mutant += 1

    # -- substring / fragment detection -----------------------------------
    #
    # Domain constructs are frequently literal substrings of the full-length
    # chain. Flagged here because the specification asks for fragments and
    # constructs to be handled explicitly rather than silently treated as
    # unrelated proteins; the homology stage turns these into guaranteed edges.
    n_substring = 0
    for uniprot, refs in by_uniprot.items():
        if len(refs) < 2:
            continue
        ordered = sorted(refs, key=lambda r: len(sequence_by_ref[r]), reverse=True)
        for position, ref in enumerate(ordered[1:], start=1):
            short = sequence_by_ref[ref]
            if any(short and short in sequence_by_ref[longer] for longer in ordered[:position]):
                construct = index[ref]
                if ConstructFlag.SUBSTRING_OF_ANOTHER not in construct.flags:
                    construct.flags = (*construct.flags, ConstructFlag.SUBSTRING_OF_ANOTHER)
                    report["flag_counts"][ConstructFlag.SUBSTRING_OF_ANOTHER] += 1
                    n_substring += 1

    if lengths:
        lengths.sort()
        report["length"] = {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "mean": round(sum(lengths) / len(lengths), 1),
            "p90": lengths[int(0.90 * len(lengths))],
            "max": lengths[-1],
            "total_residues": sum(lengths),
            "over_1024": sum(1 for x in lengths if x > 1024),
            "over_2046": sum(1 for x in lengths if x > 2046),
        }
    report["status_counts"] = dict(report["status_counts"])
    report["flag_counts"] = dict(report["flag_counts"])
    report["uniprot_sources"] = dict(report["uniprot_sources"])
    report["n_unique_uniprot"] = len(by_uniprot)
    report["n_uniprot_with_multiple_constructs"] = n_groups_multi
    report["n_constructs_with_mutations"] = n_mutant
    report["n_substring_constructs"] = n_substring
    report["caveat"] = (
        "Mutations are point differences against the LONGEST construct sharing a "
        "UniProt accession. That reference is not established to be wild type; "
        "no offline source can establish that. Target Name tokens are "
        "corroboration only."
    )
    return constructs, report


def build_constructs_from_processed(
    processed_dir: Path | str,
    *,
    policy: ProteinPolicy | None = None,
) -> dict[str, Any]:
    """Build ``target_constructs`` and ``protein_report`` from ingest output.

    Annotations come from the first observation seen for each sequence, which is
    deterministic because the ingest shards are written in source order.
    """
    import glob

    import pandas as pd

    from bindra.data.tables import write_json, write_table

    processed_dir = Path(processed_dir)
    sequences = pd.read_parquet(processed_dir / "sequences_raw.parquet")
    # Canonicalize here too, so an older sequences_raw written before
    # canonicalization still yields canonical constructs.
    sequences["sequence"] = [canonical_sequence(s) for s in sequences["sequence"]]

    columns = [
        "sequence_ref",
        "uniprot_primary",
        "uniprot_all",
        "uniprot_source",
        "organism",
        "target_name",
    ]
    parts = [
        pd.read_parquet(path, columns=columns)
        for path in sorted(glob.glob(str(processed_dir / "raw_observations" / "*.parquet")))
    ]
    observations = pd.concat(parts, ignore_index=True)
    annotations = (
        observations[observations["sequence_ref"] >= 0]
        .drop_duplicates("sequence_ref")
        .set_index("sequence_ref")
        .to_dict("index")
    )

    constructs, report = build_target_constructs(
        sequences.to_dict("records"), annotations, policy
    )
    report["mutation_corroboration"] = corroborate_mutations(constructs)
    report["n_annotated_sequences"] = len(annotations)

    reference = write_table(
        pd.DataFrame([construct.to_row() for construct in constructs]),
        processed_dir / "target_constructs",
    )
    write_json(report, processed_dir / "protein_report.json")
    report["artifact"] = reference.to_dict()
    return report


def corroborate_mutations(
    constructs: Sequence[TargetConstruct],
    *,
    max_offset: int = 5,
) -> dict[str, Any]:
    """Check alignment-derived mutation calls against target-name tokens.

    These are two independent signals -- one from comparing construct strings,
    one from uncontrolled free text -- so their agreement rate is a real
    validation of the offline detection rather than a self-consistency check.

    Measured on release 202610: of 302 constructs carrying both signals, 258
    agree exactly and 17 more agree on the substitution at a constant positional
    offset (15 of those at exactly -1), leaving 27 unexplained -- **91.1%
    agreement on the substitution itself**.

    The offsets are the expected consequence of numbering a construct from 1
    while the literature numbers the canonical protein, which commonly differs by
    the initiator methionine. It is also the concrete form of the limitation
    this module documents: without the UniProt reference sequence we can detect
    that two constructs differ at a position, but not what that position is
    called in the canonical numbering.
    """
    parsed = re.compile(r"^([A-Z])(\d+)([A-Z])$")

    def split(token: str) -> tuple[str, int, str] | None:
        match = parsed.match(token)
        return (match.group(1), int(match.group(2)), match.group(3)) if match else None

    exact = offset_match = unexplained = no_token = 0
    offsets: dict[int, int] = {}
    for construct in constructs:
        if not construct.mutations:
            continue
        tokens = {t for t in (split(x) for x in construct.name_mutation_tokens) if t}
        if not tokens:
            no_token += 1
            continue
        calls = {c for c in (split(str(m)) for m in construct.mutations) if c}
        if calls & tokens:
            exact += 1
            offsets[0] = offsets.get(0, 0) + 1
            continue
        for delta in range(-max_offset, max_offset + 1):
            if delta == 0:
                continue
            if {(w, p + delta, m) for w, p, m in calls} & tokens:
                offset_match += 1
                offsets[delta] = offsets.get(delta, 0) + 1
                break
        else:
            unexplained += 1

    comparable = exact + offset_match + unexplained
    return {
        "n_with_mutations": sum(1 for c in constructs if c.mutations),
        "n_without_name_token": no_token,
        "n_exact_agreement": exact,
        "n_agreement_at_constant_offset": offset_match,
        "n_unexplained": unexplained,
        "n_comparable": comparable,
        "substitution_agreement_rate": (
            round((exact + offset_match) / comparable, 4) if comparable else None
        ),
        "offset_histogram": {str(k): v for k, v in sorted(offsets.items())},
        "interpretation": (
            "Alignment calls and free-text target-name tokens are independent "
            "signals; their agreement validates the offline detection. Non-zero "
            "offsets reflect construct-relative numbering versus canonical protein "
            "numbering, which cannot be reconciled without the UniProt reference "
            "sequence."
        ),
    }


__all__ = [
    "STANDARD_RESIDUES",
    "canonical_sequence",
    "corroborate_mutations",
    "ConstructFlag",
    "ConstructStatus",
    "Mutation",
    "TargetConstruct",
    "build_target_constructs",
    "mutation_tokens_from_name",
    "sequence_sha256",
    "validate_sequence",
]
