# Sequence module

[Back to the main README](../../README.md) · [Documentation overview](../README.md)

## Inhaltsverzeichnis

- [Installation and external tools](#installation-and-external-tools)
- [Global protein alignment](#global-protein-alignment)
  - [Aligned sequences](#aligned-sequences)
  - [Score, identity, and similarity](#score-identity-and-similarity)
  - [Gap penalties](#gap-penalties)
- [FASTA files](#fasta-files)
  - [Write FASTA](#write-fasta)
  - [Read FASTA](#read-fasta)
- [Local BLAST searches](#local-blast-searches)
- [Compatibility imports](#compatibility-imports)

`biotools.sequence` provides small helpers for protein-sequence alignment,
simple FASTA files, and local protein BLAST databases.

## Installation and external tools

Sequence alignment and FASTA support are included in the base installation:

```bash
python -m pip install -e .
```

Local BLAST workflows additionally require the NCBI BLAST+ executables
`makeblastdb` and `blastp` on `PATH`.

## Global protein alignment

The alignment helpers use Biotite's standard protein substitution matrix and
return the first optimal global alignment.

### Aligned sequences

```python
from biotools.sequence import global_alignment_seqs

sequence_a = "MKTAYIAKQRQISFVKSHFSRQ"
sequence_b = "MKTAYIAKQRTISFVKSHFSRQ"

aligned_a, aligned_b = global_alignment_seqs(sequence_a, sequence_b)
print(aligned_a)
print(aligned_b)
```

The returned strings contain `-` characters for alignment gaps.

### Score, identity, and similarity

```python
from biotools.sequence import (
    global_alignment_identity,
    global_alignment_score,
    global_alignment_similarity,
)

score = global_alignment_score(sequence_a, sequence_b)
identity = global_alignment_identity(sequence_a, sequence_b)
similarity = global_alignment_similarity(sequence_a, sequence_b)

print(score)
print(f"Identity: {identity:.1f}%")
print(f"Similarity: {similarity:.1f}%")
```

Identity counts exact matches and is normalized by the ungapped length of the
first sequence. Similarity considers aligned, non-gap residue pairs similar
when their score in Biotite's standard protein substitution matrix is
positive.

### Gap penalties

All four alignment functions accept the same `gap_penalty` argument. The
default is the affine penalty `(-10, -1)` for gap opening and extension. A
single integer selects a linear penalty:

```python
aligned_a, aligned_b = global_alignment_seqs(
    sequence_a,
    sequence_b,
    gap_penalty=(-12, -2),
)
```

## FASTA files

The FASTA helpers use mappings whose keys are complete header strings without
the leading `>`.

### Write FASTA

```python
from biotools.sequence import dict_to_fasta

dict_to_fasta(
    {
        "protein_a": "MKTAYIAKQRQISFVKSHFSRQ",
        "protein_b description": "GILGFVFTLTVPSER",
    },
    "proteins.fasta",
)
```

Each sequence is written as one line in insertion order.

### Read FASTA

```python
from biotools.sequence import fasta_to_dict

sequences = fasta_to_dict("proteins.fasta")
print(sequences["protein_a"])
```

Wrapped input sequences are joined into a single string. Empty lines are
ignored.

## Local BLAST searches

`BlastSearch` creates a local protein database when its index is absent and
reuses it on later calls:

```python
from biotools.sequence import BlastSearch

database = BlastSearch(
    input_fasta="proteins.fasta",
    db_name="proteins",
    description="Example protein database",
)
hits = database.search(
    "query.fasta",
    evalue=1e-10,
    min_coverage=0.9,
)
print(hits)
```

`min_coverage` accepts either a fraction such as `0.9` or a percentage such
as `90`. Results are returned as a pandas `DataFrame`; the current parser is
intended for database identifiers in the PDB-style format expected by the
project and retains hits above 50% sequence identity.

## Compatibility imports

The preferred public imports come from `biotools.sequence`. The historical
`biotools.seqtools` module remains a compatibility facade for existing code.
