"""Utilities for sequence alignment, FASTA I/O, and BLAST-based search."""

import logging

import biotite.sequence as sequence
import biotite.sequence.align as align

import pandas as pd
import subprocess
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

def _global_alignment_obj(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Return the optimal global protein alignment for two sequences.

    Args:
        seq_a: First amino acid sequence.
        seq_b: Second amino acid sequence.
        gap_penalty: Affine gap penalty passed to ``biotite.align_optimal()``.

    Returns:
        The first optimal alignment object returned by Biotite.
    """
    # Proteinsequenzen erstellen
    s1 = sequence.ProteinSequence(seq_a)
    s2 = sequence.ProteinSequence(seq_b)

    # Substitutionsmatrix, z.B. BLOSUM62
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    # globales Alignment
    alignments = align.align_optimal(
        s1, s2,
        matrix,
        gap_penalty=gap_penalty,
        local=False
    )[0]
    return alignments

def global_alignment_score(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Compute the score of the optimal global protein alignment.

    Args:
        seq_a: First amino acid sequence.
        seq_b: Second amino acid sequence.
        gap_penalty: Affine gap penalty passed to the aligner.

    Returns:
        Alignment score of the best global alignment.
    """
    alignments = _global_alignment_obj(seq_a, seq_b, gap_penalty)
    return alignments.score

def global_alignment_seqs(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Return the gapped sequences from the optimal global alignment.

    Args:
        seq_a: First amino acid sequence.
        seq_b: Second amino acid sequence.
        gap_penalty: Affine gap penalty passed to the aligner.

    Returns:
        A tuple containing both aligned sequences with gap characters.
    """
    alignments = _global_alignment_obj(seq_a, seq_b, gap_penalty)
    gaped_a, gaped_b = alignments.get_gapped_sequences()
    return gaped_a, gaped_b

def global_alignment_identity(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Calculate percent identity for the optimal global alignment.

    Args:
        seq_a: First amino acid sequence.
        seq_b: Second amino acid sequence.
        gap_penalty: Affine gap penalty passed to the aligner.

    Returns:
        Sequence identity in percent, normalized by ungapped ``seq_a`` length.
    """
    gapped_a, gapped_b = global_alignment_seqs(seq_a, seq_b, gap_penalty)
    matches = sum(1 for a, b in zip(gapped_a, gapped_b) if a == b)
    alignment_length = len(gapped_a.replace('-', ''))
    sequence_identity = (matches / alignment_length) * 100
    return sequence_identity
    
def global_alignment_similarity(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Calculate percent similarity for the optimal global alignment.

    Similarity is counted for aligned residue pairs with a positive score in
    the default Biotite protein substitution matrix.

    Args:
        seq_a: First amino acid sequence.
        seq_b: Second amino acid sequence.
        gap_penalty: Affine gap penalty passed to the aligner.

    Returns:
        Sequence similarity in percent, normalized by aligned non-gap residue
        pairs. Returns ``0.0`` if the alignment contains no comparable pairs.
    """
    gapped_a, gapped_b = global_alignment_seqs(seq_a, seq_b, gap_penalty)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    comparable_pairs = [
        (a, b) for a, b in zip(gapped_a, gapped_b)
        if a != '-' and b != '-'
    ]
    if not comparable_pairs:
        return 0.0

    similar_positions = sum(1 for a, b in comparable_pairs if matrix[a, b] > 0)
    similarity = (similar_positions / len(comparable_pairs)) * 100
    return similarity

def dict_to_fasta(seq_dict, save_file):
    """Write a mapping of identifiers to sequences as a FASTA file.

    Args:
        seq_dict: Mapping from FASTA header to amino acid or nucleotide sequence.
        save_file: Output path for the FASTA file.
    """
    with open(save_file, 'w') as fp:
        for name, seq in seq_dict.items():
            fp.write(f">{name}\n")
            fp.write(f"{seq}\n")
    logger.info("Wrote %d sequences to %s", len(seq_dict), save_file)

def fasta_to_dict(load_file):
    """Parse a FASTA file into a dictionary keyed by record identifier.

    Args:
        load_file: Path to the FASTA file to read.

    Returns:
        A dictionary mapping FASTA headers to concatenated sequence strings.
    """
    with open(load_file, 'r') as fp:
        fasta_lines = fp.readlines()

    seq_dict = {}
    last_name = None
    current_seq = ""
    for line in fasta_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if last_name is not None:
                seq_dict[last_name] = current_seq
            last_name = line[1:]
            current_seq = ""
        else:
            current_seq += line

    if last_name is not None:
        seq_dict[last_name] = current_seq

    logger.debug("Loaded %d sequences from %s", len(seq_dict), load_file)
    return seq_dict

class BlastSearch:
    """Create and query a local BLAST protein database from a FASTA file."""

    def __init__(self,
            input_fasta: str,
            db_name: str,
            description: str = None
        ):
        """Initialize a BLAST database if the index files are missing.

        Args:
            input_fasta: FASTA file used as BLAST database input.
            db_name: Basename for the generated BLAST index files.
            description: Optional database title shown by BLAST tools.
        """
        if description is None:
            description = db_name

        
        self.index_file = Path(input_fasta).parent / db_name

        final_index = Path(str(self.index_file) + ".pdb")
        if not final_index.exists():
            logger.info("Creating BLAST protein database at %s", self.index_file)
            subprocess.run(
                [
                    'makeblastdb',
                    '-in', input_fasta,
                    '-dbtype', 'prot',
                    '-parse_seqids',
                    '-title', description,
                    '-out', str(self.index_file)
                ],
                check=True,
            )
        else:
            logger.debug("Using existing BLAST database at %s", self.index_file)

    def search(self,
               query_fasta,
               evalue=1e-10,
               min_coverage=0.9,
               max_targets=1000
            ) -> pd.DataFrame:
        """Search the local BLAST database with protein query sequences.

        Args:
            query_fasta: FASTA file containing one or more query sequences.
            evalue: Maximum BLAST E-value threshold.
            min_coverage: Minimum query coverage threshold as fraction
                (``0.9`` for 90%) or percentage value.
            max_targets: Maximum number of target sequences reported by BLAST.

        Returns:
            DataFrame with filtered BLAST hits and parsed PDB identifiers.
        """
        blast_coverage = min_coverage * 100 if min_coverage <= 1 else min_coverage
        logger.info(
            "Searching BLAST database %s with query %s",
            self.index_file,
            query_fasta,
        )
        with tempfile.NamedTemporaryFile() as temp:
            out_str = "6 qseqid sseqid pident length qcovs evalue bitscore"
            subprocess.run(
                [
                    "blastp",
                    "-query", query_fasta,
                    "-db", str(self.index_file),
                    "-evalue", str(evalue),
                    "-qcov_hsp_perc", str(blast_coverage),
                    "-max_target_seqs", str(max_targets),
                    "-outfmt", str(out_str)
                ],
                stdout=temp,
                check=True,
            )
            results = self._convert_output_to_df(temp.name)
            logger.info("BLAST search returned %d filtered hits", len(results))
            return results

    def _convert_output_to_df(self, tempfile_path):
        """Convert tabular BLAST output into a filtered pandas DataFrame.

        Args:
            tempfile_path: Path to the temporary BLAST output file.

        Returns:
            DataFrame containing hits above the internal similarity cutoff.
        """
        rows = []
        with open(tempfile_path, 'r') as fp:
            for k, line in enumerate(fp):
                line = line.strip()
                allele, chain, sim, _, _, evalue, _ = line.split('\t')
                _, pdb_id, chain_id = chain.split('|')
                evalue = float(evalue)
                sim = float(sim)
                if sim > 50:
                    rows.append({
                        "pdb_id": pdb_id,
                        'chain_id': chain_id,
                        "allele": allele,
                        "e-value": evalue,
                        "similarity": sim
                    })

        df_pdb = pd.DataFrame(rows)
        return df_pdb
