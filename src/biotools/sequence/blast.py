"""Local protein BLAST database helpers."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)


class BlastSearch:
    """Create and query a local BLAST protein database from a FASTA file.

    Attributes:
        index_file: Basename of the local BLAST database files.
    """

    def __init__(
        self,
        input_fasta: str | PathLike[str],
        db_name: str,
        description: str | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        """Initialize a BLAST database if its index files are missing.

        Args:
            input_fasta: Protein FASTA file used to build the database.
            db_name: Basename for generated BLAST index files.
            description: Optional database title; defaults to ``db_name``.
            verbose: Log database creation or reuse.

        Raises:
            FileNotFoundError: If ``makeblastdb`` is not installed.
            subprocess.CalledProcessError: If database creation fails.
        """
        if description is None:
            description = db_name

        self.index_file: Path = Path(input_fasta).parent / db_name
        final_index = Path(str(self.index_file) + ".pdb")
        if not final_index.exists():
            if verbose:
                logger.info(
                    "Creating BLAST protein database at %s", self.index_file
                )
            subprocess.run(
                [
                    "makeblastdb",
                    "-in",
                    str(input_fasta),
                    "-dbtype",
                    "prot",
                    "-parse_seqids",
                    "-title",
                    description,
                    "-out",
                    str(self.index_file),
                ],
                check=True,
            )
        else:
            if verbose:
                logger.info(
                    "Using existing BLAST database at %s", self.index_file
                )

    def search(
        self,
        query_fasta: str | PathLike[str],
        evalue: float = 1e-10,
        min_coverage: float = 0.9,
        max_targets: int = 1000,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Search the local BLAST database with protein query sequences.

        Args:
            query_fasta: FASTA file containing one or more protein queries.
            evalue: Maximum E-value accepted by BLAST.
            min_coverage: Minimum query coverage as a fraction or percentage.
            max_targets: Maximum number of target sequences reported per query.
            verbose: Log search progress and the number of filtered hits.

        Returns:
            Filtered hits with PDB ID, chain ID, allele, E-value, and identity.

        Raises:
            FileNotFoundError: If ``blastp`` is not installed.
            subprocess.CalledProcessError: If the BLAST search fails.
        """
        blast_coverage = min_coverage * 100 if min_coverage <= 1 else min_coverage
        if verbose:
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
                    "-query",
                    str(query_fasta),
                    "-db",
                    str(self.index_file),
                    "-evalue",
                    str(evalue),
                    "-qcov_hsp_perc",
                    str(blast_coverage),
                    "-max_target_seqs",
                    str(max_targets),
                    "-outfmt",
                    out_str,
                ],
                stdout=temp,
                check=True,
            )
            results = self._convert_output_to_df(temp.name)
            if verbose:
                logger.info(
                    "BLAST search returned %d filtered hits", len(results)
                )
            return results

    def _convert_output_to_df(
        self,
        tempfile_path: str | PathLike[str],
    ) -> pd.DataFrame:
        """Convert tabular BLAST output into a filtered pandas DataFrame.

        Args:
            tempfile_path: BLAST tabular output file in the expected format.

        Returns:
            Hits with sequence identity greater than 50 percent.

        Raises:
            ValueError: If a row has malformed columns or numeric values.
        """
        rows: list[dict[str, str | float]] = []
        with open(tempfile_path, "r") as fp:
            for line in fp:
                line = line.strip()
                allele, chain, sim, _, _, evalue, _ = line.split("\t")
                _, pdb_id, chain_id = chain.split("|")
                evalue = float(evalue)
                sim = float(sim)
                if sim > 50:
                    rows.append(
                        {
                            "pdb_id": pdb_id,
                            "chain_id": chain_id,
                            "allele": allele,
                            "e-value": evalue,
                            "similarity": sim,
                        }
                    )

        return pd.DataFrame(rows)
