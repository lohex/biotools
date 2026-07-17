"""Local protein BLAST database helpers."""

import logging
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)


class BlastSearch:
    """Create and query a local BLAST protein database from a FASTA file."""

    def __init__(self, input_fasta: str, db_name: str, description: str = None):
        """Initialize a BLAST database if the index files are missing."""
        if description is None:
            description = db_name

        self.index_file = Path(input_fasta).parent / db_name
        final_index = Path(str(self.index_file) + ".pdb")
        if not final_index.exists():
            logger.info("Creating BLAST protein database at %s", self.index_file)
            subprocess.run(
                [
                    "makeblastdb",
                    "-in",
                    input_fasta,
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
            logger.debug("Using existing BLAST database at %s", self.index_file)

    def search(
        self,
        query_fasta,
        evalue=1e-10,
        min_coverage=0.9,
        max_targets=1000,
    ) -> pd.DataFrame:
        """Search the local BLAST database with protein query sequences."""
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
                    "-query",
                    query_fasta,
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
            logger.info("BLAST search returned %d filtered hits", len(results))
            return results

    def _convert_output_to_df(self, tempfile_path):
        """Convert tabular BLAST output into a filtered pandas DataFrame."""
        rows = []
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
