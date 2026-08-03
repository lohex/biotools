"""Tests for RCSB PDB metadata retrieval."""

from __future__ import annotations

from datetime import date
import json
from urllib.error import HTTPError, URLError
import unittest
from unittest.mock import MagicMock, patch

from biotools import pdbtools
from biotools.structure.metadata import (
    PDBMetadata,
    RCSBMetadataError,
    get_pdb_metadata,
)


def _json_response(payload) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    return response


class PDBMetadataTests(unittest.TestCase):
    @patch("biotools.structure.metadata.urlopen")
    def test_fetches_dates_and_unique_polymer_organisms(
        self,
        open_url: MagicMock,
    ) -> None:
        entry = {
            "rcsb_accession_info": {
                "deposit_date": "1984-03-07T00:00:00.000+00:00",
                "initial_release_date": "1984-07-17T00:00:00.000+00:00",
            },
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": ["1", "2"],
            },
        }
        entity_1 = {
            "rcsb_entity_source_organism": [
                {"ncbi_scientific_name": "Homo sapiens"},
            ]
        }
        entity_2 = {
            "rcsb_entity_source_organism": [
                {"ncbi_scientific_name": "Homo sapiens"},
                {"scientific_name": "Synthetic construct"},
            ]
        }
        open_url.side_effect = [
            _json_response(entry),
            _json_response(entity_1),
            _json_response(entity_2),
        ]

        result = get_pdb_metadata("4hhb", timeout=3.5)

        self.assertEqual(
            result,
            PDBMetadata(
                pdb_id="4HHB",
                organisms=("Homo sapiens", "Synthetic construct"),
                deposited=date(1984, 3, 7),
                released=date(1984, 7, 17),
            ),
        )
        urls = [call.args[0].full_url for call in open_url.call_args_list]
        self.assertEqual(
            urls,
            [
                "https://data.rcsb.org/rest/v1/core/entry/4HHB",
                "https://data.rcsb.org/rest/v1/core/polymer_entity/4HHB/1",
                "https://data.rcsb.org/rest/v1/core/polymer_entity/4HHB/2",
            ],
        )
        self.assertTrue(
            all(call.kwargs["timeout"] == 3.5 for call in open_url.call_args_list)
        )

    @patch("biotools.structure.metadata.urlopen")
    def test_allows_entries_without_release_date(
        self,
        open_url: MagicMock,
    ) -> None:
        open_url.return_value = _json_response(
            {
                "rcsb_accession_info": {
                    "deposit_date": "2026-01-02T00:00:00+00:00",
                },
                "rcsb_entry_container_identifiers": {
                    "polymer_entity_ids": [],
                },
            }
        )

        result = get_pdb_metadata("9XYZ")

        self.assertEqual(result.organisms, ())
        self.assertIsNone(result.released)

    @patch("biotools.structure.metadata.urlopen")
    def test_converts_not_found_to_file_not_found(
        self,
        open_url: MagicMock,
    ) -> None:
        open_url.side_effect = HTTPError(
            "https://data.rcsb.org/rest/v1/core/entry/0ABC",
            404,
            "Not Found",
            hdrs=None,
            fp=None,
        )

        with self.assertRaisesRegex(FileNotFoundError, "RCSB resource"):
            get_pdb_metadata("0ABC")

    @patch("biotools.structure.metadata.urlopen")
    def test_wraps_network_errors(self, open_url: MagicMock) -> None:
        open_url.side_effect = URLError("offline")

        with self.assertRaisesRegex(RCSBMetadataError, "Could not reach"):
            get_pdb_metadata("4HHB")

    def test_validates_identifier_and_timeout_before_request(self) -> None:
        for invalid_id in ("", "ABC", "ABCDE", "A?CD", 1234):
            with self.subTest(pdb_id=invalid_id), self.assertRaises(ValueError):
                get_pdb_metadata(invalid_id)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "timeout"):
            get_pdb_metadata("4HHB", timeout=0)

    def test_pdbtools_facade_exports_metadata_api(self) -> None:
        self.assertIs(pdbtools.get_pdb_metadata, get_pdb_metadata)
        self.assertIs(pdbtools.PDBMetadata, PDBMetadata)
        self.assertIs(pdbtools.RCSBMetadataError, RCSBMetadataError)


if __name__ == "__main__":
    unittest.main()
