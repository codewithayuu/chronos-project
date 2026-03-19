"""
Drug Database — Pharmacological lookup table.

Loads drug profiles from a JSON file and provides lookup by name or class.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

from ..models import DrugDatabaseEntry


class DrugDatabase:
    """
    In-memory drug database loaded from JSON.
    Provides O(1) lookups by drug name and O(n) lookups by drug class.
    """

    def __init__(self, path: str = "data/drug_database.json"):
        self.drugs: List[DrugDatabaseEntry] = []
        self._by_name: Dict[str, DrugDatabaseEntry] = {}
        self._by_class: Dict[str, List[DrugDatabaseEntry]] = {}
        self._load(path)

    def _load(self, path: str):
        """Load drug database from JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            print(f"[DrugDB] WARNING: {path} not found. Using empty drug database.")
            return

        with open(file_path, "r") as f:
            raw = json.load(f)

        for entry in raw:
            drug = DrugDatabaseEntry(**entry)
            self.drugs.append(drug)
            self._by_name[drug.drug_name.lower()] = drug

            if drug.drug_class not in self._by_class:
                self._by_class[drug.drug_class] = []
            self._by_class[drug.drug_class].append(drug)

        print(f"[DrugDB] Loaded {len(self.drugs)} drugs across {len(self._by_class)} classes.")

    def lookup(self, drug_name: str) -> Optional[DrugDatabaseEntry]:
        """Look up a drug by name (case-insensitive)."""
        return self._by_name.get(drug_name.lower())

    def lookup_by_class(self, drug_class: str) -> List[DrugDatabaseEntry]:
        """Get all drugs in a given class."""
        return self._by_class.get(drug_class, [])

    def get_all_classes(self) -> List[str]:
        """Return all drug classes in the database."""
        return list(self._by_class.keys())

    def get_affected_vitals(self, drug_name: str) -> List[str]:
        """
        Return list of vital signs that a drug is expected to affect.
        Maps drug effect fields to our vital names.
        """
        drug = self.lookup(drug_name)
        if drug is None:
            return []

        affected = []
        if drug.expected_hr_effect != "none":
            affected.append("heart_rate")
        if drug.expected_bp_effect != "none":
            affected.append("bp_systolic")
            affected.append("bp_diastolic")
        if drug.expected_rr_effect != "none":
            affected.append("resp_rate")
        if drug.expected_spo2_effect != "none":
            affected.append("spo2")
        return affected

    def get_expected_change(self, drug_name: str, vital_name: str) -> Optional[float]:
        """
        Get the expected magnitude of change for a vital due to a drug.
        Returns the signed magnitude (negative = decrease).
        Returns None if drug doesn't affect this vital.
        """
        drug = self.lookup(drug_name)
        if drug is None:
            return None

        mapping = {
            "heart_rate": (drug.expected_hr_effect, drug.expected_hr_magnitude),
            "bp_systolic": (drug.expected_bp_effect, drug.expected_bp_magnitude),
            "bp_diastolic": (drug.expected_bp_effect, drug.expected_bp_magnitude * 0.7),
            "resp_rate": (drug.expected_rr_effect, drug.expected_rr_magnitude),
            "spo2": (drug.expected_spo2_effect, drug.expected_spo2_magnitude),
            "temperature": ("none", 0.0),
        }

        effect, magnitude = mapping.get(vital_name, ("none", 0.0))
        if effect == "none":
            return None
        return magnitude

    def __len__(self):
        return len(self.drugs)
