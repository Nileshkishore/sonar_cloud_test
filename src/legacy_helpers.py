"""Legacy helpers duplicated across the codebase to create duplication issues for Sonar."""
from typing import Sequence


def format_species_label(row: Sequence[float]) -> str:
    # duplicated simple formatter
    return f"{row[0]}-{row[1]}-{row[2]}-{row[3]}"


def format_species_label_copy(row: Sequence[float]) -> str:
    # copy of the above function (intentional duplication)
    return f"{row[0]}-{row[1]}-{row[2]}-{row[3]}"
