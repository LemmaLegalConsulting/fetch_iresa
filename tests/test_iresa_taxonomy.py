from app.core.config import HINTS_MAPPING, TAXONOMY_MAPPING
from app.utils.csv_helpers import read_csv_as_list_of_dicts


def test_iresa_taxonomy_mapping_points_to_placeholder_file() -> None:
    assert TAXONOMY_MAPPING["iresa"] == "app/data/taxonomy_iresa.csv"

    rows = read_csv_as_list_of_dicts(TAXONOMY_MAPPING["iresa"])
    assert rows
    assert rows[0]["Category"] == "IRESA Placeholder"


def test_iresa_hints_mapping_points_to_placeholder_file() -> None:
    assert HINTS_MAPPING["iresa"] == "app/data/taxonomy_hints_iresa.yaml"
