from app.core.config import HINTS_MAPPING, TAXONOMY_MAPPING
from app.utils.csv_helpers import read_csv_as_list_of_dicts


def test_iresa_taxonomy_mapping_points_to_runtime_file() -> None:
    assert TAXONOMY_MAPPING["iresa"] == "app/data/taxonomy_iresa.csv"

    rows = read_csv_as_list_of_dicts(TAXONOMY_MAPPING["iresa"])
    assert rows
    assert rows[0]["Category"] == "finanziell"
    assert "subcategory_description_de" in rows[0]
    assert "subcategory_description_en_dev" in rows[0]


def test_iresa_hints_mapping_points_to_hints_file() -> None:
    assert HINTS_MAPPING["iresa"] == "app/data/taxonomy_hints_iresa.yaml"
