from app.providers.base import load_prompt


def test_iresa_prompt_includes_german_description_not_dev_reference() -> None:
    taxonomy = [
        {
            "Category": "finanziell",
            "Subcategory": "wohngeld",
            "subcategory_description_de": "Wohngeld",
            "subcategory_description_en_dev": "housing benefit",
        }
    ]

    prompt, _ = load_prompt("openai", taxonomy, taxonomy_name="iresa")

    assert "finanziell > wohngeld > Wohngeld" in prompt
    assert "housing benefit" not in prompt
