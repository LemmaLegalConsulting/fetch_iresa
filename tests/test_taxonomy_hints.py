from app.providers.base import clear_all_prompt_caches, load_prompt


def test_hints_omitted_for_iresa_taxonomy_when_no_matches() -> None:
    """Hints block should be omitted when no configured labels are present."""
    taxonomy = [{"Category": "finanziell", "Subcategory": "unknown_subcategory"}]
    clear_all_prompt_caches()
    prompt, _ = load_prompt("openai", taxonomy, taxonomy_name="iresa")
    assert "TAXONOMY HINTS" not in prompt
    assert "unknown_subcategory" in prompt


def test_hints_loaded_for_iresa_when_label_matches() -> None:
    taxonomy = [
        {
            "Category": "finanziell",
            "Subcategory": "kommunale_eingliederungsleistungen_kinderbetreuung_SGBII",
        }
    ]
    clear_all_prompt_caches()
    prompt, _ = load_prompt("openai", taxonomy, taxonomy_name="iresa")
    assert "HINTS:" in prompt
