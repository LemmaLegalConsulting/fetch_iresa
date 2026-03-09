import os
import sys
import textwrap

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.data.list_taxonomy import load_list_taxonomy_simple
from app.providers.base import clear_all_prompt_caches, load_prompt


def _print_section(title: str, text: str) -> None:
    print(f"\n== {title} ==")
    print(textwrap.shorten(text, width=800, placeholder=" ..."))


def main() -> None:
    sample_input = "I am getting kicked out"

    default_taxonomy = [{"Category": "Real Property", "Subcategory": "Tenant (Residential)"}]
    clear_all_prompt_caches()
    prompt_with_hints, _ = load_prompt(
        "openai",
        default_taxonomy,
        taxonomy_name="default",
    )

    clear_all_prompt_caches()
    prompt_without_hints, _ = load_prompt(
        "openai",
        default_taxonomy,
        taxonomy_name=None,
    )

    _print_section("Input", sample_input)
    _print_section("Default taxonomy (hints on)", prompt_with_hints)
    _print_section("Default taxonomy (hints off)", prompt_without_hints)

    list_taxonomy = load_list_taxonomy_simple()
    clear_all_prompt_caches()
    list_prompt, _ = load_prompt(
        "openai",
        list_taxonomy,
        taxonomy_name="list",
    )
    _print_section("LIST taxonomy (no hints configured)", list_prompt)


if __name__ == "__main__":
    main()
