import asyncio
from app.providers.spot import SpotProvider
from app.core.config import TAXONOMY_MAPPING
import os

if os.getenv("ENV") != "production":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
from app.utils.csv_helpers import read_csv_as_list_of_dicts


async def run_spot_test():
    spot_provider = SpotProvider()

    # Load the sample data
    script_dir = os.path.dirname(__file__)
    sample_data_path = os.path.join(script_dir, "promptfoo", "iresa_sample_data.csv")

    try:
        sample_rows = read_csv_as_list_of_dicts(sample_data_path)
    except FileNotFoundError:
        print(f"Error: Sample data file not found at {sample_data_path}")
        return

    # Load the taxonomy (required by classify method, even if not directly used by SPOT)
    taxonomy_file = TAXONOMY_MAPPING.get("iresa")
    if not taxonomy_file:
        print("Error: IRESA taxonomy not found in configuration.")
        return

    # Adjust taxonomy_file path to be absolute
    taxonomy_file_absolute = os.path.join(script_dir, taxonomy_file)
    try:
        taxonomy_rows = read_csv_as_list_of_dicts(taxonomy_file_absolute)
    except FileNotFoundError:
        print(f"Error: Taxonomy file not found at {taxonomy_file_absolute}")
        return

    print("Running SPOT classifier tests...")
    for row in sample_rows:
        problem_description = row.get("problem_description")
        if not problem_description:
            continue
        print(f"\n--- Classifying: {problem_description[:70]}...")
        try:
            # The classify method expects a taxonomy list of dicts
            result = await spot_provider.classify(problem_description, taxonomy_rows)
            print(f"Mapped Labels: {result.get('labels')}")
            print(f"Follow-up Questions: {result.get('questions')}")
        except Exception as e:
            print(f"An error occurred during classification: {e}")


if __name__ == "__main__":
    # Set PYTHONPATH to include the project root for module imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in os.sys.path:
        os.sys.path.insert(0, project_root)

    asyncio.run(run_spot_test())
