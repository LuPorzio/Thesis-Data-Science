from pathlib import Path
from typing import Optional
import json
import logging

import pandas as pd
from tqdm import tqdm

DEBUG = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_json(path: Path):
    with open(path, "r") as f:
        file = json.load(f)
    return file

def build_persona_dataset(input_dir: Path, output_dir: Optional[Path] = None, model_filter: Optional[str] = None):
    """
    Given a path to a directory containing models and another subdirectory with JSON records,
    extract a dataset containing the flattened persona attributes using a list of dictionaries.
    """
    
    rows = [] # Master list to hold all our row dictionaries

    files_to_process = []
    for model in input_dir.iterdir():
        if not model.is_dir():
            continue
        
        if model_filter and model_filter not in str(model.name).lower():
            continue
            
        files_to_process.extend(list(model.rglob("*call1*")))

    if DEBUG:
        files_to_process = files_to_process[:10]

    # Helper function to safely parse lists into comma-separated strings
    def parse_list(item):
        if isinstance(item, list):
            return ", ".join(str(i) for i in item)
        return str(item) if item is not None else ""

    for file in tqdm(files_to_process, desc="Extracting Personas"):
        data = read_json(file)

        # 1. Extract the mode immediately
        mode = data.get("mode", "unknown")
        
        # 2. Skip this file if the mode is "llm"
        if mode == "llm":
            continue
        
        # Safe fallback: 'or {}' ensures that if a key is explicitly null in the JSON, 
        # it defaults to an empty dictionary rather than a NoneType object.
        persona = data.get("persona") or {}
        parents_edu = persona.get("parents_education") or {}
        ocean = persona.get("ocean") or {}
        
        # Build the row dictionary for this specific file
        row = {
            'run_id': data.get("run_id", "unknown"),
            'model_name': data.get("model", "unknown"),
            'mode': data.get("mode", "unknown"),
            
            'age': persona.get("age"),
            'gender': persona.get("gender") if persona.get("gender") else persona.get("gender_identity"),
            'sexual_orientation': persona.get("sexual_orientation") if persona.get("sexual_orientation") else persona.get("sexual_identity"),
            'city_of_living': persona.get("city_of_living"),
            'employment_status': persona.get("employment_status"),
            'education_level': persona.get("education_level"),
            'marital_status': persona.get("marital_status"),
            'children': persona.get("children"),
            'migration_status': persona.get("migration_status"),
            'religious_beliefs': persona.get("religious_beliefs"),
            
            'parent_1_education': parents_edu.get("parent_1"),
            'parent_2_education': parents_edu.get("parent_2"),
            
            'hobbies': parse_list(persona.get("hobbies")),
            'fav_subjects': parse_list(persona.get("fav_subjects")),
            'hat_subjects': parse_list(persona.get("hat_subjects"))
        }
        
        # Dynamically unpack the OCEAN traits
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            trait_data = ocean.get(trait) or {}
            row[f'ocean_{trait}_score'] = trait_data.get("score")
            row[f'ocean_{trait}_level'] = trait_data.get("level")

        # Append the completed row to our master list
        rows.append(row)

    # Let Pandas handle the alignment and NaN filling natively
    res = pd.DataFrame(rows)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir.joinpath("persona_dataset.csv")
        res.to_csv(out_file, index=False)
        logger.info(f"Persona dataset saved to {out_file}")
        
    return res

if __name__ == "__main__":
    FILTERED_DATA_PATH = Path("data/01-original_data").resolve().absolute()
    OUTPUT_DIR_PATH = Path("code/Task1/individual_tfmn").resolve().absolute()
    
    df_personas = build_persona_dataset(FILTERED_DATA_PATH, OUTPUT_DIR_PATH)
    print(df_personas.head())