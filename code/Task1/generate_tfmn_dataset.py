"""
Module to generate Textual Forma Mentis networks and extract z-scores from
the dataset looking at call1 records.
"""
from pathlib import Path
from typing import List, Optional
import json
import logging

import pandas as pd
from emoatlas import EmoScores
from tqdm import tqdm

DEBUG = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_json(path: Path):
    with open(path, "r") as f:
        file = json.load(f)
    return file

def build_dataset(input_dir: Path, output_dir: Optional[Path] = None, model_filter: Optional[str] = None):
    """
    Given a path to a directory containing models and another subdirectory with JSON records
    extract a dataset containing z scores for each model.
    """
    
    # Initialize EmoScores ONCE to save massive amounts of processing time
    logger.info("Loading emoatlas language models...")
    emos = EmoScores(language="english")

    df = { 
        'model_name' : [],
        'run_id' : [],
        'mode' : [],
        'question_number' : [],
        'answer_text' : [],
        'z_scores_anger' : [],
        'z_scores_trust' : [],
        'z_scores_surprise' : [],
        'z_scores_disgust' : [],
        'z_scores_joy' : [],
        'z_scores_sadness' : [],
        'z_scores_fear' : [],
        'z_scores_anticipation' : []
    }

    # 1. Gather all files first so tqdm knows the total count for the progress bar
    files_to_process = []
    for model in input_dir.iterdir():
        if not model.is_dir():
            continue
        
        if model_filter and model_filter not in str(model.name).lower():
            logger.info(f"Skipping model {model.name}")
            continue
            
        files_to_process.extend(list(model.rglob("*call1*")))

    # Apply debug limit if necessary
    if DEBUG:
        files_to_process = files_to_process[:10]
        logger.info("DEBUG mode enabled: Limiting to 10 files total.")

    # 2. Process the files with a progress bar
    for file in tqdm(files_to_process, desc="Extracting Z-Scores"):
        data = read_json(file)
        
        # Depending on JSON structure, ensure these keys exist
        model_name = [data.get("model", "unknown")]
        run_id = [data.get("run_id", "unknown")]
        mode = [data.get("mode", "unknown")]
        
        questions: List = data["response_parsed"]["input_topics"]
        replies = [ans for _, ans in data["response_parsed"]["parsed"]["replies"].items()]
        
        anger, trust, surprise, disgust = [], [], [], []
        joy, sadness, fear, anticipation = [], [], [], []

        n_questions = len(questions)

        for reply in replies:
            # Use the pre-loaded emos instance
            dct = emos.zscores(reply)
            anger.append(dct["anger"])
            trust.append(dct["trust"])
            surprise.append(dct["surprise"])
            disgust.append(dct["disgust"])
            joy.append(dct["joy"])
            sadness.append(dct["sadness"])
            fear.append(dct["fear"])
            anticipation.append(dct["anticipation"])
        
        extended_model_name = model_name * n_questions
        extended_run_id = run_id * n_questions
        extended_mode = mode * n_questions

        df["model_name"].extend(extended_model_name)
        df["run_id"].extend(extended_run_id)
        df["mode"].extend(extended_mode)
        df["question_number"].extend(questions)
        df["answer_text"].extend(replies)
        df["z_scores_anger"].extend(anger)
        df["z_scores_trust"].extend(trust)
        df["z_scores_surprise"].extend(surprise)
        df["z_scores_disgust"].extend(disgust)
        df["z_scores_joy"].extend(joy)
        df["z_scores_sadness"].extend(sadness)
        df["z_scores_fear"].extend(fear)
        df["z_scores_anticipation"].extend(anticipation)

    res = pd.DataFrame(df)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        res.to_csv(output_dir.joinpath("tfmn_dataset.csv"), index=False)
        logger.info(f"Dataset saved to {output_dir.joinpath('tfmn_dataset.csv')}")
        
    return res

if __name__ == "__main__":
    FILTERED_DATA_PATH = Path("src/data_validation/complete-models-filtered-all").resolve().absolute()
    OUPUT_DIR_PATH = Path("src/data_validation/individual_tfmn").resolve().absolute()
    
    # Run the builder
    df_results = build_dataset(FILTERED_DATA_PATH, OUPUT_DIR_PATH)
    print(df_results.head())