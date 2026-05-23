import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import nltk
import numpy as np
import pandas as pd
import scipy.stats as stats
import spacy
from nltk import word_tokenize
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ==========================================
# 1. TEXT PROCESSING ENGINE
# ==========================================

class TextProcessor:
    """Handles NLP model loading, text cleaning, valence mapping, and edge list extraction."""
    
    def __init__(self, language: str = "eng"):
        self.language = language
        self.spacy_model_name = "it_core_news_lg" if language == "ita" else "en_core_web_lg"
        self.nlp = self._load_spacy()
        self._ensure_nltk()

    def _load_spacy(self):
        """Loads the required spaCy model, downloading it if necessary."""
        try:
            return spacy.load(self.spacy_model_name)
        except OSError:
            logger.info(f"Downloading spaCy model {self.spacy_model_name}...")
            from spacy.cli import download
            download(self.spacy_model_name)
            return spacy.load(self.spacy_model_name)

    def _ensure_nltk(self):
        """Ensures required NLTK tokenizers are downloaded."""
        for package in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package, quiet=True)

    def _process_with_nltk(self, text: str) -> str:
        """Tokenizes and cleans a single text string."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower().replace("à", "a")
        tokens = word_tokenize(text)
        
        # Keep only alphabetic tokens (ignoring hyphens for the check)
        clean_tokens = [word for word in tokens if word.replace("-", "").isalpha()]
        return " ".join(clean_tokens)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies NLTK cleaning to DataFrame columns and drops invalid rows."""
        if df.empty:
            return df
            
        for col in ["cue_word", "word"]:
            if col in df.columns:
                df[col] = df[col].apply(self._process_with_nltk)
        
        df = df[(df["word"] != "nan") & (df["word"] != "")]
        df = df.dropna(subset=["word", "valence"])
        
        df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
        if "cue_valence" in df.columns:
            df["cue_valence"] = pd.to_numeric(df["cue_valence"], errors="coerce")
            
        df = df.dropna(subset=["valence"])
        
        return df

    def categorize_valence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizes valence based on quartiles and mode."""
        if df.empty:
            return df
            
        arr = np.array(df["valence"])
        if len(arr) == 0:
            logger.warning("Array of valences is 0!")
            df["associated_valence_cat"] = 0
            df["cue_valence_cat"] = 0
            return df
            
        Q1, Q3 = np.percentile(arr, [25, 75])
        
        mode_result = stats.mode(arr, keepdims=False)
        mode_val = mode_result.mode if hasattr(mode_result, 'mode') else mode_result[0]
        if isinstance(mode_val, np.ndarray):
            mode_val = mode_val[0]

        def cat_score(v):
            if pd.isna(v):
                return "" # Return blank if a specific valence is missing
            iqr = abs(Q1 - Q3)
            if iqr == 0:
                if v == mode_val: return 0
                return -1 if v < mode_val else 1
            elif iqr == 1:
                if v > Q3: return 1
                if v < Q1: return -1
                return 0
            else:
                if v <= Q1: return -1
                if v >= Q3: return 1
                return 0

        # Apply the exact same logic to both valences to keep everything equal
        df["associated_valence_cat"] = df["valence"].apply(cat_score)
        if "cue_valence" in df.columns:
            df["cue_valence_cat"] = df["cue_valence"].apply(cat_score)
        else:
            df["cue_valence_cat"] = ""
            
        return df

    def extract_lemmatized_edgelist(
        self, 
        df: pd.DataFrame, 
        out_file_path: Optional[Path] = None
    ) -> Optional[pd.DataFrame]:
        """Extracts a lemmatized edge list with valence and optionally saves to CSV."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        def get_lemma(text):
            if pd.isna(text):
                return text
            return " ".join([token.lemma_ for token in self.nlp(str(text).lower())])

        process_df = df.copy() 
        process_df["cue_lemma"] = process_df["cue_word"].apply(get_lemma)
        process_df["word_lemma"] = process_df["word"].apply(get_lemma)

        edges_df = process_df[process_df["cue_lemma"] != process_df["word_lemma"]].copy()
        
        # Keep exactly 4 columns and rename them to your requirements
        edges_df = edges_df[["cue_lemma", "word_lemma", "cue_valence_cat", "associated_valence_cat"]]
        edges_df = edges_df.rename(
            columns={
                "cue_lemma": "cue_word", 
                "word_lemma": "association_word", 
                "cue_valence_cat": "cue_valence",
                "associated_valence_cat": "associated_valence"
            }
        )
        edges_df = edges_df.drop_duplicates(subset=["cue_word", "association_word"])

        if out_file_path:
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            edges_df.to_csv(out_file_path, index=False)
            
        return edges_df


# ==========================================
# 2. PIPELINE MANAGER
# ==========================================

class FormamentisPipeline:
    """Orchestrates file reading, batch matching, and executing the text processor."""
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None, filter_str: Optional[str] = None):
        self.input_dir = Path(input_dir).resolve()
        self.output_base_dir = Path(output_dir).resolve() if output_dir else None
        self.filter_str = filter_str
        self.processor = TextProcessor(language="eng")  # Instantiate the NLP engine

    @staticmethod
    def parse_json_to_long_df(file_path: Path) -> pd.DataFrame:
        """Reads JSON and converts nested Formamentis data into a long-format DataFrame."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        participant_id = data.get("run_id")
        forma_mentis = data.get("response_parsed", {}).get("parsed", {}).get("forma_mentis", {})
        
        rows = []
        for cue_word, data_dict in forma_mentis.items():
            associations = data_dict.get("associations", [])
            valences = data_dict.get("valence", {})
            
            # Extract the valence for the cue_word itself
            cue_valence = valences.get(cue_word, data_dict.get("cue_valence", None))
            
            for word in associations:
                rows.append({
                    "participant_id": participant_id,
                    "cue_word": cue_word,
                    "word": word,
                    "cue_valence": cue_valence,
                    "valence": valences.get(word, None)
                })
        return pd.DataFrame(rows)

    def process_model_directory(self, model_dir: Path, rel_path: Path):
        """Handles batch matching and pipeline execution for a single model directory."""
        print(f"\nProcessing model: {rel_path}")
        
        model_out_dir = self.output_base_dir / rel_path if self.output_base_dir else None
        if model_out_dir:
            model_out_dir.mkdir(parents=True, exist_ok=True)

        json_files_batch1 = list(model_dir.glob("*call3*batch1*.json"))
        json_files_batch2 = list(model_dir.glob("*call3*batch2*.json"))
        
        if len(json_files_batch1) != len(json_files_batch2):
            logger.error("Unequal numbers of batch1 and batch2 files")
        
        for f_b1 in tqdm(json_files_batch1, desc=f"Processing {rel_path.name}"):
            filename_parts = f_b1.name.split('_')
            if len(filename_parts) < 3:
                continue
                
            run_id_hash = filename_parts[2]
            logger.debug(f"Processing run id: {run_id_hash}")
            matching_b2_files = [f for f in json_files_batch2 if run_id_hash in f.name]
            
            if not matching_b2_files:
                continue
            
            df_b1 = self.parse_json_to_long_df(f_b1)
            df_b2 = self.parse_json_to_long_df(matching_b2_files[0])
            df = pd.concat([df_b1, df_b2], ignore_index=True)
            
            if df.empty:
                continue
            
            run_id = df["participant_id"].iloc[0]
            
            # Utilize the TextProcessor engine
            clean_df = self.processor.clean_dataframe(df)
            final_df = self.processor.categorize_valence(clean_df)
            
            if final_df.empty:
                logger.warning(f"Skipping {run_id}: DataFrame is empty after cleaning (likely missing valences from LLM).")
                continue
            
            run_out_dir = model_out_dir / str(run_id) if model_out_dir else None
            out_file = run_out_dir / "edgelist.csv" if run_out_dir else None
            
            self.processor.extract_lemmatized_edgelist(df=final_df, out_file_path=out_file)

    def run(self):
        """Executes the full pipeline across all valid directories."""
        if not self.input_dir.exists():
            print(f"Error: Input directory {self.input_dir} does not exist.")
            sys.exit(1)
            
        call3_files = list(self.input_dir.rglob("*call3*.json"))
        model_dirs = set(f.parent for f in call3_files)
        
        if not model_dirs:
            print(f"No call3 JSON files found in {self.input_dir}")
            sys.exit(0)
        
        for model_dir in model_dirs:
            rel_path = model_dir.relative_to(self.input_dir)
            
            if self.filter_str and self.filter_str not in str(rel_path):
                continue
                
            self.process_model_directory(model_dir, rel_path)

        logger.info("\nPipeline finished successfully!")


# ==========================================
# 3. CLI ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a lemmatized edge list from Formamentis JSONs.")
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory containing JSONs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--filter", type=str, default=None, help="Filter string for folder paths.")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Formamentis analysis pipeline (Saving results: {'Yes' if args.output_dir else 'No'})...")
    
    pipeline = FormamentisPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filter_str=args.filter
    )
    pipeline.run()