import pandas as pd
import networkx as nx
import numpy as np
from scipy import stats
from pathlib import Path

def text_cleaning(df):
    """Cleans text columns by making them lowercase and removing digits/punctuation."""
    df = df.copy()
    columns_to_clean = ["cue_word", "association_1", "association_2", "association_3"]
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace("à", "a")
                .str.replace(r"[\d\W_]+", " ", regex=True)
                .str.strip()
            )
    return df

def build_association_df(df, filtering="no"):
    """Transforms wide format data to long format."""
    df["cue_word_original"] = df["cue_word"]
    df_long = df.rename(
        columns={
            "cue_word": "association_0",
            "valence_cue_word": "valence_association_0",
        }
    )
    df_long = pd.wide_to_long(
        df_long,
        stubnames=["association", "valence_association"],
        i=["run_id", "cue_word_original"],
        j="index",
        sep="_",
        suffix=r"\d+",
    ).reset_index()

    df_long = df_long.sort_values(by=["cue_word_original", "index"]).reset_index(drop=True)
    df_long = df_long.rename(
        columns={
            "cue_word_original": "cue_word",
            "association": "word",
            "valence_association": "valence",
        }
    )
    df_long = df_long.dropna()
    df_long = df_long[(df_long["word"] != "nan") & (df_long["word"] != "")]

    if filtering == "yes":
        df_long = df_long.groupby(["cue_word", "word"]).filter(
            lambda g: g["run_id"].nunique() > 1
        )
    return df_long

def categorize_valence_multiple(df):
    """Categorizes word valence (-1, 0, 1) based on global distribution."""
    all_words = list(w for w in df["word"].unique())
    valence_map = {}

    for word in all_words:
        keyword_valence = df["valence"][df["word"] == word]
        all_valences = df["valence"][~(df["word"] == word)]
        count = len(keyword_valence)
        
        if count < 3:
            valence_map[word] = 0
        else:
            stat, p_value = stats.kruskal(keyword_valence, all_valences)
            mean_keyword_valence = np.mean(keyword_valence)
            mean_valences = np.mean(all_valences)

            if p_value < 0.1 and mean_keyword_valence < mean_valences:
                valence_map[word] = -1
            elif p_value < 0.1 and mean_keyword_valence > mean_valences:
                valence_map[word] = 1
            else:
                valence_map[word] = 0

    df["valence_cat"] = df["word"].map(valence_map)
    return df

def extract_aura_data(df, keywords, model_name):
    """Calculates the aura fractions (pos, neg, net) for target keywords."""
    G = nx.Graph()
    valence_dict = dict(zip(df["word"], df["valence_cat"]))

    pos_nodes = {node for node, val in valence_dict.items() if val == 1}
    neg_nodes = {node for node, val in valence_dict.items() if val == -1}

    # Build edges from cue to association
    edges = [(cue, word) for cue, word in zip(df["cue_word"], df["word"]) if cue != word]
    G.add_edges_from(edges)

    aura_results = []
    for kw in keywords:
        if kw in G:
            neighbors = list(G.neighbors(kw))
            total = len(neighbors)
            if total > 0:
                pos_frac = sum(1 for n in neighbors if n in pos_nodes) / total
                neg_frac = sum(1 for n in neighbors if n in neg_nodes) / total
                
                aura_results.append({
                    "model_name": model_name,
                    "cue_word": kw,
                    "aura_positive_fraction": pos_frac,
                    "aura_negative_fraction": neg_frac,
                    "aura_net_valence": pos_frac - neg_frac,
                    "total_associations": total
                })
    return aura_results

if __name__ == "__main__":
    # 1. Setup paths and targets
    target_keywords = ["mathematic", "science", "physics"]
    base_dir = Path("src/data_validation/NEW_final_wide_format_dataset").resolve().absolute()
    output_file = "aggregated_aura_valences.csv"

    all_model_data = []

    # 2. Iterate through folders and files
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found.")
    else:
        for dir_path in base_dir.iterdir():
            if dir_path.is_dir() and dir_path.name != "out":
                for file_path in dir_path.glob("*.csv"):
                    print(f"Processing {file_path.name}...")
                    
                    # Run the pipeline
                    raw_df = pd.read_csv(file_path)
                    clean_df = text_cleaning(raw_df)
                    long_df = build_association_df(clean_df)
                    valence_df = categorize_valence_multiple(long_df)
                    
                    # Extract numeric aura data
                    aura_data = extract_aura_data(valence_df, target_keywords, file_path.stem)
                    all_model_data.extend(aura_data)

        # 3. Aggregate and Save
        if all_model_data:
            final_df = pd.DataFrame(all_model_data)
            final_df.to_csv(output_file, index=False)
            print(f"\n✅ Success! Aggregated aura dataset saved to: {output_file}")
            print(final_df.head())
        else:
            print("No data was extracted. Check your file paths and cue_word matches.") 