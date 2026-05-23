"""
Use this script to generate TFMN for every model. This is aggregated
by model and question. Output contains a network for each model and for each question.

Example output:

- model_x/question1/tfmn.gefx (.gefx is the standard extenstion to save graphs to disk)
- model_x/question2/tfmn.gefx
- ...
- model_z/question7/tfmn.gefx

NOTE: The script takes a lot of time to run. (Around 6/7h)
"""

import pandas as pd
import networkx as nx
from emoatlas import EmoScores
from tqdm import tqdm
from pathlib import Path
import re

from src.utils.constants import MAPPING_CALL1_QUESTIONS

def sanitize_filename(name: str) -> str:
    """Replaces slashes and illegal characters to create safe folder names."""
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))

def get_text_chunks(texts, max_chars=500000):
    """Yields chunks of text safely under the spaCy character limit for maximum speed."""
    current_chunk = []
    current_length = 0
    
    for text in texts:
        text = str(text).strip()
        if not text or text.lower() == 'nan':
            continue
            
        if current_length + len(text) > max_chars:
            yield " ".join(current_chunk)
            current_chunk = [text]
            current_length = len(text)
        else:
            current_chunk.append(text)
            current_length += len(text)
            
    if current_chunk:
        yield " ".join(current_chunk)

def generate_and_save_model_networks(csv_path: Path, output_base_dir: Path):
    """
    Groups data by model and question, builds a TFMN for each, 
    and saves them in a nested folder structure.
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    # Replace the full questions with their numbers
    df["question_number"] = df["question_number"].map(MAPPING_CALL1_QUESTIONS)
    
    # If only considering the human persona simulations, uncomment the next line
    # df = df[df['mode'] == 'human']

    # Group the dataframe by Model and Question
    grouped = df.groupby(['model_name', 'question_number'])
    
    print("Loading emoatlas language models (this only happens once)...")
    emos = EmoScores(language="english")
    
    # Create the base output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # tqdm progress bar for the overall grouping process
    for (model_name, question_num), group_df in tqdm(grouped, desc="Processing Model/Question Groups"):
        
        # 1. Setup the safe directory paths
        safe_model_name = sanitize_filename(model_name)
        safe_question_name = f"question_{sanitize_filename(question_num)}"
        
        target_dir = output_base_dir / safe_model_name / safe_question_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = target_dir / "tfmn.gexf"
        
        # Skip if already processed (great for resuming interrupted runs!)
        if output_file.exists():
            continue

        # 2. Extract text and prepare chunks
        texts = group_df['answer_text'].tolist()
        chunks = list(get_text_chunks(texts, max_chars=500000))
        
        G_master = nx.Graph()

        # 3. Process chunks and merge the network
        for chunk in tqdm(chunks, desc=f"Parsing {safe_model_name[:15]} Q{question_num}", leave=False):
            try:
                g_chunk = emos.formamentis_network(chunk)
                
                # --- UNIVERSAL GRAPH ADAPTER ---
                edges_to_merge = []
                
                # Case 1: emoatlas returned an igraph Graph
                if hasattr(g_chunk, "es") and hasattr(g_chunk, "vs"):
                    has_weight = "weight" in g_chunk.es.attributes()
                    for edge in g_chunk.es:
                        u = g_chunk.vs[edge.source]["name"]
                        v = g_chunk.vs[edge.target]["name"]
                        w = edge["weight"] if has_weight else 1
                        edges_to_merge.append((u, v, w))
                        
                # Case 2: emoatlas returned a standard NetworkX graph
                elif hasattr(g_chunk, "edges") and callable(g_chunk.edges):
                    for u, v, data in g_chunk.edges(data=True):
                        w = data.get('weight', 1) if isinstance(data, dict) else 1
                        edges_to_merge.append((u, v, w))
                        
                # Case 3: emoatlas returned a custom object where .edges is a plain list
                elif hasattr(g_chunk, "edges"):
                    for edge_item in g_chunk.edges:
                        # Expecting tuples like (word1, word2) or (word1, word2, {'weight': 1})
                        if isinstance(edge_item, (tuple, list)):
                            u = edge_item[0]
                            v = edge_item[1]
                            w = edge_item[2].get('weight', 1) if len(edge_item) > 2 and isinstance(edge_item[2], dict) else 1
                            edges_to_merge.append((u, v, w))
                else:
                    tqdm.write(f"Warning: Unknown graph format returned by emoatlas.")
                    continue

                # Safely sum the extracted edge weights into our NetworkX master graph
                for u, v, weight in edges_to_merge:
                    if G_master.has_edge(u, v):
                        G_master[u][v]['weight'] += weight
                    else:
                        G_master.add_edge(u, v, weight=weight)
                        
            except Exception as e:
                # If a chunk fails, we log it and keep going without crashing the whole pipeline!
                tqdm.write(f"Error parsing chunk for {model_name} Q{question_num}: {e}")
                continue
                
        # 4. Save the completed graph as a .gexf file
        if G_master.number_of_nodes() > 0:
            nx.write_gexf(G_master, output_file)
        else:
            tqdm.write(f"Warning: No valid network generated for {model_name} Q{question_num}")

    print(f"\nAll networks successfully saved to: {output_base_dir.resolve()}")

if __name__ == "__main__":
    # Point this to your master dataset
    DATASET_PATH = Path("src/data_validation/individual_tfmn/tfmn_dataset.csv")
    
    # Where you want the new folder structure to be created
    OUTPUT_DIRECTORY = Path("src/data_validation/networks_by_model")
    
    generate_and_save_model_networks(DATASET_PATH, OUTPUT_DIRECTORY)