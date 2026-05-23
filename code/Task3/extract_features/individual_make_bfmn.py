import argparse
import logging
from pathlib import Path

import community.community_louvain as community_louvain
from emoatlas import EmoScores
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    datefmt=r"%Y/%m/%d_%H:%M:%S"
)

logger = logging.getLogger(__name__)

def build_bfmn_from_edgelist(csv_path, draw_fmnt: bool = False):
    # 1. Load the edge list
    df = pd.read_csv(csv_path)
    
    # 2. Extract a unified dictionary of words and their valences
    # We iterate through the dataframe to ensure both cue words and association words are captured
    valence_dict = {}
    for _, row in df.iterrows():
        valence_dict[row['cue_word']] = row['cue_valence']
        valence_dict[row['association_word']] = row['associated_valence']
        
    # 3. Initialize an empty undirected graph
    G = nx.Graph()
    
    # Create sets for negative, positive, and neutral words for drawing later
    positive_words = set()
    negative_words = set()
    neutral_words = set()
    
    # 4. Add nodes and categorize them by valence
    for node, val in valence_dict.items():
        G.add_node(node, valence=val)
        if val == 1:
            positive_words.add(node)
        elif val == -1:
            negative_words.add(node)
        elif val == 0:
            neutral_words.add(node)
            
    custom_valences = [positive_words, negative_words, neutral_words]
    
    # 5. Add edges between cue words and their associations
    edges = list(zip(df['cue_word'], df['association_word']))
    G.add_edges_from(edges)
    
    # 6. Initialize EmoScores and convert to a Forma Mentis Network
    # Using English since the edge list contains English words (e.g., "grade", "exam", "science")
    emo = EmoScores(language="english", spacy_model="en_core_web_lg")
    fmn = emo.nxgraph_to_formamentis(G)
    
    # 7. Plot the BFMN
    if draw_fmnt:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.axis("off")
        
        emo.draw_formamentis(
            fmn=fmn,
            layout="edge_bundling",
            custom_valences=custom_valences,
            ax=ax
        )
        
        plt.title("Behavioral Forma Mentis Network", fontsize=16)
        plt.show()
    
    return G, fmn


def extract_network_features(G, df_edgelist, model_name, run_id):
    """Calculates all requested topological and node-level features."""
    
    # 1. Largest Connected Component (LCC)
    if G.number_of_nodes() > 0:
        lcc_nodes = max(nx.connected_components(G), key=len)
        LCC = G.subgraph(lcc_nodes).copy()
    else:
        LCC = nx.Graph()

    # 2. Basic Metrics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_nodes_LCC = LCC.number_of_nodes()
    
    # 3. Degrees and Max Degree
    degrees_dict = dict(G.degree())
    max_deg = max(degrees_dict.values()) if degrees_dict else 0
    
    # 4. LCC Path Length & Diameter
    if LCC.number_of_nodes() > 1:
        avg_path_len_LCC = nx.average_shortest_path_length(LCC)
        diameter_LCC = nx.diameter(LCC)
    else:
        avg_path_len_LCC = 0
        diameter_LCC = 0
        
    # 5. Clustering and Assortativity
    avg_clustering = nx.average_clustering(G)
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = None 
        
    # 6. Modularity (Louvain)
    try:
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
    except Exception:
        modularity = None
        
    # 7. Frequency of vertices
    all_words = pd.concat([df_edgelist['cue_word'], df_edgelist['association_word']])
    frequency_dict = all_words.value_counts().to_dict()
    
    # 8. Topic centralities
    topic_centralities = nx.betweenness_centrality(G)

    return {
        "model_name": model_name,
        "run_id": str(run_id), # Cast to string for cleaner CSV saving
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_nodes_LCC": n_nodes_LCC,
        "max_deg": max_deg,
        "avg_path_len_LCC": avg_path_len_LCC,
        "diameter_LCC": diameter_LCC,
        "average_local_clustering_coefficient": avg_clustering,
        "modularity_louvain": modularity,
        "assortativity_degree": assortativity,
        "degree_of_vertices": str(degrees_dict),
        "frequency_of_vertices": str(frequency_dict),
        "topic_centralities": str(topic_centralities)
    }

def process_all_networks(input_base_dir, output_base_dir):
    """
    Iterates over runs from a single model, extracts features, and saves them.
    """
    input_path = Path(input_base_dir)
    output_path = Path(output_base_dir) if output_base_dir else None
    
    # Pre-gather valid run directories so tqdm knows the total length
    run_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not run_dirs:
        logger.warning(f"No valid run directories found in {input_path}")
        return

    model_name = input_path.name
    
    # Initialize the list OUTSIDE the loop so it accumulates all runs
    model_features_list = []
    
    # Wrap the list in tqdm
    for run_id in tqdm(run_dirs, desc=f"Processing Model: {model_name}", unit="run"):
        edgelist_path = run_id / "edgelist.csv" 
        
        if edgelist_path.exists():
            # Read the edge list
            df = pd.read_csv(edgelist_path)
            
            # Build graph 
            G, _ = build_bfmn_from_edgelist(str(edgelist_path))
            
            # Extract the required features
            features = extract_network_features(G, df, model_name, run_id.name)
            
            # Append the feature dictionary to our list
            model_features_list.append(features)
        else:
            # Safely write warnings without breaking the progress bar
            tqdm.write(f"Warning: No edgelist.csv found in {run_id.name}")
            
    # Save the accumulated data once the progress bar completes
    if model_features_list and output_path:
        out_model_dir = output_path / model_name
        out_model_dir.mkdir(parents=True, exist_ok=True)
        
        out_df = pd.DataFrame(model_features_list)
        
        output_file = out_model_dir / "network_features.csv"
        out_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(model_features_list)} runs for {model_name} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Extract an edge list and network features given a path to a directory containing fmnt JSON files."
        )
        
    parser.add_argument("--input_dir", required=True, type=str, 
                        help="Path to the directory containing input JSON files.")
    parser.add_argument("--output_dir", type=str,
                        help="Path to directory for saving results. If absent, files will not be saved.")
    
    args = parser.parse_args()

    processed_data_dir = Path(args.input_dir).resolve()
    output_base_dir = Path(args.output_dir).resolve() if args.output_dir else None
    
    logger.info(f"Input directory is: {processed_data_dir}")
    if output_base_dir:
        logger.info(f"Output directory is: {output_base_dir}")
    
    process_all_networks(input_base_dir=processed_data_dir, output_base_dir=output_base_dir)