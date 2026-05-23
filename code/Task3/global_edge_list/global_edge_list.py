import pandas as pd
from pathlib import Path

def aggregate_edgelists(input_directory, output_directory):
    """
    Iterates through model folders, combines run_id edgelists, 
    and saves them into a designated output directory with model subfolders.
    """
    input_dir = Path(input_directory).resolve().absolute()
    output_dir = Path(output_directory).resolve().absolute()

    # 1. Iterate through all folders in the input directory (anita, claude, etc.)
    for model_folder in input_dir.iterdir():
        
        # Check if it's a directory AND ignore the output folder if it's in the same place
        if model_folder.is_dir() and model_folder.name != output_dir.name:
            print(f"Processing model: {model_folder.name}...")
            
            all_edges_for_model = []
            
            # 2. Iterate through run_id folders (the long string names)
            for run_folder in model_folder.iterdir():
                if run_folder.is_dir():
                    edgelist_path = run_folder / "edgelist.csv"
                    
                    if edgelist_path.exists():
                        try:
                            df = pd.read_csv(edgelist_path)
                            df['run_id'] = run_folder.name 
                            all_edges_for_model.append(df)
                        except Exception as e:
                            print(f"  [!] Error reading {edgelist_path}: {e}")
            
            # 3. Combine, aggregate, and save
            if all_edges_for_model:
                combined_df = pd.concat(all_edges_for_model, ignore_index=True)
                
                # Calculate the weight (frequency) of each edge across all runs
                aggregated_df = combined_df.groupby(
                    ['cue_word', 'association_word', 'cue_valence', 'associated_valence']
                ).size().reset_index(name='weight')
                
                # --- SAVING LOGIC ---
                # Create the specific folder for this model inside the new global folder
                model_output_folder = output_dir / model_folder.name 
                model_output_folder.mkdir(parents=True, exist_ok=True) 
                
                # Define the path and save
                output_path = model_output_folder / f"global_edgelist_{model_folder.name}.csv"
                aggregated_df.to_csv(output_path, index=False)
                
                print(f"  -> Success! Saved to {output_path.name} with {len(aggregated_df)} unique edges.\n")
            else:
                print(f"  -> No edgelist.csv files found for {model_folder.name}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a global edge list for each LLM.")
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory containing models with associated run ids as sub folders each containing a file called `edgelist.csv`")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory where global edge list will be saved after sub folder with name of the model is automatically created.")

    args = parser.parse_args()
    # Pointing to the root of the test run folder
    # INPUT_DIRECTORY = "./src/data_validation/final_edge_list_individual" 
    
    # Pointing to the new destination folder
    # OUTPUT_DIRECTORY = "./src/data_validation/final_edge_list_global"
    
    aggregate_edgelists(args.input_dir, args.output_dir)
    print("All models processed.")