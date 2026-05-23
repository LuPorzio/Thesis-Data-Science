import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend (no pop-ups, no hanging)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from emoatlas import EmoScores
from scipy import stats
from pathlib import Path

# Load models globally so they don't reload multiple times
try:
    spacy.load("it_core_news_lg")
    spacy.load("en_core_web_lg")
except OSError:
    print("⚠️ Spacy models not found. Please ensure they are downloaded.")


def text_cleaning(df):
    df = df.copy()
    # clean columns that have words inside
    columns_to_clean = ["cue_word", "association_1", "association_2", "association_3"]
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace("à", "a")  # for words like 'università'
                .str.replace(r"[\d\W_]+", " ", regex=True)
                .str.strip()
            )
    return df


def build_association_df(df, filtering="no"):
    # preserve original cue word for later indexing (before being renamed)
    df["cue_word_original"] = df["cue_word"]

    # rename cue and valence to fit wide format
    df_long = df.rename(
        columns={
            "cue_word": "association_0",
            "valence_cue_word": "valence_association_0",
        }
    )

    # transform the df from wide to long format based on their columns and unique identifiers
    df_long = pd.wide_to_long(
        df_long,
        stubnames=["association", "valence_association"],
        i=["run_id", "cue_word_original"],
        j="index",
        sep="_",
        suffix=r"\d+",
    ).reset_index()

    # sort df for the cue_word and the index of the association
    df_long = df_long.sort_values(by=["cue_word_original", "index"]).reset_index(
        drop=True
    )

    # rename columns to be more explicit
    df_long = df_long.rename(
        columns={
            "cue_word_original": "cue_word",
            "association": "word",
            "valence_association": "valence",
        }
    )

    df_long = df_long.dropna()
    df_long = df_long[(df_long["word"] != "nan") & (df_long["word"] != "")]

    ## !!! OPTIONAL: FILTERED !!! filter out words that were given by only one participant
    if filtering == "yes":
        df_long = df_long.groupby(["cue_word", "word"]).filter(
            lambda g: g["run_id"].nunique() > 1
        )

    return df_long


# function to plot valence categories distribution
def plot_valence_distribution(df):
    plt.figure(figsize=(6, 5))

    # count the number of occurrences of each valence category
    counts = df.value_counts().sort_index()
    counts.plot(kind="bar")

    plt.title("Distribuzione Categorie Valenza")
    plt.xlabel("Categoria")
    plt.ylabel("Conteggio")

    plt.tight_layout()
    # plt.show()  <-- REMOVED so it doesn't freeze the script
    plt.close()   # <-- ADDED to clear memory silently


def categorize_valence_single(df):
    # convert to NumPy array
    arr = np.array(df["valence"])

    # compute value range (gamma)
    minv, maxv = arr.min(), arr.max()
    gamma = maxv - minv

    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    mode_val = stats.mode(arr, keepdims=False)

    #  defines classification logic based on position relative to quartiles
    def cat_score(v):
        if abs(Q1 - Q3) == 0:
            if v == mode_val: return 0
            if v < mode_val: return -1
            if v > mode_val: return 1
        elif abs(Q1 - Q3) == 1:
            if v > Q3: return 1
            elif v < Q1: return -1
            else: return 0
        elif abs(Q1 - Q3) > 1:
            if v <= Q1: return -1
            elif v >= Q3: return 1
            else: return 0

    # apply classification to the new column of the df
    df["valence_cat"] = df["valence"].apply(cat_score)
    plot_valence_distribution(df["valence_cat"])

    return df


def build_single_fmnt(
    df=pd.DataFrame(), keywords=list(), folder_path=str, file_path=str, p_id=str
):
    emo = EmoScores(language="italian", spacy_model="it_core_news_lg")
    # initialize an empty undirected graph
    G = nx.Graph()

    # create a valence dictionary for all nodes (cue words + responses)
    valence_dict = dict(zip(df["word"], df["valence_cat"]))

    # create lists for negative, positive and neutral words
    positive = set()
    negative = set()
    neutral = set()

    # add all unique words as nodes with their valence category as a node attribute
    for node, val in valence_dict.items():
        G.add_node(node, valence=val)
        if val == 1: positive.add(node)
        elif val == -1: negative.add(node)
        elif val == 0: neutral.add(node)
        
    custom_valences = [positive, negative, neutral]
    
    # add undirected edges between cue words and their responses (for each row)
    edges = [
        (cue, word) for cue, word in zip(df["cue_word"], df["word"]) if cue != word
    ]

    G.add_edges_from(edges)
    fmn = emo.nxgraph_to_formamentis(G)

    for keyword in keywords:
        try:  # plot bfmn
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.axis("off")
            fmnt_keyword = emo.extract_word_from_formamentis(fmn, keyword)
            emo.draw_formamentis(
                fmn=fmnt_keyword,
                layout="edge_bundling",
                custom_valences=custom_valences,
                ax=ax,
            )
            fig.savefig(
                f"{folder_path}/analysis/bfmn/{file_path}/{p_id}_{keyword}.png",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close(fig) # <-- ADDED to clear memory

        except ValueError as e:
            plt.close(fig) # <-- ADDED to clear memory on error
            if "min() arg is an empty sequence" in str(e):
                print(f"⚠️ Skipping {keyword}: not enough data to build the network.")

    # get the valence, degree and closeness centrality of the words of interest
    features_fmnt_dict = {}
    for word in ["ansia", "matematica"]:
        valence = (
            1 if word in positive
            else -1 if word in negative else 0 if word in neutral else None
        )
        degree = G.degree(word) if word in G else 0
        clo_cen = nx.closeness_centrality(G, word) if word in G else 0

        features_fmnt_dict[f"valence_{word}"] = valence
        features_fmnt_dict[f"degree_{word}"] = degree
        features_fmnt_dict[f"clo_cen_{word}"] = clo_cen

    # wrap in list to create a single-row DataFrame
    features_fmnt = pd.DataFrame([features_fmnt_dict])
    features_fmnt.insert(loc=0, column="run_id", value=df["run_id"].iloc[0])

    return features_fmnt, custom_valences


def categorize_valence_multiple(df):
    all_words = list(w for w in df["word"].unique())
    print(f"Number of unique words: {len(all_words)}")
    valence_map = {}

    for word in all_words:
        keyword_valence = df["valence"][df["word"] == word]
        all_valences = df["valence"][~(df["word"] == word)]

        count = df["word"].str.count(word).sum()
        if count < 3:
            valence_map[word] = 0
        else:
            stat, p_value = stats.kruskal(keyword_valence, all_valences)
            mean_keyword_valence = np.mean(keyword_valence)
            mean_valences = np.mean(all_valences)

            if p_value < 0.1 and mean_keyword_valence < mean_valences:
                valence_map[word] = -1  # negative
            elif p_value < 0.1 and mean_keyword_valence > mean_valences:
                valence_map[word] = 1   # positive
            elif p_value >= 0.1 or (p_value < 0.1 and mean_keyword_valence == mean_valences):
                valence_map[word] = 0   # neutral

    df["valence_cat"] = df["word"].map(valence_map)
    return df


def multiple_fmnt_features(G, fmn, keyword, file_path, percentage):
    local_clustering_coeff = nx.clustering(G, keyword) if keyword else 0

    G_kw = nx.Graph()
    G_kw.add_nodes_from(fmn.vertices)
    G_kw.add_edges_from(fmn.edges)

    # If the graph is disconnected, operate on the largest connected component
    if len(G_kw.nodes) > 0:
        if not nx.is_connected(G_kw):
            components = list(nx.connected_components(G_kw))
            largest_component = max(components, key=len)
            G_kw_sub = G_kw.subgraph(largest_component).copy()
        else:
            G_kw_sub = G_kw
        avg_shortest_path = nx.average_shortest_path_length(G_kw_sub)
        diameter = nx.diameter(G_kw_sub)
    else:
        avg_shortest_path = 0
        diameter = 0

    deg_dict = dict(G_kw.degree(fmn.vertices))
    
    if deg_dict:
        degrees = np.array(list(deg_dict.values()))
        threshold = np.percentile(degrees, percentage)
        hubs = sorted(
            [(f"{n} ({d})", d) for n, d in deg_dict.items() if d >= threshold],
            key=lambda x: x[1],
            reverse=True,
        )
        hubs = [h[0] for h in hubs]
    else:
        hubs = []

    rows = [{
        "Sample": file_path,
        "keyword": keyword,
        "$N_v$": len(fmn.vertices),
        "$N_e$": len(fmn.edges),
        "$C_i$": local_clustering_coeff,
        "$l_G$": avg_shortest_path,
        "$d$": diameter,
        "Hubs": hubs,
    }]

    features_fmnt_df = pd.DataFrame(rows)
    features_fmnt_df["Hubs"] = features_fmnt_df["Hubs"].apply(lambda x: list(x))

    return features_fmnt_df


def build_multiple_fmnt(
    df, keywords, folder_path, file_path, flower="yes", language="ita"
):
    if language == "ita":
        emo = EmoScores(language="italian", spacy_model="it_core_news_lg")
    elif language == "eng":
        emo = EmoScores(language="english", spacy_model="en_core_web_lg")

    G = nx.Graph()
    valence_dict = dict(zip(df["word"], df["valence_cat"]))

    positive_words = set()
    negative_words = set()
    neutral_words = set()

    for node, val in valence_dict.items():
        G.add_node(node, valence=val)
        if val == 1: positive_words.add(node)
        elif val == -1: negative_words.add(node)
        elif val == 0: neutral_words.add(node)

    custom_valences = [positive_words, negative_words, neutral_words]

    edges = [(cue, word) for cue, word in zip(df["cue_word"], df["word"]) if cue != word]
    G.add_edges_from(edges)

    fmn = emo.nxgraph_to_formamentis(G)
    fmnts_keyword = []
    all_features_df = pd.DataFrame()

    features_fmn = multiple_fmnt_features(G, fmn, None, file_path, 99).drop(
        columns=["keyword", "$C_i$"]
    )

    features = []
    for keyword in keywords:
        try:
            # plot bfmn
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.axis("off")
            fmnt_keyword = emo.extract_word_from_formamentis(fmn, keyword)
            emo.draw_formamentis(
                fmn=fmnt_keyword,
                layout="edge_bundling",
                custom_valences=custom_valences,
                ax=ax,
            )
            fmnts_keyword.append(fmnt_keyword)

            # Ensure we use the clean file_path string here to avoid FileExistsError
            output_dir_fmn = Path(folder_path) / "out" / "analysis" / "bfmn" / file_path
            output_dir_fmn.mkdir(parents=True, exist_ok=True)
            
            # Save the network image
            save_path = output_dir_fmn / f"{keyword}.png"
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig) # <-- ADDED to clear memory for the network plot
            print(f"Saved: {save_path}")

            print(f"Local clustering {keyword}: {nx.clustering(G, keyword)}")
            print(f"Degree {keyword}: {G.degree(keyword)}")

            counts = {
                "positive": sum(1 for w in fmnt_keyword.vertices if w in positive_words),
                "negative": sum(1 for w in fmnt_keyword.vertices if w in negative_words),
                "neutral": sum(1 for w in fmnt_keyword.vertices if w in neutral_words),
            }
            print(f"Counts of '{keyword}' = {counts}")
            
            if counts:
                print(f"Mode of '{keyword}' = {max(counts, key=counts.get)}")

            ### !!! OPTIONAL: PLOT EMOTIONAL FLOWER !!!
            if flower == "yes":
                zeta_scores = emo.zscores(fmnt_keyword)
                emo.draw_plutchik(zeta_scores, reject_range=(-1.96, 1.96))

                output_dir = Path(folder_path) / "out" / "analysis" / "emo_flowers_bfmn" / file_path
                output_dir.mkdir(parents=True, exist_ok=True)
                
                flower_save_path = output_dir / f"{keyword}.png"
                plt.savefig(flower_save_path, bbox_inches="tight", pad_inches=0.1)
                plt.close("all") # <-- Keeps the flower memory clean

            features_fmnt_keyword = multiple_fmnt_features(
                G, fmnt_keyword, keyword, file_path, 95
            )
            features.append(features_fmnt_keyword)
            all_features_df = pd.concat(features, ignore_index=True).drop(
                columns=["$d$"]
            )

        except ValueError as e:
            plt.close("all") # <-- Cleans up if an error occurs mid-plot
            print(f"⚠️ Skipping {keyword}:", e)

    return fmnts_keyword, all_features_df, features_fmn


def pipeline_multiple_participants(
    df, keywords, folder_path, file_path, flower, filtering="no"
):
    clean_df = text_cleaning(df)
    long_df = build_association_df(clean_df, filtering=filtering)
    final_df = categorize_valence_multiple(long_df)
    G, fmn, features_fmnt_df = build_multiple_fmnt(
        final_df, keywords, folder_path, file_path, flower
    )
    return final_df, G, fmn, features_fmnt_df


def pipeline_df_building(df, filter="no"):
    clean_df = text_cleaning(df)
    long_df = build_association_df(clean_df, filter)
    final_df = categorize_valence_multiple(long_df)
    return final_df


def pipeline_single_participant(df, keywords, folder_path, file_path, p_id):
    clean_df = text_cleaning(df)
    long_df = build_association_df(clean_df)
    final_df = categorize_valence_single(long_df)
    features_fmnt, valences = build_single_fmnt(
        final_df, keywords, folder_path, file_path, p_id
    )
    return final_df, features_fmnt, valences


def run_pipeline_all_participants(df):
    results = []
    groups_total = df.groupby("run_id")

    for run_id, participant_df in groups_total:
        final_df, G, features_fmnt_df = pipeline_single_participant(participant_df)
        results.append(features_fmnt_df)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    wide_format_dir = Path("src/data_validation/NEW_final_wide_format_dataset").resolve().absolute()

    for dir in wide_format_dir.iterdir():
        if dir.is_dir() and dir.name != "out": # skip the out folder itself to avoid looping over generated data
            for file in dir.iterdir():
                if file.suffix == ".csv": # ensure we only process CSVs
                    print(f"\nProcessing file: {file.name}")
                    df = pd.read_csv(file)
                    folder_path = wide_format_dir
                    
                    # USE THE STEM (the name without .csv) to prevent FileExistsError
                    file_name_clean = file.stem 
                    
                    pipeline_multiple_participants(
                        df, ["mathematics","math", "science", "physics", "mathematic"], 
                        folder_path, file_name_clean, 
                        flower="no", filtering="no"
                    )