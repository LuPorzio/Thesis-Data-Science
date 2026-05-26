from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DISPLAY_NAME_MAPPING = {
    "MANX_LLM_anitamistral": "Anita 24B (Uncensored)",
    "MANX_LLM_DeepSeekLarge": "DeepSeek Chat",
    "MANX_LLM_granite4h": "Granite 4 Tiny",
    "MANX_LLM_Grok41FastReasoning": "Grok 4.1 Fast (Reasoning)",
    "MANX_LLM_magistralsmall": "Magistral Small",
    "MANX_LLM_ministral3b": "Ministral 3B",
    "MANX_LLM_ministral14b": "Ministral 14B (Reasoning)",
    "MANX_LLM_mistralsmall": "Mistral Small 3.2",
    "MANX_LLM_MistralSmall4": "Mistral Small 4",
    "MANX_LLM_phi4reasoning": "Phi-4 (Reasoning+)",
    "MANX_LLM_qwen4bthink": "Qwen3 4B (Thinking)",
    "MANX_LLM_qwen4bunce": "Qwen3 4B (Uncensored)",
    "MANX_LLM_qwen34binstruct": "Qwen3 4B",
    "MANX_LLM_qwen35_9b": "Qwen3.5 9B",
}


def load_network_feature_files(base_dir: str | Path) -> pd.DataFrame:
    base_path = Path(base_dir)
    csv_paths = sorted(base_path.rglob("network_features.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No network_features.csv files found under {base_path}")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["source_path"] = str(csv_path)
        frame["model_display"] = frame["model_name"].map(MODEL_DISPLAY_NAME_MAPPING).fillna(frame["model_name"])
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["model_display", "run_id"]).reset_index(drop=True)
    return combined


def load_global_edge_list(
    model_name: str, 
    base_dir: str | Path = Path("code/Task3/global_edge_list/NEW_edge_list_global").resolve().absolute()
    ) -> pd.DataFrame:

    path = Path(base_dir) / model_name / f"global_edgelist_{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Global edge list not found for {model_name}: {path}")
    return pd.read_csv(path)


@dataclass
class WeightedGraph:
    adjacency: dict[str, dict[str, float]]

    @property
    def nodes(self) -> set[str]:
        return set(self.adjacency.keys())

    @property
    def edges(self) -> list[tuple[str, str, float]]:
        seen = set()
        edges = []
        for left, neighbors in self.adjacency.items():
            for right, weight in neighbors.items():
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((left, right, weight))
        return edges

    def __len__(self) -> int:
        return len(self.adjacency)

    def __contains__(self, node: str) -> bool:
        return node in self.adjacency

    def has_edge(self, left: str, right: str) -> bool:
        return right in self.adjacency.get(left, {})

    def degree(self, node: str) -> int:
        return len(self.adjacency.get(node, {}))

    def neighbors(self, node: str) -> set[str]:
        return set(self.adjacency.get(node, {}).keys())


def build_weighted_graph(edge_df: pd.DataFrame) -> WeightedGraph:
    required = {"cue_word", "association_word", "weight"}
    missing = required.difference(edge_df.columns)
    if missing:
        raise KeyError(f"Missing required edge columns: {sorted(missing)}")

    adjacency: dict[str, dict[str, float]] = {}
    for row in edge_df[["cue_word", "association_word", "weight"]].itertuples(index=False):
        cue = str(row.cue_word)
        association = str(row.association_word)
        weight = float(row.weight)
        if cue == association:
            continue
        adjacency.setdefault(cue, {})
        adjacency.setdefault(association, {})
        adjacency[cue][association] = adjacency[cue].get(association, 0.0) + weight
        adjacency[association][cue] = adjacency[association].get(cue, 0.0) + weight
    return WeightedGraph(adjacency=adjacency)


def hub_context_subgraph(G: WeightedGraph, hub_terms: list[str], radius: int = 1) -> WeightedGraph:
    nodes = set()
    for term in hub_terms:
        if term in G:
            frontier = {term}
            nodes.add(term)
            for _ in range(radius):
                next_frontier = set()
                for node in frontier:
                    next_frontier.update(G.neighbors(node))
                next_frontier -= nodes
                nodes.update(next_frontier)
                frontier = next_frontier
    if not nodes:
        return WeightedGraph(adjacency={})

    adjacency: dict[str, dict[str, float]] = {}
    for left in nodes:
        for right, weight in G.adjacency.get(left, {}).items():
            if right in nodes:
                adjacency.setdefault(left, {})
                adjacency.setdefault(right, {})
                adjacency[left][right] = weight
                adjacency[right][left] = weight
    return WeightedGraph(adjacency=adjacency)


def _parse_value(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {} if not isinstance(value, list) else []
    if isinstance(value, str):
        value = value.strip()
        if not value or value == "None":
            return {}
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            cleaned = re.sub(r"\bnan\b", "None", value, flags=re.IGNORECASE)
            cleaned = re.sub(r"\binf\b", "None", cleaned, flags=re.IGNORECASE)
            return ast.literal_eval(cleaned)
    return value


def _parse_dict_value(value):
    parsed = _parse_value(value)
    return parsed if isinstance(parsed, dict) else {}


def _parse_list_value(value):
    parsed = _parse_value(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return list(parsed.keys())
    return []


def parse_network_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["closeness_centralities"] = frame["closeness_centralities"].apply(_parse_dict_value)
    frame["hubs_scores_hits"] = frame["hubs_scores_hits"].apply(_parse_dict_value)
    frame["degree_hubs"] = frame["degree_hubs"].apply(_parse_list_value)
    return frame


def closeness_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in df[["model_name", "model_display", "run_id", "closeness_centralities"]].to_dict("records"):
        for term, value in record["closeness_centralities"].items():
            rows.append(
                {
                    "model_name": record["model_name"],
                    "model_display": record["model_display"],
                    "run_id": record["run_id"],
                    "term": str(term),
                    "closeness": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model_display"] = frame["model_display"].astype("category")
        frame["model_name"] = frame["model_name"].astype("category")
        frame["run_id"] = frame["run_id"].astype("category")
        frame["term"] = frame["term"].astype("category")
    return frame


def hits_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in df[["model_name", "model_display", "run_id", "hubs_scores_hits"]].to_dict("records"):
        for term, value in record["hubs_scores_hits"].items():
            rows.append(
                {
                    "model_name": record["model_name"],
                    "model_display": record["model_display"],
                    "run_id": record["run_id"],
                    "term": str(term),
                    "hub_score": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model_display"] = frame["model_display"].astype("category")
        frame["model_name"] = frame["model_name"].astype("category")
        frame["run_id"] = frame["run_id"].astype("category")
        frame["term"] = frame["term"].astype("category")
    return frame


def degree_hubs_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in df[["model_name", "model_display", "run_id", "degree_hubs"]].to_dict("records"):
        for term in record["degree_hubs"]:
            rows.append(
                {
                    "model_name": record["model_name"],
                    "model_display": record["model_display"],
                    "run_id": record["run_id"],
                    "term": str(term),
                    "present": 1,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model_display"] = frame["model_display"].astype("category")
        frame["model_name"] = frame["model_name"].astype("category")
        frame["run_id"] = frame["run_id"].astype("category")
        frame["term"] = frame["term"].astype("category")
    return frame


def _model_term_summary(frame: pd.DataFrame, value_col: str, agg_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["model_display", "term", agg_name, "median_value", "std_value", "n_runs"])

    summary = (
        frame.groupby(["model_display", "term"], as_index=False)
        .agg(
            mean_value=(value_col, "mean"),
            median_value=(value_col, "median"),
            std_value=(value_col, "std"),
            n_runs=("run_id", "nunique"),
        )
        .rename(columns={"mean_value": agg_name})
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    return summary


def closeness_model_summary(closeness_long: pd.DataFrame) -> pd.DataFrame:
    return _model_term_summary(closeness_long, "closeness", "mean_closeness")


def hits_model_summary(hits_long: pd.DataFrame) -> pd.DataFrame:
    return _model_term_summary(hits_long, "hub_score", "mean_hub_score")


def degree_hubs_model_summary(degree_hubs_long: pd.DataFrame) -> pd.DataFrame:
    if degree_hubs_long.empty:
        return pd.DataFrame(columns=["model_display", "term", "hub_frequency", "n_runs", "hub_rate"])

    summary = (
        degree_hubs_long.groupby(["model_display", "term"], as_index=False)
        .agg(hub_frequency=("present", "sum"), n_runs=("run_id", "nunique"))
        .assign(hub_rate=lambda x: x["hub_frequency"] / x["n_runs"])
    )
    return summary


def model_run_closeness_summary(closeness_long: pd.DataFrame) -> pd.DataFrame:
    return (
        closeness_long.groupby(["model_display", "run_id"], as_index=False)
        .agg(mean_closeness=("closeness", "mean"), median_closeness=("closeness", "median"))
        .sort_values(["model_display", "run_id"])
    )


def top_terms(frame: pd.DataFrame, value_col: str, top_n: int = 15) -> list[str]:
    if frame.empty:
        return []
    return (
        frame.groupby("term", as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
        .head(top_n)["term"]
        .tolist()
    )


def pivot_model_term_matrix(frame: pd.DataFrame, value_col: str, terms: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    matrix = frame[frame["term"].isin(terms)].pivot_table(
        index="model_display", columns="term", values=value_col, aggfunc="mean"
    )
    return matrix.reindex(sorted(matrix.index))


def model_top_terms_table(frame: pd.DataFrame, value_col: str, top_n: int = 5) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["model_display", "term", value_col])
    return (
        frame.groupby(["model_display", "term"], as_index=False)[value_col]
        .mean()
        .sort_values(["model_display", value_col], ascending=[True, False])
        .groupby("model_display")
        .head(top_n)
        .reset_index(drop=True)
    )


def top_terms_overlap_jaccard(left_frame: pd.DataFrame, right_frame: pd.DataFrame, value_col_left: str, value_col_right: str, top_n: int = 10) -> pd.DataFrame:
    left_top = (
        left_frame.groupby(["model_display", "term"], as_index=False)[value_col_left]
        .mean()
        .sort_values(["model_display", value_col_left], ascending=[True, False])
        .groupby("model_display")
        .head(top_n)
    )
    right_top = (
        right_frame.groupby(["model_display", "term"], as_index=False)[value_col_right]
        .sum()
        .sort_values(["model_display", value_col_right], ascending=[True, False])
        .groupby("model_display")
        .head(top_n)
    )

    models = sorted(set(left_top["model_display"]).union(right_top["model_display"]))
    rows = []
    for model in models:
        left_terms = set(left_top.loc[left_top["model_display"] == model, "term"])
        right_terms = set(right_top.loc[right_top["model_display"] == model, "term"])
        union = left_terms | right_terms
        rows.append(
            {
                "model_display": model,
                "jaccard_overlap": len(left_terms & right_terms) / len(union) if union else 0.0,
                "left_top_terms": sorted(left_terms),
                "right_top_terms": sorted(right_terms),
            }
        )
    return pd.DataFrame(rows)
