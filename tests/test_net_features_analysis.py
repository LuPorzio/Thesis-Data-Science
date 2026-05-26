from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK3_FEATURES_DIR = PROJECT_ROOT / "code" / "Task3" / "extract_features"
if str(TASK3_FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_FEATURES_DIR))

from net_features_analysis_utils import (  # noqa: E402
    build_weighted_graph,
    closeness_long_frame,
    closeness_model_summary,
    degree_hubs_long_frame,
    degree_hubs_model_summary,
    hub_context_subgraph,
    hits_long_frame,
    hits_model_summary,
    load_network_feature_files,
    load_global_edge_list,
    model_run_closeness_summary,
    model_top_terms_table,
    parse_network_feature_columns,
    pivot_model_term_matrix,
    top_terms,
    top_terms_overlap_jaccard,
)


class TestNetFeaturesAnalysis(unittest.TestCase):
    def test_parse_and_long_frames(self):
        df = pd.DataFrame(
            {
                "model_name": ["MANX_LLM_MistralSmall4"],
                "model_display": ["Mistral Small 4"],
                "run_id": ["run1"],
                "closeness_centralities": ["{'a': 0.2, 'b': 0.5}"],
                "hubs_scores_hits": ["{'a': 0.1, 'b': 0.7}"],
                "degree_hubs": ["['b']"],
            }
        )

        parsed = parse_network_feature_columns(df)
        self.assertEqual(parsed.loc[0, "degree_hubs"], ["b"])

        closeness_long = closeness_long_frame(parsed)
        hits_long = hits_long_frame(parsed)
        degree_long = degree_hubs_long_frame(parsed)

        self.assertEqual(closeness_long.shape[0], 2)
        self.assertEqual(hits_long.shape[0], 2)
        self.assertEqual(degree_long.shape[0], 1)

    def test_model_aggregation_helpers(self):
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M1", "M2"],
                "model_display": ["Model 1", "Model 1", "Model 2"],
                "run_id": ["r1", "r2", "r3"],
                "closeness_centralities": [{"a": 0.2, "b": 0.5}, {"a": 0.4, "c": 0.8}, {"a": 0.1, "b": 0.3}],
                "hubs_scores_hits": [{"a": 0.1, "b": 0.7}, {"a": 0.2, "c": 0.9}, {"a": 0.4, "b": 0.1}],
                "degree_hubs": [["b"], ["c"], ["a", "b"]],
            }
        )

        closeness_long = closeness_long_frame(df)
        hits_long = hits_long_frame(df)
        degree_long = degree_hubs_long_frame(df)

        closeness_summary = closeness_model_summary(closeness_long)
        hits_summary = hits_model_summary(hits_long)
        degree_summary = degree_hubs_model_summary(degree_long)
        run_summary = model_run_closeness_summary(closeness_long)

        self.assertIn("mean_closeness", closeness_summary.columns)
        self.assertIn("mean_hub_score", hits_summary.columns)
        self.assertIn("hub_frequency", degree_summary.columns)
        self.assertEqual(run_summary.shape[0], 3)

        terms = top_terms(closeness_summary, "mean_closeness", top_n=1)
        matrix = pivot_model_term_matrix(closeness_summary, "mean_closeness", terms)
        self.assertEqual(matrix.shape[1], 1)

        top_table = model_top_terms_table(closeness_summary, "mean_closeness", top_n=1)
        self.assertEqual(top_table.shape[0], 2)

        overlap = top_terms_overlap_jaccard(closeness_long, degree_long, "closeness", "present", top_n=1)
        self.assertIn("jaccard_overlap", overlap.columns)

    def test_load_network_feature_files_discovers_csvs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "MANX_LLM_MistralSmall4" / "MANX_LLM_MistralSmall4"
            model_dir.mkdir(parents=True)
            csv_path = model_dir / "network_features.csv"
            pd.DataFrame(
                {
                    "model_name": ["MANX_LLM_MistralSmall4"],
                    "run_id": ["r1"],
                    "closeness_centralities": ["{'a': 0.2}"],
                    "hubs_scores_hits": ["{'a': 0.1}"],
                    "degree_hubs": ["['a']"],
                }
            ).to_csv(csv_path, index=False)

            loaded = load_network_feature_files(root)
            self.assertEqual(loaded.shape[0], 1)
            self.assertIn("model_display", loaded.columns)

    def test_load_network_feature_files_real_repo_layout(self):
        root = PROJECT_ROOT / "code" / "Task3" / "extract_features" / "net_features"
        loaded = load_network_feature_files(root)
        self.assertEqual(loaded["model_name"].nunique(), 14)
        self.assertEqual(loaded["model_display"].nunique(), 14)
        self.assertTrue((loaded["source_path"].str.contains("/" + "network_features.csv")).all())

    def test_build_graph_and_hub_subgraph(self):
        edge_df = pd.DataFrame(
            {
                "cue_word": ["a", "a", "b"],
                "association_word": ["b", "c", "d"],
                "weight": [1, 2, 3],
            }
        )
        G = build_weighted_graph(edge_df)
        subgraph = hub_context_subgraph(G, ["a"], radius=1)
        self.assertTrue(G.has_edge("a", "b"))
        self.assertTrue(G.has_edge("a", "c"))
        self.assertGreaterEqual(len(subgraph), 3)

    def test_load_global_edge_list_reads_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "MANX_LLM_MistralSmall4"
            model_dir.mkdir(parents=True)
            csv_path = model_dir / "global_edgelist_MANX_LLM_MistralSmall4.csv"
            pd.DataFrame(
                {
                    "cue_word": ["a"],
                    "association_word": ["b"],
                    "cue_valence": [0],
                    "associated_valence": [1],
                    "weight": [2],
                }
            ).to_csv(csv_path, index=False)

            loaded = load_global_edge_list("MANX_LLM_MistralSmall4", root)
            self.assertEqual(loaded.shape[0], 1)

    def test_load_global_edge_list_real_repo_layout(self):
        root = PROJECT_ROOT / "code" / "Task3" / "global_edge_list" / "NEW_edge_list_global"
        network_root = PROJECT_ROOT / "code" / "Task3" / "extract_features" / "net_features"
        loaded_network = load_network_feature_files(network_root)
        model_name = loaded_network.iloc[0]["model_name"]
        loaded = load_global_edge_list(model_name, root)
        self.assertGreater(loaded.shape[0], 0)
        self.assertIn("cue_word", loaded.columns)


if __name__ == "__main__":
    unittest.main()
