import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import seaborn as sns
    from pathlib import Path

    from net_features_analysis_utils import (
        WeightedGraph,
        build_weighted_graph,
        closeness_long_frame,
        closeness_model_summary,
        degree_hubs_long_frame,
        degree_hubs_model_summary,
        hub_context_subgraph,
        hits_long_frame,
        hits_model_summary,
        load_global_edge_list,
        load_network_feature_files,
        model_run_closeness_summary,
        model_top_terms_table,
        parse_network_feature_columns,
        pivot_model_term_matrix,
        top_terms,
        top_terms_overlap_jaccard,
    )

    sns.set_theme(style="whitegrid", context="talk")
    return (
        Path,
        WeightedGraph,
        build_weighted_graph,
        closeness_long_frame,
        closeness_model_summary,
        degree_hubs_long_frame,
        degree_hubs_model_summary,
        hits_long_frame,
        hits_model_summary,
        hub_context_subgraph,
        load_global_edge_list,
        load_network_feature_files,
        model_run_closeness_summary,
        model_top_terms_table,
        nx,
        parse_network_feature_columns,
        pivot_model_term_matrix,
        plt,
        sns,
        top_terms,
        top_terms_overlap_jaccard,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model-level BFMN feature comparison

    This notebook compares all 14 models using:
    - closeness centrality
    - HITS hub scores
    - degree-based hubs

    It also includes an interactive hub-focused BFMN subgraph for one selected model.
    """)
    return


@app.cell
def _(Path, load_network_feature_files, parse_network_feature_columns):
    base_dir = Path("code/Task3/extract_features/net_features").resolve()
    df_raw = load_network_feature_files(base_dir)
    df = parse_network_feature_columns(df_raw)
    return (df,)


@app.cell
def _(df, mo):
    model_options = sorted(df["model_display"].unique())
    model_picker = mo.ui.dropdown(options=model_options, value=model_options[0], label="Select model")
    model_picker
    return (model_picker,)


@app.cell
def _(mo):
    layout_picker = mo.ui.dropdown(
        options=["concentric", "force-directed"],
        value="concentric",
        label="Layout",
    )
    radius_slider = mo.ui.slider(start=1, stop=2, step=1, value=1, label="Hub context radius")
    min_neighbor_degree = mo.ui.slider(
        start=1, stop=10, step=1, value=2, label="Min hub connections for non-hub nodes"
    )
    mo.vstack(
        [
            mo.hstack([layout_picker, radius_slider], justify="space-around"),
            min_neighbor_degree,
        ]
    )
    return layout_picker, min_neighbor_degree, radius_slider


@app.cell
def _(closeness_long_frame, degree_hubs_long_frame, df, hits_long_frame):
    closeness_long = closeness_long_frame(df)
    hits_long = hits_long_frame(df)
    degree_hubs_long = degree_hubs_long_frame(df)
    return closeness_long, degree_hubs_long, hits_long


@app.cell
def _(
    closeness_long,
    closeness_model_summary,
    degree_hubs_long,
    degree_hubs_model_summary,
    hits_long,
    hits_model_summary,
    model_run_closeness_summary,
):
    closeness_summary = closeness_model_summary(closeness_long)
    hits_summary = hits_model_summary(hits_long)
    degree_summary = degree_hubs_model_summary(degree_hubs_long)
    run_closeness_summary = model_run_closeness_summary(closeness_long)
    return (
        closeness_summary,
        degree_summary,
        hits_summary,
        run_closeness_summary,
    )


@app.cell
def _(closeness_summary, degree_summary, hits_summary, mo):
    mo.md(f"""
    ## Summary tables

    ### Closeness centrality
    `{closeness_summary.head(12).to_string(index=False)}`

    ### HITS hub scores
    `{hits_summary.head(12).to_string(index=False)}`

    ### Degree hubs
    `{degree_summary.head(12).to_string(index=False)}`
    """)
    return


@app.cell
def _(closeness_summary, pivot_model_term_matrix, plt, sns, top_terms):
    closeness_terms = top_terms(closeness_summary, "mean_closeness", top_n=15)
    closeness_matrix = pivot_model_term_matrix(closeness_summary, "mean_closeness", closeness_terms)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(closeness_matrix, cmap="viridis", linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("Mean closeness centrality for top terms")
    ax.set_xlabel("Term")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(fname="code/figures/Mean_closeness_centrality")
    fig
    return


@app.cell
def _(plt, run_closeness_summary, sns):
    _fig, _ax = plt.subplots(figsize=(14, 6))

    _order = run_closeness_summary.groupby("model_display")["mean_closeness"].median().sort_values().index

    sns.boxplot(
        data=run_closeness_summary, 
        x="model_display", 
        y="mean_closeness", 
        order=_order, 
        ax=_ax, 
        color="#7aa6c2"
    )

    sns.stripplot(
        data=run_closeness_summary, 
        x="model_display", 
        y="mean_closeness", 
        order=_order, 
        ax=_ax, 
        color="black", 
        size=3, 
        alpha=0.45
    )

    _ax.set_title("Run-level mean closeness by model")
    _ax.set_xlabel("Model")
    _ax.set_ylabel("Mean closeness across nodes")
    _ax.tick_params(axis="x", rotation=45)


    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Closeness Centrality — Detailed View
    Hierarchically clustered heatmap, per-model lollipop chart, and multi-panel summary for the closeness centrality feature.
    """)
    return


@app.cell
def _(closeness_summary, pivot_model_term_matrix, plt, sns):
    _per_model_top = (
        closeness_summary.sort_values(["model_display", "mean_closeness"], ascending=[True, False])
        .groupby("model_display")
        .head(15)
        .reset_index(drop=True)
    )
    _term_model_count = _per_model_top.groupby("term")["model_display"].nunique()
    _heatmap_terms = sorted(
        t for t in _per_model_top["term"].unique()
        if _term_model_count[t] >= 2
    )

    if len(_heatmap_terms) < 3:
        _fig, _ax = plt.subplots(figsize=(10, 4))
        _ax.text(0.5, 0.5, "Fewer than 3 shared closeness terms across models", ha="center", va="center")
        _out = _fig
    else:
        _matrix = pivot_model_term_matrix(closeness_summary, "mean_closeness", _heatmap_terms).fillna(0)
        _g = sns.clustermap(
            _matrix,
            cmap="viridis",
            method="average",
            linewidths=0.5,
            linecolor="white",
            figsize=(max(10, len(_heatmap_terms) * 0.35 + 4), 9),
            dendrogram_ratio=(0.1, 0.2),
            # Moved to the right side of the figure (x=1.02)
            cbar_pos=(1.02, 0.2, 0.03, 0.5), 
            cbar_kws={"label": "Mean closeness"},
        )
        _g.fig.suptitle(
            f"Closeness centrality — shared per-model top-15 terms ({len(_heatmap_terms)} terms)",
            fontsize=18, y=1.02,
        )
        # Added bbox_inches="tight" so the new colorbar position isn't cut off
        _g.savefig("code/figures/All_models_closeness_heatmap", bbox_inches="tight")
        _out = _g.fig

    _out
    return


@app.cell
def _(closeness_summary, mo, model_picker, plt, sns):
    _selected_model = model_picker.value

    _model_data = (
        closeness_summary[closeness_summary["model_display"] == _selected_model]
        .nlargest(15, "mean_closeness")
    )

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _ax.hlines(
        y=range(len(_model_data)), xmin=0, xmax=_model_data["mean_closeness"],
        color="#6a9fb5", linewidth=3,
    )
    _ax.plot(
        _model_data["mean_closeness"], range(len(_model_data)),
        "o", markersize=10, color="#1f497d", alpha=0.8,
    )
    _ax.set_yticks(range(len(_model_data)))
    _ax.set_yticklabels(_model_data["term"], fontsize=12)
    _ax.set_xlabel("Mean closeness centrality", fontsize=12, fontweight="bold")
    _ax.set_title(f"Top 15 closeness terms — {_selected_model}", fontsize=14, fontweight="bold")
    _ax.invert_yaxis()
    sns.despine(left=True, bottom=True, ax=_ax)
    _ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"code/figures/Lollipop_Closeness_{_selected_model.replace(' ', '_')}")

    _table_data = (
        closeness_summary[closeness_summary["model_display"] == _selected_model]
        .nlargest(10, "mean_closeness")
        [["term", "mean_closeness", "median_value", "std_value", "n_runs"]]
        .round(5)
    )
    _table_data.columns = ["Term", "Mean", "Median", "Std", "Runs"]
    _table = mo.ui.table(_table_data, label=f"Top 10 closeness terms — {_selected_model}")

    mo.vstack([_fig, _table])
    return


@app.cell
def _(closeness_summary, pivot_model_term_matrix, plt, sns, top_terms):
    _model_means = (
        closeness_summary.groupby("model_display")
        .agg(mean_closeness=("mean_closeness", "mean"), std_closeness=("std_value", "mean"))
        .sort_values("mean_closeness")
        .reset_index()
    )

    _top15 = top_terms(closeness_summary, "mean_closeness", top_n=15)
    _matrix = pivot_model_term_matrix(closeness_summary, "mean_closeness", _top15)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.5]})

    _colors = plt.cm.viridis(
        (_model_means["mean_closeness"] - _model_means["mean_closeness"].min())
        / (_model_means["mean_closeness"].max() - _model_means["mean_closeness"].min() + 1e-10)
    )
    _ax1.barh(range(len(_model_means)), _model_means["mean_closeness"], color=_colors, edgecolor="white")
    _ax1.set_yticks(range(len(_model_means)))
    _ax1.set_yticklabels(_model_means["model_display"], fontsize=9)
    _ax1.set_xlabel("Mean closeness (avg across terms)")
    _ax1.set_title("Model-level mean closeness", fontsize=12)
    _ax1.invert_yaxis()

    sns.heatmap(
        _matrix, cmap="viridis", linewidths=0.3, linecolor="white",
        ax=_ax2, cbar_kws={"label": "Mean closeness"},
    )
    _ax2.set_title("Global top-15 closeness terms", fontsize=12)
    _ax2.set_ylabel("")
    _ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("code/figures/Closeness_multi_panel_summary")

    _fig
    return


@app.cell
def _(hits_summary, pivot_model_term_matrix, plt, sns, top_terms):
    hits_terms = top_terms(hits_summary, "mean_hub_score", top_n=15)
    _hub_matrix = pivot_model_term_matrix(hits_summary, "mean_hub_score", hits_terms)

    _fig, _ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        _hub_matrix, 
        cmap="viridis", 
        linewidths=0.3, 
        linecolor="white", 
        ax=_ax
    )

    _ax.set_title("Mean HITS hub scores for top terms")
    _ax.set_xlabel("Term")
    _ax.set_ylabel("Model")

    plt.savefig(fname="code/figures/Mean_HITS_HUBS_scores")
    plt.tight_layout()

    _fig
    return (hits_terms,)


@app.cell
def _(degree_summary, pivot_model_term_matrix, plt, sns, top_terms):
    degree_terms = top_terms(degree_summary, "hub_frequency", top_n=15)
    _degree_matrix = pivot_model_term_matrix(degree_summary, "hub_frequency", degree_terms)

    _fig, _ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        _degree_matrix, 
        cmap="crest", 
        linewidths=0.3, 
        linecolor="white", 
        ax=_ax
    )

    _ax.set_title("Degree-hub frequency across models", fontsize=23, y=1.02,)
    _ax.set_xlabel("Term")
    _ax.set_ylabel("Model")

    plt.savefig(fname="code/figures/Degree_HUB_frequency")
    plt.tight_layout()

    _fig
    return (degree_terms,)


@app.cell
def _(degree_hubs_long, hits_long, plt, sns, top_terms_overlap_jaccard):
    _overlap = top_terms_overlap_jaccard(hits_long, degree_hubs_long, "hub_score", "present", top_n=10)

    _fig, _ax = plt.subplots(figsize=(14, 5))

    sns.barplot(
        data=_overlap.sort_values("jaccard_overlap"), 
        x="model_display", 
        y="jaccard_overlap", 
        ax=_ax, 
        color="#6a9fb5"
    )

    _ax.set_title("Overlap between top HITS hubs and top degree hubs")
    _ax.set_xlabel("Model")
    _ax.set_ylabel("Jaccard overlap")
    _ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    _fig
    return


@app.cell
def _(
    closeness_summary,
    degree_summary,
    hits_summary,
    mo,
    model_top_terms_table,
):
    closeness_top = model_top_terms_table(closeness_summary, "mean_closeness", top_n=3)
    hits_top = model_top_terms_table(hits_summary, "mean_hub_score", top_n=3)
    degree_top = model_top_terms_table(degree_summary, "hub_frequency", top_n=3)

    mo.vstack(
        [
            mo.md("## Top terms per model"),
            mo.md("### Closeness"),
            mo.ui.table(closeness_top),
            mo.md("### HITS hubs"),
            mo.ui.table(hits_top),
            mo.md("### Degree hubs"),
            mo.ui.table(degree_top),
        ]
    )
    return (closeness_top,)


@app.cell
def _(closeness_top, mo):
    mo.md(f"""
    ### Closeness Table (LaTeX)
    ```latex
    {closeness_top.to_latex(index=False)}
    ```
    """)
    return


@app.cell
def _(
    Path,
    WeightedGraph,
    build_weighted_graph,
    degree_terms,
    df,
    hits_summary,
    hits_terms,
    hub_context_subgraph,
    layout_picker,
    load_global_edge_list,
    min_neighbor_degree,
    model_picker,
    nx,
    plt,
    radius_slider,
):
    import numpy as np
    import matplotlib.patches as mpatches

    _selected_model = model_picker.value
    _model_name = df.loc[df["model_display"] == _selected_model, "model_name"].iloc[0]
    _edge_base = Path("code/Task3/global_edge_list/NEW_edge_list_global").resolve()
    _edge_df = load_global_edge_list(_model_name, _edge_base)
    _graph = build_weighted_graph(_edge_df)
    _hub_terms = sorted(set(hits_terms).union(degree_terms))
    _layout_mode = layout_picker.value
    _radius = radius_slider.value
    _min_deg = min_neighbor_degree.value
    _raw_subgraph = hub_context_subgraph(_graph, _hub_terms, radius=_radius)

    if len(_raw_subgraph) == 0:
        _fig, _ax = plt.subplots(figsize=(12, 12))
        _ax.text(0.5, 0.5, "No hub subgraph available", ha="center", va="center")
        _ax.set_title(f"BFMN hub view: {_selected_model}")
        _ax.axis("off")
        _fig

    _hub_nodes = {node for node in _hub_terms if node in _raw_subgraph.nodes}
    _filtered_nodes = set(_hub_nodes) | {
        node
        for node in _raw_subgraph.nodes
        if node not in _hub_nodes and _raw_subgraph.degree(node) >= _min_deg
    }

    _adj = _raw_subgraph.adjacency
    _filtered_adj = {}
    for _node in _filtered_nodes:
        _neighbors = {}
        for _nbr, _w in _adj.get(_node, {}).items():
            if _nbr in _filtered_nodes:
                _neighbors[_nbr] = _w
        if _neighbors:
            _filtered_adj[_node] = _neighbors

    _subgraph = WeightedGraph(adjacency=_filtered_adj)
    _hc_hub_nodes = sorted(_hub_nodes & _subgraph.nodes)
    _hc_other_nodes = sorted(node for node in _subgraph.nodes if node not in _hc_hub_nodes)

    _node_colors_arr = []
    _node_sizes_arr = []
    for _node in _subgraph.nodes:
        _is_hits = _node in hits_terms
        _is_degree = _node in degree_terms
        if _is_hits and _is_degree:
            _node_colors_arr.append("#8e44ad")
        elif _is_hits:
            _node_colors_arr.append("#e45756")
        elif _is_degree:
            _node_colors_arr.append("#4c78a8")
        else:
            _node_colors_arr.append("#bdbdbd")
        _node_sizes_arr.append(120 + 40 * _subgraph.degree(_node))

    _model_hub_scores = (
        hits_summary[hits_summary["model_display"] == _selected_model]
        .set_index("term")["mean_hub_score"]
        .to_dict()
    )
    _hub_scores_arr = [_model_hub_scores.get(node, 0) for node in _hc_hub_nodes]
    _max_score = max(_hub_scores_arr) if _hub_scores_arr else 1

    _fig, _ax = plt.subplots(figsize=(12, 12))
    _ax.set_title(
        f"BFMN hub view: {_selected_model}  "
        f"({_layout_mode}, radius={_radius}, min_deg={_min_deg})"
    )
    _ax.axis("off")

    if _layout_mode == "force-directed":
        _nx_graph = nx.Graph()
        for _left, _right, _weight in _subgraph.edges:
            _nx_graph.add_edge(_left, _right, weight=_weight)
        _pos = nx.spring_layout(_nx_graph, k=0.6, iterations=50, seed=42)
    else:
        _pos = {}
        _inner_radius = 0.55
        _outer_radius = 1.55
        if _hc_hub_nodes:
            _hub_angles = np.linspace(0, 2 * np.pi, len(_hc_hub_nodes), endpoint=False)
            for _node, _angle in zip(_hc_hub_nodes, _hub_angles):
                _pos[_node] = (_inner_radius * np.cos(_angle), _inner_radius * np.sin(_angle))
        if _hc_other_nodes:
            _other_angles = np.linspace(0, 2 * np.pi, len(_hc_other_nodes), endpoint=False)
            for _node, _angle in zip(_hc_other_nodes, _other_angles):
                _pos[_node] = (_outer_radius * np.cos(_angle), _outer_radius * np.sin(_angle))

    for _left, _right, _weight in _subgraph.edges:
        _x1, _y1 = _pos.get(_left, (0, 0))
        _x2, _y2 = _pos.get(_right, (0, 0))
        _ax.plot(
            [_x1, _x2], [_y1, _y2],
            color="#666666", alpha=0.18,
            linewidth=0.4 + 0.04 * _weight,
        )

    _xs = [_pos[_node][0] for _node in _subgraph.nodes]
    _ys = [_pos[_node][1] for _node in _subgraph.nodes]
    _ax.scatter(
        _xs, _ys, s=_node_sizes_arr, c=_node_colors_arr,
        edgecolors="white", linewidths=0.5, zorder=3,
    )

    _legend_handles = [
        mpatches.Patch(color="#8e44ad", label="HITS & Degree Hub"),
        mpatches.Patch(color="#e45756", label="HITS Hub"),
        mpatches.Patch(color="#4c78a8", label="Degree Hub"),
        mpatches.Patch(color="#bdbdbd", label="Other Node"),
    ]
    _ax.legend(handles=_legend_handles, loc="upper right", fontsize=10, frameon=True)

    for _node in _hc_hub_nodes:
        _x, _y = _pos[_node]
        _score = _model_hub_scores.get(_node, 0)
        _font_size = 8 + 10 * (_score / _max_score) if _max_score > 0 else 8
        _ax.text(
            _x, _y, _node,
            fontsize=_font_size,
            ha="center", va="center",
            zorder=4, fontweight="bold",
        )

    if _layout_mode == "concentric":
        _ax.set_aspect("equal")
    plt.savefig(fname="code/figures/BFMN_HUBS_view_Ministral_14_B")
    plt.tight_layout()

    _fig
    return mpatches, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Degree HUB Only View
    This visualization uses **only degree-based hubs** — terms that most frequently appear as top-degree nodes across runs. Non-hub nodes are filtered by the minimum number of hub connections. Hub labels are scaled by their degree hub frequency.
    """)
    return


@app.cell
def _(
    Path,
    WeightedGraph,
    build_weighted_graph,
    degree_summary,
    degree_terms,
    df,
    hub_context_subgraph,
    layout_picker,
    load_global_edge_list,
    min_neighbor_degree,
    model_picker,
    mpatches,
    np,
    nx,
    plt,
    radius_slider,
):


    _selected_model = model_picker.value
    _model_name = df.loc[df["model_display"] == _selected_model, "model_name"].iloc[0]
    _edge_base = Path("code/Task3/global_edge_list/NEW_edge_list_global").resolve()
    _edge_df = load_global_edge_list(_model_name, _edge_base)
    _graph = build_weighted_graph(_edge_df)
    _hub_terms = sorted(degree_terms)
    _layout_mode = layout_picker.value
    _radius = radius_slider.value
    _min_deg = min_neighbor_degree.value
    _raw_subgraph = hub_context_subgraph(_graph, _hub_terms, radius=_radius)

    if len(_raw_subgraph) == 0:
        _fig, _ax = plt.subplots(figsize=(12, 12))
        _ax.text(0.5, 0.5, "No hub subgraph available", ha="center", va="center")
        _ax.set_title(f"Degree HUB only: {_selected_model}")
        _ax.axis("off")
        _fig
        #return

    _hub_nodes = {node for node in _hub_terms if node in _raw_subgraph.nodes}
    _filtered_nodes = set(_hub_nodes) | {
        node
        for node in _raw_subgraph.nodes
        if node not in _hub_nodes and _raw_subgraph.degree(node) >= _min_deg
    }

    _adj = _raw_subgraph.adjacency
    _filtered_adj = {}
    for _node in _filtered_nodes:
        _neighbors = {}
        for _nbr, _w in _adj.get(_node, {}).items():
            if _nbr in _filtered_nodes:
                _neighbors[_nbr] = _w
        if _neighbors:
            _filtered_adj[_node] = _neighbors

    _subgraph = WeightedGraph(adjacency=_filtered_adj)
    _hc_hub_nodes = sorted(_hub_nodes & _subgraph.nodes)
    _hc_other_nodes = sorted(node for node in _subgraph.nodes if node not in _hc_hub_nodes)

    _node_colors_arr = []
    _node_sizes_arr = []
    for _node in _subgraph.nodes:
        _is_hub = _node in _hub_terms
        _node_colors_arr.append("#4c78a8" if _is_hub else "#bdbdbd")
        _node_sizes_arr.append(120 + 40 * _subgraph.degree(_node))

    _model_degree_scores = (
        degree_summary[degree_summary["model_display"] == _selected_model]
        .set_index("term")["hub_frequency"]
        .to_dict()
    )
    _hub_scores_arr = [_model_degree_scores.get(node, 0) for node in _hc_hub_nodes]
    _max_score = max(_hub_scores_arr) if _hub_scores_arr else 1

    _fig, _ax = plt.subplots(figsize=(12, 12))
    _ax.set_title(
        f"Degree HUB only: {_selected_model}  "
        f"({_layout_mode}, radius={_radius}, min_deg={_min_deg})"
    )
    _ax.axis("off")

    if _layout_mode == "force-directed":
        _nx_graph = nx.Graph()
        for _left, _right, _weight in _subgraph.edges:
            _nx_graph.add_edge(_left, _right, weight=_weight)
        _pos = nx.spring_layout(_nx_graph, k=0.6, iterations=50, seed=42)
    else:
        _pos = {}
        _inner_radius = 0.55
        _outer_radius = 1.55
        if _hc_hub_nodes:
            _hub_angles = np.linspace(0, 2 * np.pi, len(_hc_hub_nodes), endpoint=False)
            for _node, _angle in zip(_hc_hub_nodes, _hub_angles):
                _pos[_node] = (_inner_radius * np.cos(_angle), _inner_radius * np.sin(_angle))
        if _hc_other_nodes:
            _other_angles = np.linspace(0, 2 * np.pi, len(_hc_other_nodes), endpoint=False)
            for _node, _angle in zip(_hc_other_nodes, _other_angles):
                _pos[_node] = (_outer_radius * np.cos(_angle), _outer_radius * np.sin(_angle))

    for _left, _right, _weight in _subgraph.edges:
        _x1, _y1 = _pos.get(_left, (0, 0))
        _x2, _y2 = _pos.get(_right, (0, 0))
        _ax.plot(
            [_x1, _x2], [_y1, _y2],
            color="#666666", alpha=0.18,
            linewidth=0.4 + 0.04 * _weight,
        )

    _xs = [_pos[_node][0] for _node in _subgraph.nodes]
    _ys = [_pos[_node][1] for _node in _subgraph.nodes]
    _ax.scatter(
        _xs, _ys, s=_node_sizes_arr, c=_node_colors_arr,
        edgecolors="white", linewidths=0.5, zorder=3,
    )

    _legend_handles = [
        mpatches.Patch(color="#4c78a8", label="Degree Hub"),
        mpatches.Patch(color="#bdbdbd", label="Other Node"),
    ]
    _ax.legend(handles=_legend_handles, loc="upper right", fontsize=10, frameon=True)

    for _node in _hc_hub_nodes:
        _x, _y = _pos[_node]
        _score = _model_degree_scores.get(_node, 0)
        _font_size = 8 + 10 * (_score / _max_score) if _max_score > 0 else 8
        _ax.text(
            _x, _y, _node,
            fontsize=_font_size,
            ha="center", va="center",
            zorder=4, fontweight="bold",
        )

    if _layout_mode == "concentric":
        _ax.set_aspect("equal")
    plt.savefig(fname="code/figures/Degree_HUB_only_view")
    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Degree Hubs per Model — Heatmap
    Hierarchically clustered heatmap showing which terms are degree hubs for which model. Terms = top 15 per model, unioned across all 14 models. Color = hub frequency (how many runs the term was a top-degree node). Dendrograms reveal model families and shared hub structure.
    """)
    return


@app.cell
def _(degree_summary, pivot_model_term_matrix, sns):
    _per_model_top = (
        degree_summary.groupby("model_display")
        .apply(lambda _x: _x.nlargest(15, "hub_frequency"))
        .reset_index(drop=True)
    )
    _all_terms = sorted(_per_model_top["term"].unique())

    matrix = pivot_model_term_matrix(degree_summary, "hub_frequency", _all_terms).fillna(0)

    _term_order = matrix.mean().sort_values(ascending=False).index
    matrix = matrix[_term_order]

    _g = sns.clustermap(
        matrix,
        cmap="crest",
        method="average",
        linewidths=0.5,
        linecolor="white",
        figsize=(16, 10),
        dendrogram_ratio=(0.1, 0.2),
        # Shifted to the right (1.02), centered vertically (0.4), made slightly taller (0.4)
        cbar_pos=(1.02, 0.4, 0.02, 0.4), 
        cbar_kws={"label": "Hub frequency"},
    )
    _g.fig.suptitle(
        "Degree hubs per model (top-15 per model, union set)",
        fontsize=25, y=1.02,
    )

    # ADDED bbox_inches="tight" so the external legend isn't cropped out
    _g.savefig("code/figures/All_models_degree_hub_heatmap", bbox_inches="tight")

    _g.fig
    return (matrix,)


@app.cell
def _():
    return


@app.cell
def _(matrix):
    matrix
    return


@app.cell
def _(matrix):


    # Assuming your dataframe is named 'matrix' and 'model_display' is set as the index.
    # If 'model_display' is currently a regular column, uncomment the line below:
    # matrix = matrix.set_index('model_display')

    # 1. Create a list to hold your 4 smaller dataframes
    split_tables = []
    chunk_size = 4

    for i in range(8):
        start_col = i * chunk_size
        end_col = start_col + chunk_size

        # iloc slices all rows (:), and chunks of 8 columns
        chunk_df = matrix.iloc[:, start_col:end_col]
        split_tables.append(chunk_df)

    # 2. Function to bold the maximum value in each column
    def highlight_max(s):
        is_max = s == s.max()
        # Returns CSS for HTML display, or LaTeX bolding if exported to LaTeX
        return ['font-weight: bold' if v else '' for v in is_max]

    # 3. Display and Export the Tables
    for i, table in enumerate(split_tables):
        print(f"\n--- Part {i+1} of 8 ---")

        # Format with commas for thousands and highlight the column max
        styled_table = table.style\
            .format("{:,.0f}")\
            .apply(highlight_max, axis=0)

        # CHOOSE YOUR OUTPUT METHOD BELOW:

        # Option A: If you are working in a Jupyter Notebook and want to see the formatted tables
        #print(styled_table) 

        # Option B: If you are writing your paper in LaTeX
        print(styled_table.to_latex(hrules=True))

        # Option C: If you are writing in Word/Google Docs (outputs Markdown)
        # print(styled_table.to_markdown())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GEXF Export
    Exports the degree-hub subgraph for each model as a `.gexf` file (openable in Gephi). Each file uses per-model top-15 degree hubs, radius=1, min neighbor degree=2. Node attributes: `hub_type`, `hub_frequency`, `degree`. Edge attribute: `weight`.
    """)
    return


@app.cell
def _(
    Path,
    WeightedGraph,
    build_weighted_graph,
    degree_summary,
    df,
    hub_context_subgraph,
    load_global_edge_list,
    mo,
    nx,
):
    _edge_base = Path("code/Task3/global_edge_list/NEW_edge_list_global").resolve()
    _export_dir = Path("code/figures/gexf").resolve()
    _export_dir.mkdir(parents=True, exist_ok=True)

    _files = []
    for _model_display in sorted(df["model_display"].unique()):
        _model_name = df[df["model_display"] == _model_display]["model_name"].iloc[0]

        _model_degree_terms = (
            degree_summary[degree_summary["model_display"] == _model_display]
            .sort_values("hub_frequency", ascending=False)
            .head(15)["term"]
            .tolist()
        )
        if not _model_degree_terms:
            continue

        _edge_df = load_global_edge_list(_model_name, _edge_base)
        _graph = build_weighted_graph(_edge_df)
        _raw = hub_context_subgraph(_graph, _model_degree_terms, radius=1)

        if len(_raw) == 0:
            continue

        _hub_nodes = {_t for _t in _model_degree_terms if _t in _raw.nodes}
        _filtered = set(_hub_nodes) | {
            _t for _t in _raw.nodes if _t not in _hub_nodes and _raw.degree(_t) >= 2
        }

        _adj = {}
        for _n in _filtered:
            _ns = {}
            for _nb, _w in _raw.adjacency.get(_n, {}).items():
                if _nb in _filtered:
                    _ns[_nb] = _w
            if _ns:
                _adj[_n] = _ns
        _sg = WeightedGraph(adjacency=_adj)

        _model_scores = (
            degree_summary[degree_summary["model_display"] == _model_display]
            .set_index("term")["hub_frequency"]
            .to_dict()
        )

        _nxg = nx.Graph()
        for _node in _sg.nodes:
            _is_hub = _node in _model_degree_terms
            _nxg.add_node(
                _node,
                hub_type="degree" if _is_hub else "other",
                hub_frequency=_model_scores.get(_node, 0),
                degree=_sg.degree(_node),
            )
        for _left, _right, _weight in _sg.edges:
            _nxg.add_edge(_left, _right, weight=_weight)

        _path = _export_dir / f"{_model_name}_hub_subgraph.gexf"
        nx.write_gexf(_nxg, str(_path))
        _files.append(str(_path.name))

    _n_success = len(_files)
    _n_total = df["model_display"].nunique()
    mo.md(
        f"Exported **{_n_success}/{_n_total}** model subgraphs to `code/figures/gexf/`:\n"
        + "\n".join(f"- `{f}`" for f in sorted(_files))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Other Visualizations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top Hubs Lollipop Chart
    This visualization extracts the top 15 hubs for the selected model and displays them as a quantitative ranking, offering a clear view of the most central concepts without network visual clutter.
    """)
    return


if __name__ == "__main__":
    app.run()
