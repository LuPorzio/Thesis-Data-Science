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
    import seaborn as sns
    from pathlib import Path

    from net_features_analysis_utils import (
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

    _ax.set_title("Degree-hub frequency across models")
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
    build_weighted_graph,
    degree_terms,
    df,
    hits_terms,
    hub_context_subgraph,
    load_global_edge_list,
    model_picker,
    plt,
):
    import numpy as np
    import matplotlib.patches as mpatches

    _selected_model = model_picker.value
    _model_name = df.loc[df["model_display"] == _selected_model, "model_name"].iloc[0]
    _edge_base = Path("code/Task3/global_edge_list/NEW_edge_list_global").resolve()
    _edge_df = load_global_edge_list(_model_name, _edge_base)
    _graph = build_weighted_graph(_edge_df)
    _hub_terms = sorted(set(hits_terms).union(degree_terms))
    _subgraph = hub_context_subgraph(_graph, _hub_terms, radius=1)

    _fig, _ax = plt.subplots(figsize=(12, 12))
    _ax.set_title(f"BFMN hub view: {_selected_model}")
    _ax.axis("off")

    if len(_subgraph) == 0:
        _ax.text(0.5, 0.5, "No hub subgraph available", ha="center", va="center")
    else:
        _hub_nodes = [node for node in _hub_terms if node in _subgraph.nodes]
        _other_nodes = sorted(node for node in _subgraph.nodes if node not in _hub_nodes)
        _pos = {}

        _inner_radius = 0.55
        _outer_radius = 1.55
        if _hub_nodes:
            _hub_angles = np.linspace(0, 2 * np.pi, len(_hub_nodes), endpoint=False)
            for _node, _angle in zip(_hub_nodes, _hub_angles):
                _pos[_node] = (_inner_radius * np.cos(_angle), _inner_radius * np.sin(_angle))
        if _other_nodes:
            _other_angles = np.linspace(0, 2 * np.pi, len(_other_nodes), endpoint=False)
            for _node, _angle in zip(_other_nodes, _other_angles):
                _pos[_node] = (_outer_radius * np.cos(_angle), _outer_radius * np.sin(_angle))

        _node_colors = []
        _node_sizes = []
        for _node in _subgraph.nodes:
            _is_hits = _node in hits_terms
            _is_degree = _node in degree_terms
            if _is_hits and _is_degree:
                _node_colors.append("#8e44ad")
            elif _is_hits:
                _node_colors.append("#e45756")
            elif _is_degree:
                _node_colors.append("#4c78a8")
            else:
                _node_colors.append("#bdbdbd")
            _node_sizes.append(120 + 40 * _subgraph.degree(_node))

        for _left, _right, _weight in _subgraph.edges:
            _x1, _y1 = _pos[_left]
            _x2, _y2 = _pos[_right]
            _ax.plot([_x1, _x2], [_y1, _y2], color="#666666", alpha=0.18, linewidth=0.4 + 0.04 * _weight)

        _xs = [_pos[_node][0] for _node in _subgraph.nodes]
        _ys = [_pos[_node][1] for _node in _subgraph.nodes]
        _ax.scatter(_xs, _ys, s=_node_sizes, c=_node_colors, edgecolors="white", linewidths=0.5, zorder=3)

        _legend_handles = [
            mpatches.Patch(color="#8e44ad", label="HITS & Degree Hub"),
            mpatches.Patch(color="#e45756", label="HITS Hub"),
            mpatches.Patch(color="#4c78a8", label="Degree Hub"),
            mpatches.Patch(color="#bdbdbd", label="Other Node")
        ]
        _ax.legend(handles=_legend_handles, loc="upper right", fontsize=10, frameon=True)

        for _node in _hub_nodes:
            _x, _y = _pos[_node]
            _ax.text(_x, _y, _node, fontsize=8, ha="center", va="center", zorder=4)

        _ax.set_aspect("equal")
        plt.savefig(fname="code/figures/BFMN_HUBS_view_Ministral_14_B")
        plt.tight_layout()

    _fig
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


@app.cell
def _(hits_long, model_picker, plt, sns):
    _selected_model = model_picker.value
    _model_data = hits_long[hits_long["model_display"] == _selected_model]

    # Aggregate scores in case of multiple runs, getting the top 15 terms
    _top_hubs = _model_data.groupby("term")["degree_hubs"].mean().reset_index()
    _top_hubs = _top_hubs.sort_values(by="degree_hubs", ascending=False).head(15)

    _fig, _ax = plt.subplots(figsize=(10, 8))

    # Draw the stems
    _ax.hlines(y=range(len(_top_hubs)), xmin=0, xmax=_top_hubs["degree_hubs"], color="#6a9fb5", linewidth=3)

    # Draw the markers
    _ax.plot(_top_hubs["degree_hubs"], range(len(_top_hubs)), "o", markersize=10, color="#1f497d", alpha=0.8)

    # Formatting
    _ax.set_yticks(range(len(_top_hubs)))
    _ax.set_yticklabels(_top_hubs["term"], fontsize=12)
    _ax.set_xlabel("HITS Hub Score", fontsize=12, fontweight="bold")
    _ax.set_title(f"Top 15 Semantic Hubs for {_selected_model}", fontsize=14, fontweight="bold")
    _ax.invert_yaxis() # Highest score at the top
    sns.despine(left=True, bottom=True, ax=_ax)
    _ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Matching the savefig convention of the rest of the script
    plt.savefig(fname=f"code/figures/Lollipop_HITS_HUBS_{_selected_model.replace(' ', '_')}")

    _fig
    return


if __name__ == "__main__":
    app.run()
