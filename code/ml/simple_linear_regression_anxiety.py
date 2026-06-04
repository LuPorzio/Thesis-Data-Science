import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    from anxiety_math_hater_ols_utils import (
        fit_math_hater_gender_ols,
        fit_simple_ols,
        gender_mean_sem_frame,
        load_ml_dataset,
        multi_regression_metrics_frame,
        multi_regression_summary_frame,
        prepare_regression_frame,
        prepare_gender_regression_frame,
        regression_metrics_frame,
        regression_summary_frame,
    )

    return (
        Path,
        fit_math_hater_gender_ols,
        fit_simple_ols,
        gender_mean_sem_frame,
        load_ml_dataset,
        multi_regression_metrics_frame,
        multi_regression_summary_frame,
        pd,
        plt,
        prepare_gender_regression_frame,
        prepare_regression_frame,
        regression_metrics_frame,
        regression_summary_frame,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple OLS: anxiety vs. math hater flag

    Outcome: `anxiety_score = amas_score + mseaq_anx`

    Predictor: `math_hater_flg`
    """)
    return


@app.cell
def _(Path, load_ml_dataset):
    data_path = Path("code/ml/ml_dataset.csv").resolve()
    df = load_ml_dataset(data_path)
    return (df,)


@app.cell
def _(df, prepare_regression_frame):
    regression_df = prepare_regression_frame(df)
    regression_df.head()
    return (regression_df,)


@app.cell
def _(fit_simple_ols, regression_df):
    result = fit_simple_ols(regression_df)
    return (result,)


@app.cell
def _(regression_metrics_frame, regression_summary_frame, result):
    summary_df = regression_summary_frame(result)
    metrics_df = regression_metrics_frame(result)
    return metrics_df, summary_df


@app.cell
def _(summary_df):
    print(summary_df.to_latex(index=False, formatters={'p_value': '{:.2e}'.format}, escape=True
    ))
    return


@app.cell
def _(metrics_df, mo, summary_df):
    mo.md(f"""
    ## Model Output

    ```
    {summary_df.to_string(index=False)}
    ```

    ## Fit Statistics

    ```
    {metrics_df.to_string(index=False)}
    ```
    """)
    return


@app.cell
def _(plt, regression_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = regression_df.groupby("math_hater_flg")["anxiety_score"]
    means = grouped.mean().reindex([0, 1])
    sems = grouped.sem().reindex([0, 1])

    ax.bar(
        [0, 1],
        means.values,
        yerr=sems.values,
        capsize=6,
        color=["#4C78A8", "#F58518"],
        edgecolor="black",
        alpha=0.9,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["math_hater_flg = 0", "math_hater_flg = 1"])
    ax.set_xlabel("math_hater_flg")
    ax.set_ylabel("anxiety_score")
    ax.set_title("Mean anxiety score by math_hater_flg")
    ax.grid(alpha=0.2)
    plt.savefig(fname="code/figures/Mean_Anxiety_Hater")
    plt.tight_layout()
    plt.gca()
    return (ax,)


@app.cell
def _(mo, result):
    mo.md(f"""
    ### Key takeaways

    - Intercept: `{result.intercept:.4f}`
    - Slope: `{result.slope:.4f}`
    - Slope p-value: `{result.slope_pvalue:.4g}`
    - R^2: `{result.r2:.4f}`
    - Adjusted R^2: `{result.adj_r2:.4f}`
    - N: `{result.n_obs}`
    """)
    return


@app.cell
def _(ax, df, gender_mean_sem_frame, pd, plt):


    # 1. Get your summary dataframe
    _summary = gender_mean_sem_frame(df)

    # 2. Format the data and FILTER for only woman/man
    display_df = _summary.copy()
    display_df = display_df[display_df["gender_group"].isin(["woman", "man"])]

    # Sort to match the order of your bar charts
    display_df["gender_group"] = pd.Categorical(display_df["gender_group"], categories=["woman", "man"])
    display_df = display_df.sort_values(["gender_group", "math_hater_flg"])

    # Round numerical columns and calculate CI
    display_df["mean_anxiety"] = display_df["mean_anxiety"].round(2)
    display_df["sem_anxiety"] = display_df["sem_anxiety"].round(2)
    display_df["95% CI (+/-)"] = (display_df["sem_anxiety"] * 1.96).round(2)

    # Rename columns
    display_df = display_df.rename(columns={
        "gender_group": "Gender",
        "math_hater_flg": "Math Hater",
        "mean_anxiety": "Mean Anxiety",
        "sem_anxiety": "Standard Error"
    })

    # 3. Create the table figure (Made slightly taller to fit the text)
    _fig, _ax = plt.subplots(figsize=(8, 4)) 

    # Hide the axes
    _ax.axis("off")
    _ax.axis("tight")

    # Add the title
    _ax.set_title("Mean Anxiety Score by Math Hater Flag and Gender", fontweight="bold", fontsize=14, pad=20)

    # 4. Draw the table
    _table = _ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center", 
        loc="center"
    )

    # 5. Styling the table
    _table.scale(1, 1.8) 
    for (row, col), cell in _table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0') 

    # ---> NEW: Add the Key Takeaways as a footer block <---
    # We manually type out the formatted takeaways from your screenshot.
    # Note: I changed "2.645e-100" to "< 0.001" as that is the standard way to report tiny p-values visually!
    stats_text = (
        "Model Key Takeaways:\n"
        "Intercept: 90.77  |  Slope: 8.64  |  Slope p-value: < 0.001\n"
        "R²: 0.0213  |  Adjusted R²: 0.0213  |  N: 20,998"
    )

    # Place the text slightly below the bottom-left corner of the table
    ax.text(
        0, -0.15, stats_text, 
        transform=ax.transAxes,    # Aligns the text relative to the table boundaries
        fontsize=500, 
        verticalalignment='top', 
        linespacing=1.6, 
        # Adds a nice rounded box around the text so it stands out
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#dee2e6") 
    )

    # 6. Save as an image
    plt.savefig(
        fname="code/figures/Summary_Table_Complete.png", 
        bbox_inches="tight", 
        dpi=300              
    )
    plt.close()
    return (display_df,)


@app.cell
def _(display_df):
    print(display_df.to_latex(index=False, float_format="%.e", escape=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gender-adjusted OLS

    Outcome: `anxiety_score = amas_score + mseaq_anx`

    Predictors: `math_hater_flg`, `gender` (baseline: woman)
    """)
    return


@app.cell
def _(df, prepare_gender_regression_frame):
    gender_regression_df = prepare_gender_regression_frame(df)
    gender_regression_df.head()
    return (gender_regression_df,)


@app.cell
def _(fit_math_hater_gender_ols, gender_regression_df):
    gender_result = fit_math_hater_gender_ols(gender_regression_df)
    return (gender_result,)


@app.cell
def _(
    gender_result,
    multi_regression_metrics_frame,
    multi_regression_summary_frame,
):
    gender_summary_df = multi_regression_summary_frame(gender_result)
    gender_metrics_df = multi_regression_metrics_frame(gender_result)
    return gender_metrics_df, gender_summary_df


@app.cell
def _(gender_metrics_df, gender_summary_df, mo):
    mo.md(f"""
    ## Gender-adjusted Model Output

    ```
    {gender_summary_df.to_string(index=False)}
    ```

    ## Fit Statistics

    ```
    {gender_metrics_df.to_string(index=False)}
    ```
    """)
    return


@app.cell
def _(gender_summary_df):
    print(gender_summary_df.to_latex(
        index=False,
        formatters={'p_value': '{:.2e}'.format},
        escape=True
    ))
    return


@app.cell
def _(df, gender_mean_sem_frame, plt):
    summary = gender_mean_sem_frame(df)
    _gender_order = ["woman", "man"]

    # 1. Define your color dictionary here. 
    # The first color in the list is for math_hater_flg 0, the second is for 1.
    color_mapping = {
        "woman": ["#ffb3c1", "#ff4d6d"],  # Light pink, Dark pink
        "man": ["#90e0ef", "#00b4d8"]     # Light blue, Dark blue
    }

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for _ax, _gender_name in zip(_axes, _gender_order):
        subset = summary[summary["gender_group"] == _gender_name].set_index("math_hater_flg").reindex([0, 1])

        _ax.bar(
            [0, 1],
            subset["mean_anxiety"].values,
            yerr=subset["sem_anxiety"].values * 1.96, # 95% C.I.
            capsize=5,
            # 2. Use the dictionary to look up the correct colors for the current gender
            color=color_mapping[_gender_name], 
            edgecolor="black",
            alpha=0.9,
        )
        _ax.set_title(_gender_name)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["0", "1"])
        _ax.set_xlabel("math_hater_flg")
        _ax.grid(alpha=0.2)

    _axes[0].set_ylabel("mean anxiety_score")
    _fig.suptitle("Mean anxiety score by math_hater_flg and gender", y=1.02)
    plt.tight_layout()
    plt.savefig(fname="code/figures/Mean_Anxiety_Hater_Gender", dpi = 300, bbox_inches="tight")
    plt.show() # Note: replaced plt.gca() with plt.show() to actually render the plot if running interactively
    return


@app.cell
def _(gender_summary_df, plt):
    _coeff_df = gender_summary_df.copy()
    _fig, _ax = plt.subplots(figsize=(8, 4.5))

    _ypos = range(len(_coeff_df))
    _ax.errorbar(
        _coeff_df["estimate"],
        _ypos,
        xerr=[_coeff_df["estimate"] - _coeff_df["ci_low"], _coeff_df["ci_high"] - _coeff_df["estimate"]],
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        capsize=4,
    )
    _ax.axvline(0, color="black", linewidth=1, linestyle="--")
    _ax.set_yticks(list(_ypos))
    _ax.set_yticklabels(_coeff_df["term"])
    _ax.set_xlabel("Coefficient estimate")
    _ax.set_title("Gender-adjusted OLS coefficients with 95% CI")
    _ax.grid(alpha=0.2, axis="x")
    plt.savefig(fname="code/figures/OLS_coefficients")
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
