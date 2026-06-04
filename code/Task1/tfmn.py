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

    import pandas as pd
    import numpy as np

    #from src.utils.constants import MAPPING_CALL1_QUESTIONS
    return Path, np, pd


@app.cell
def _():
    MAPPING_CALL1_QUESTIONS = {
        "What is your relationship with mathematics?": 1,
        "Do you ever get anxious when thinking about mathematics?": 2,
        "Did you ever use AI to support your math learning in the last year? If yes, how was your experience?": 3,
        "How would you explain, step by step, how to solve a second order algebraic equation?": 4,
        "How would you explain, step by step, how to find the stationary points of an equation y=f(x)?": 5,
        "Briefly, how do you perform a Principal Component Analysis? Should I get anxious about its mathematics? Please, teach me.": 6,
        "According to you, how can LLMs be used to innovate math learning in schools and universities?": 7
    }
    return (MAPPING_CALL1_QUESTIONS,)


@app.cell
def _():
    MODEL_NAME_MAPPING = {
        'deepseek-chat': 'DeepSeek Chat',
        'qwen/qwen3-4b-thinking-2507': 'Qwen3 4B (Thinking)',
        'Qwen/Qwen3-4B-Thinking-2507': 'Qwen3 4B (Thinking)',
        'nvidia/nemotron-3-nano': 'Nemotron-3 Nano',
        'mistralai/ministral-3-3b': 'Ministral 3B',
        'mistral-small-latest': 'Mistral Small 4',
        'ministral-3-3b-reasoning-2512': 'Ministral 3B', 
        'mistralai/mistral-small-3.2': 'Mistral Small 3.2',
        'mistral-small-2506': 'Mistral Small 3.2',
        'mistral-small3.2:latest': 'Mistral Small 3.2', # Unified with the above
        'anita-next-24b-dolphin-mistral-uncensored-ita-i1': 'Anita 24B (Uncensored)',
        'qwen3-4b-instruct-2507-uncensored-unslop-v2': 'Qwen3 4B (Uncensored)',
        'electroglyph/Qwen3-4B-Instruct-2507-uncensored-unslop-v2': 'Qwen3 4B (Uncensored)',
        'ibm/granite-4-h-tiny': 'Granite 4 Tiny',
        'ibm-granite/granite-4.0-h-tiny': 'Granite 4 Tiny',
        'qwen/qwen3.5-9b': 'Qwen3.5 9B',
        'qwen/qwen3.5-9B': 'Qwen3.5 9B',
        'qwen/qwen3-4b-2507': 'Qwen3 4B',
        'Qwen/Qwen3-4B-Instruct-2507': 'Qwen3 4B',
        'mistralai/ministral-3-14b-reasoning': 'Ministral 14B (Reasoning)',
        'ministral-14b-latest': 'Ministral 14B (Reasoning)',
        'microsoft/phi-4-reasoning-plus': 'Phi-4 (Reasoning+)',
        'microsoft/Phi-4-reasoning-plus': 'Phi-4 (Reasoning+)',
        'mistralai/magistral-small-2509': 'Magistral Small',
        'magistral-small-latest': 'Magistral Small',
        'llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b': 'Llama 3.2 MoE 18.4B'
    }
    return (MODEL_NAME_MAPPING,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preliminary steps

    These include:

    1. mapping the question to their numbers
    """)
    return


@app.cell
def _(MAPPING_CALL1_QUESTIONS, MODEL_NAME_MAPPING, Path, pd):
    file_path = Path("code/Task1/individual_tfmn/tfmn_dataset.csv").resolve().absolute()
    df = pd.read_csv(file_path)

    # 1. Mapping question to number as presented in call1
    df["question_number"] = df["question_number"].map(MAPPING_CALL1_QUESTIONS)
    df["model_name"] = df["model_name"].replace(MODEL_NAME_MAPPING)
    return (df,)


@app.cell
def _(df):
    df.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis
    """)
    return


@app.cell
def _(df):
    # Define aggregation lists
    by_model_mode_question = ['model_name', 'mode', 'question_number']
    by_model_question = ['model_name', 'question_number']
    by_model = ['model_name']
    df_by_model_mode_question = df.groupby(by_model_mode_question)
    # Defines grouped DataFrames
    df_by_model_question = df.groupby(by_model_question)
    df_by_model = df.groupby(by_model)
    # Define columns concerning z_scores
    z_scores_cols = list(filter(lambda x: x.startswith('z_scores'), df.columns))
    return df_by_model_mode_question, df_by_model_question, z_scores_cols


@app.cell
def _(df_by_model_mode_question, z_scores_cols):
    df_by_model_mode_question.agg({_z_score: 'median' for _z_score in z_scores_cols})
    return


@app.cell
def _(df_by_model_question, z_scores_cols):
    df_by_model_question.agg({_z_score: 'median' for _z_score in z_scores_cols}).reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Check Emotions for Validation

    The objective is to check if humans who have math across fav subjects have a different flower compared to those who do not.
    """)
    return


@app.cell
def _(Path, df, pd):
    path_socio_demo = Path("code/Task1/individual_tfmn/persona_dataset.csv").resolve().absolute()
    QUESTIONS_TO_CONSIDER = [1, 2, 3, 7]

    df_humans = df[(df["mode"] == "human") & (df["question_number"].isin(QUESTIONS_TO_CONSIDER))]
    df_socio_demo = pd.read_csv(path_socio_demo)
    complete_df = df_humans.merge(df_socio_demo, how="left", on="run_id")
    complete_df.columns
    return QUESTIONS_TO_CONSIDER, complete_df, df_socio_demo


@app.cell
def _(complete_df):
    # Get entries having math in their favorite subjects and create a column that evaluates to True when
    # the person likes math, False otherwise
    complete_df["flg_fav_math"] = complete_df["fav_subjects"].str.contains('math', case=False, na=False)
    complete_df["flg_hat_math"] = complete_df["hat_subjects"].str.contains('math', case=False, na=False)
    complete_df.head()
    return


@app.cell
def _(df_socio_demo):
    df_socio_demo
    return


@app.cell
def _(complete_df):
    fav_df = complete_df[complete_df["flg_fav_math"]]
    hat_df = complete_df[complete_df["flg_hat_math"]]
    return fav_df, hat_df


@app.cell
def _(fav_df, z_scores_cols):
    for _z_score in z_scores_cols:
        print(_z_score, fav_df[_z_score].mean())
    return


@app.cell
def _(hat_df, z_scores_cols):
    for _z_score in z_scores_cols:
        print(_z_score, hat_df[_z_score].mean())
    return


@app.cell
def _(fav_df, z_scores_cols):
    fav_df.groupby('model_name_x').agg({_z_score: 'mean' for _z_score in z_scores_cols}).reset_index()
    return


@app.cell
def _(hat_df, z_scores_cols):
    hat_df.groupby('model_name_x').agg({_z_score: 'mean' for _z_score in z_scores_cols}).reset_index()
    return


@app.cell
def _(fav_df, z_scores_cols):
    fav_df_agg = fav_df.groupby(['model_name_x', 'question_number']).agg({_z_score: 'mean' for _z_score in z_scores_cols}).reset_index()
    return (fav_df_agg,)


@app.cell
def _(hat_df, z_scores_cols):
    hat_df_agg = hat_df.groupby(['model_name_x', 'question_number']).agg({_z_score: 'mean' for _z_score in z_scores_cols}).reset_index()
    return (hat_df_agg,)


@app.cell
def _(fav_df):
    # Get sample size for math lovers dataset aggregated
    fav_df.groupby(["model_name_x", "question_number"]).agg(
        sample_size = ("z_scores_anger", "size")
    ).reset_index().query("question_number == 1")
    return


@app.cell
def _(fav_df):
    fav = fav_df["model_name_x"]
    return (fav,)


@app.cell
def _(fav):
    fav.unique()
    return


@app.cell
def _(hat_df):
    # Get sample size math haters
    hat_df.groupby(["model_name_x", "question_number"]).agg(
        sample_size = ("z_scores_anger", "size")
    ).reset_index().query("question_number == 1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating New Groups: High Anxiety and Low Anxiety
    """)
    return


@app.cell
def _():
    # # 1. Sum up the individual MSEAQ items to get the total anxiety score
    # mseaq_cols = [col for col in complete_df.columns if 'mseaq_' in col and col.endswith('_rating')]
    # complete_df['mseaq_total_score'] = complete_df[mseaq_cols].sum(axis=1)

    # # 2. Calculate the median score across the entire dataset
    # mseaq_median = complete_df['mseaq_total_score'].median()

    # # 3. Create boolean flags exactly like your flg_fav_math example
    # complete_df['flg_high_anxiety'] = complete_df['mseaq_total_score'] > mseaq_median
    # complete_df['flg_low_anxiety'] = complete_df['mseaq_total_score'] <= mseaq_median

    # # (Optional) If you also want a single text column for plotting later:
    # complete_df['anxiety_group'] = np.where(
    #     complete_df['mseaq_total_score'] > mseaq_median, 
    #     'High_Anxiety', 
    #     'Low_Anxiety'
    # )

    # complete_df[['run_id', 'model_name_x', 'mseaq_total_score', 'flg_high_anxiety', 'anxiety_group']].head()
    return


@app.cell
def _(Path, pd):
    call2_path = Path("code/Task1/individual_tfmn/call2_dataset.csv").resolve().absolute()

    def get_item_number(x):
        if '_' in x:
            return x.split('_')[-1]
        else:
            return x

    df_call2 = pd.read_csv(call2_path)
    df_call2['item number'] = df_call2['item number'].astype(str).apply(get_item_number)
    filtered_call2_amas = df_call2[(df_call2['mode'] == 'human') & df_call2['scale'].str.contains('amas')]
    filtered_call2_mseaq = df_call2[(df_call2['mode'] == 'human') & df_call2['scale'].str.contains('mseaq')]
    filtered_call2_mseaq_anxiety = filtered_call2_mseaq[filtered_call2_mseaq['item number'].isin([str(_x) for _x in range(8, 29)])]
    # Group by model and filter to get only simulated humans
    ITEMS_TO_REVERSE = ['13', '14', '17', '22', '25', '27']
    REVERSE_ITEMS_MAP = {'1': '5', '2': '4', '3': '3', '4': '2', '5': '1'}

    def map_if_mseaq(df):
        df = df.copy()
        condition = (df['scale'] == 'mseaq') & df['item number'].astype(str).isin(ITEMS_TO_REVERSE)
        df.loc[condition, 'rating'] = df.loc[condition, 'rating'].astype(str).map(REVERSE_ITEMS_MAP).astype(int)
        return df
    filtered_call2_mseaq_anxiety = map_if_mseaq(filtered_call2_mseaq_anxiety)
    filtered_call2 = pd.concat([filtered_call2_amas, filtered_call2_mseaq_anxiety], axis=0)
    aggregated_dataset = filtered_call2.groupby(['run_id', 'scale']).agg(total_score=('rating', 'sum')).reset_index()  # Create a copy to avoid SettingWithCopyWarning if passing a slice  # 1. Create a boolean mask for the specific conditions  # We cast 'item number' to string just in case it's stored as an integer  # 2. Apply the mapping only to the rows that meet the condition  # We temporarily cast 'rating' to string to use your map, then cast back to numeric
    return aggregated_dataset, df_call2, map_if_mseaq


@app.cell
def _(aggregated_dataset):
    aggregated_dataset.head()
    return


@app.cell
def _(aggregated_dataset):
    # 1. Pivot the dataset
    pivoted_dataset = aggregated_dataset.pivot(
        index="run_id", 
        columns="scale", 
        values="total_score"
    ).reset_index()

    # 2. Rename the columns to include the '_score' suffix
    pivoted_dataset = pivoted_dataset.rename(columns={
        "amas": "amas_score",
        "mseaq": "mseaq_score"
    })

    # 3. (Optional) Remove the generic 'scale' name from the columns axis for a cleaner look
    pivoted_dataset.columns.name = None

    # Check the result
    print(pivoted_dataset.head())
    return (pivoted_dataset,)


@app.cell
def _(df_socio_demo, pivoted_dataset):
    # dataset demographics with anxiety information
    persona_merged = df_socio_demo.merge(pivoted_dataset, on="run_id", how="inner")
    return (persona_merged,)


@app.cell
def _(persona_merged):
    persona_merged["mode"].unique()
    return


@app.cell
def _(persona_merged):
    # compute the anxiety score as the sum of the scores of the two psychometric scales
    persona_merged["anxiety_score"] = persona_merged["amas_score"] + persona_merged["mseaq_score"]
    return


@app.cell
def _(np, persona_merged):
    persona_merged['model_median_anxiety'] = persona_merged.groupby('model_name')['anxiety_score'].transform('median')
    persona_merged['anxiety_level'] = np.where(persona_merged['anxiety_score'] < persona_merged['model_median_anxiety'], 'low_anxiety', 'high_anxiety')
    gender = persona_merged['gender'].astype(str).str.lower()
    persona_merged['gender_level'] = np.where(
        gender.str.contains(r'\b(?:male|man)\b', regex=True, na=False),
        'male',
        np.where(gender.str.contains(r'\b(?:female|woman)\b', regex=True, na=False), 'female', "other")
    )
    # 1. Calculate the median for each model and align it with the original rows
    # 2. Create the new category column based on the condition
    # Optional: drop the helper median column if you no longer need it
    persona_merged_1 = persona_merged.drop(columns=['model_median_anxiety'])
    return (persona_merged_1,)


@app.cell
def _(persona_merged_1):
    persona_merged_1
    return


@app.cell
def _(persona_merged_1):
    persona_merged_1["mode"].nunique()
    return


@app.cell
def _(df, persona_merged_1):
    df_anxiety_zscore = df.merge(persona_merged_1, on='run_id', how='inner')
    df_anxiety_zscore
    return (df_anxiety_zscore,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating high anxiety dataset
    Filtering by persona with anxiety_level == high_anxiety
    """)
    return


@app.cell
def _(persona_merged_1):
    df_anxiety_high = persona_merged_1[persona_merged_1["anxiety_level"] == "high_anxiety"]
    df_anxiety_high.head()
    return (df_anxiety_high,)


@app.cell
def _(Path, df_anxiety_high):
    path_high_anx = Path("code/Task1/individual_tfmn/").resolve().absolute()
    df_anxiety_high.to_csv(path_high_anx / "high_anxiety_personas.csv", index = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating low anxiety dataset
    Filtering by persona with anxiety_level == low_anxiety
    """)
    return


@app.cell
def _(persona_merged_1):
    df_anxiety_low = persona_merged_1[persona_merged_1["anxiety_level"] == "low_anxiety"]
    df_anxiety_low.head()
    return (df_anxiety_low,)


@app.cell
def _(Path, df_anxiety_low):
    path_low_anx = Path("code/Task1/individual_tfmn/").resolve().absolute()
    df_anxiety_low.to_csv(path_low_anx / "low_anxiety_personas.csv", index = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizations
    """)
    return


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt

    def plot_emotional_heatmap(df_agg, title):
        """
        Generates a FacetGrid heatmap of emotional Z-scores across different 
        questions and models, with a single uniform color scale.
        """
        df_plot = df_agg.copy()
        df_plot['model_name_x'] = df_plot['model_name_x'].apply(lambda x: str(_x)[:30] + '...' if len(str(_x)) > 30 else _x)
        _emotion_cols = [col for col in df_plot.columns if str(col).startswith('z_scores_')]
        clean_emotion_cols = {col: col.replace('z_scores_', '').capitalize() for col in _emotion_cols}
        df_plot = df_plot.rename(columns=clean_emotion_cols)
        df_melted = df_plot.melt(id_vars=['model_name_x', 'question_number'], value_vars=list(clean_emotion_cols.values()), var_name='Emotion', value_name='Z-Score')
        global_vmin = df_melted['Z-Score'].min()
        global_vmax = df_melted['Z-Score'].max()

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot_table(index='model_name_x', columns='Emotion', values='Z-Score')
            sns.heatmap(d, cmap='RdBu_r', center=0, annot=True, fmt='.1f', annot_kws={'size': 9}, cbar_kws={'label': 'Mean Z-Score'}, yticklabels=True, vmin=global_vmin, vmax=global_vmax, **kwargs)
        _g = sns.FacetGrid(df_melted, col='question_number', col_wrap=2, height=7, aspect=1.8, sharex=False, sharey=False)
        _g.map_dataframe(draw_heatmap)
        _g.set_titles(col_template='Question {col_name}', size=14, weight='bold')
        _g.set_axis_labels('', '')
        for i, _ax in enumerate(_g.axes.flat):
            _ax.set_xticklabels(_ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            if i % 2 == 0:
                _ax.tick_params(axis='y', labelleft=True)
                _ax.set_yticklabels(_ax.get_yticklabels(), fontsize=10, rotation=0)
            else:
                _ax.tick_params(axis='y', labelleft=False)
        plt.suptitle(title, y=1.03, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return plt, sns


@app.cell
def _(fav_df_agg, hat_df_agg, plt, sns):
    def plot_unified_heatmaps(fav_df_agg, hat_df_agg):
        """
        Calculates a universal color scale across two datasets (Lovers and Haters)
        and plots their FacetGrid heatmaps sequentially for perfect 1-to-1 visual comparison.
        """
        _emotion_cols = [col for col in fav_df_agg.columns if str(col).startswith('z_scores_')]
        vmin = min(fav_df_agg[_emotion_cols].min().min(), hat_df_agg[_emotion_cols].min().min())
        vmax = max(fav_df_agg[_emotion_cols].max().max(), hat_df_agg[_emotion_cols].max().max())
        print(f'Universal Scale Locked: Min {vmin:.2f} | Max {vmax:.2f}')

        def _plot_grid(df_agg, title):
            df_plot = df_agg.copy()
            df_plot['model_name_x'] = df_plot['model_name_x'].apply(lambda x: str(x)[:30] + '...' if len(str(x)) > 30 else x)
            clean_emotion_cols = {col: col.replace('z_scores_', '').capitalize() for col in _emotion_cols}
            df_plot = df_plot.rename(columns=clean_emotion_cols)
            df_melted = df_plot.melt(id_vars=['model_name_x', 'question_number'], value_vars=list(clean_emotion_cols.values()), var_name='Emotion', value_name='Z-Score')

            def draw_heatmap(*args, **kwargs):
                data = kwargs.pop('data')
                d = data.pivot_table(index='model_name_x', columns='Emotion', values='Z-Score')
                sns.heatmap(d, cmap='RdBu_r', center=0, annot=True, fmt='.1f', annot_kws={'size': 9}, cbar_kws={'label': 'Mean Z-Score'}, yticklabels=True, vmin=vmin, vmax=vmax, **kwargs)
            _g = sns.FacetGrid(df_melted, col='question_number', col_wrap=2, height=7, aspect=1.8, sharex=False, sharey=False)
            _g.map_dataframe(draw_heatmap)
            _g.set_titles(col_template='Question {col_name}', size=14, weight='bold')
            _g.set_axis_labels('', '')
            for i, _ax in enumerate(_g.axes.flat):
                _ax.set_xticklabels(_ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
                if i % 2 == 0:
                    _ax.tick_params(axis='y', labelleft=True)
                    _ax.set_yticklabels(_ax.get_yticklabels(), fontsize=10, rotation=0)
                else:
                    _ax.tick_params(axis='y', labelleft=False)
            plt.suptitle(title, y=1.03, fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.show()
        _plot_grid(fav_df_agg, 'Emotional Z-Scores by Model and Question (Math Lovers)')
        _plot_grid(hat_df_agg, 'Emotional Z-Scores by Model and Question (Math Haters)')
    plot_unified_heatmaps(fav_df_agg, hat_df_agg)
    return


@app.cell
def _(fav_df_agg, hat_df_agg, np, plt):
    def plot_comparative_radar(fav_df, hat_df, target_model: str, target_question: int):
        """
        Plots a multi-dimensional radar chart comparing the 8 Plutchik emotions
        for Math Lovers vs. Math Haters.
        """
        fav_row = fav_df[(fav_df['model_name_x'] == target_model) & (fav_df['question_number'] == target_question)]
        hat_row = hat_df[(hat_df['model_name_x'] == target_model) & (hat_df['question_number'] == target_question)]
        if fav_row.empty or hat_row.empty:
            print('Missing data for one or both groups!')
            return
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        fav_scores = [fav_row.iloc[0][f'z_scores_{emo}'] for emo in emotions]
        hat_scores = [hat_row.iloc[0][f'z_scores_{emo}'] for emo in emotions]
        fav_scores = fav_scores + [fav_scores[0]]
        hat_scores = hat_scores + [hat_scores[0]]
        angles = [n / float(len(emotions)) * 2 * np.pi for n in range(len(emotions))]
        angles = angles + [angles[0]]
        fig, _ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], [emo.capitalize() for emo in emotions], size=13, fontweight='bold')
        _ax.set_rlabel_position(30)
        _ax.plot(angles, fav_scores, linewidth=2, linestyle='solid', label='Math Lovers (Blue)', color='#1f77b4')
        _ax.fill(angles, fav_scores, '#1f77b4', alpha=0.15)
        _ax.plot(angles, hat_scores, linewidth=2, linestyle='solid', label='Math Haters (Red)', color='#d62728')
        _ax.fill(angles, hat_scores, '#d62728', alpha=0.15)
        plt.title(f'Emotional Profile Comparison: {target_model}\n(Question {target_question})', size=16, fontweight='bold', y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
        plt.tight_layout()
        plt.show()
    TARGET_MODEL = 'DeepSeek Chat'
    QUESTION = 1
    plot_comparative_radar(fav_df_agg, hat_df_agg, TARGET_MODEL, QUESTION)
    return


@app.cell
def _(Path, plt):
    import shapely.geometry as sg
    from typing import Optional
    import emoatlas.emotions as emo_draw

    def plot_final_plutchik_comparison(fav_df, hat_df, target_model: str, target_question, out_dir: Optional[Path]=None):
        fav_row = fav_df[(fav_df['model_name_x'] == target_model) & (fav_df['question_number'] == target_question)]
        hat_row = hat_df[(hat_df['model_name_x'] == target_model) & (hat_df['question_number'] == target_question)]
        if fav_row.empty or hat_row.empty:
            return print(f'Data missing for {target_model} Q{target_question}!')
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        fav_scores = {emo: fav_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        hat_scores = {emo: hat_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        all_vals = list(fav_scores.values()) + list(hat_scores.values())
        abs_limit = max(abs(min(all_vals)), abs(max(all_vals))) + 0.5
        unified_rescale = (-abs_limit, abs_limit)
        reject = (-1.645, 1.645)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        def draw_flower_via_emoatlas(ax, scores):
            mmin = (reject[0] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            mmax = (reject[1] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            reject_circle = sg.Point(0, 0).buffer(mmax)
            _ax.add_patch(emo_draw.PolygonPatch(reject_circle, fc='grey', ec=(0.5, 0.5, 0.5, 0.3), alpha=0.1, zorder=-2))
            _ax.add_artist(plt.Circle((0, 0), mmax, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            inner_white = sg.Point(0, 0).buffer(mmin)
            _ax.add_patch(emo_draw.PolygonPatch(inner_white, fc='white', ec=(0.5, 0.5, 0.5, 0), zorder=-1))
            _ax.add_artist(plt.Circle((0, 0), mmin, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            for i in range(0, 110, 20):
                _ax.add_artist(plt.Circle((0, 0), 0.15 + i / 100, color='grey', alpha=0.3, fill=False, zorder=-20))
            for emo in emotions:
                emo_draw._draw_emotion_petal(ax=_ax, emotion_score=scores[emo], emotion=emo, font='sans-serif', fontweight='light', fontsize=15, highlight='all', show_intensity_levels='none', show_coordinates=True, height_width_ratio=1, reject_range=reject, rescale=unified_rescale)
            center_circle = sg.Point(0, 0).buffer(0.15)
            _ax.add_patch(emo_draw.PolygonPatch(center_circle, fc='white', ec=(0.5, 0.5, 0.5, 0.3), alpha=1, zorder=15))
            _ax.set_xlim(-1.6, 1.6)
            _ax.set_ylim(-1.6, 1.6)
            _ax.axis('off')
        draw_flower_via_emoatlas(axes[0], fav_scores)
        axes[0].set_title(f'Math Lovers\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[1], hat_scores)
        axes[1].set_title(f'Math Haters\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        plt.suptitle(f'Emotional Profile Comparison: {target_model[:25]}', fontsize=20, y=1.05, fontweight='bold')
        plt.tight_layout()
        if not out_dir:
            plt.show()
        else:
            san_model = target_model.replace('/', '-')
            plt.savefig(out_dir.joinpath(f'{san_model}_Q{target_question}.png'))

    return Optional, emo_draw, plot_final_plutchik_comparison, sg


@app.cell
def _(QUESTIONS_TO_CONSIDER, df, z_scores_cols):
    # 1. Isolate LLM data for the considered questions
    llm_df = df[(df['mode'] == 'llm') & df['question_number'].isin(QUESTIONS_TO_CONSIDER)]
    llm_df_agg = llm_df.groupby(['model_name', 'question_number']).agg({_z_score: 'mean' for _z_score in z_scores_cols}).reset_index()
    # 2. Aggregate the z-scores
    llm_df_agg.head()
    return llm_df, llm_df_agg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Emotion comparison for Math Lovers, Haters and LLM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## grok-4-1-fast-reasoning
    """)
    return


@app.cell
def _(Optional, Path, emo_draw, fav_df_agg, hat_df_agg, llm_df_agg, plt, sg):
    def plot_final_plutchik_comparison_triple(fav_df, hat_df, llm_df, target_model: str, target_question, out_dir: Optional[Path]=None):
        fav_row = fav_df[(fav_df['model_name_x'] == target_model) & (fav_df['question_number'] == target_question)]
        hat_row = hat_df[(hat_df['model_name_x'] == target_model) & (hat_df['question_number'] == target_question)]
        llm_row = llm_df[(llm_df['model_name'] == target_model) & (llm_df['question_number'] == target_question)]
        if fav_row.empty or hat_row.empty or llm_row.empty:
            return print(f'Data missing for {target_model} Q{target_question}!')
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        fav_scores = {emo: fav_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        hat_scores = {emo: hat_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        llm_scores = {emo: llm_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        all_vals = list(fav_scores.values()) + list(hat_scores.values()) + list(llm_scores.values())
        abs_limit = max(abs(min(all_vals)), abs(max(all_vals))) + 0.5
        unified_rescale = (-abs_limit, abs_limit)
        reject = (-1.645, 1.645)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        def draw_flower_via_emoatlas(ax, scores):
            mmin = (reject[0] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            mmax = (reject[1] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            reject_circle = sg.Point(0, 0).buffer(mmax)
            ax.add_patch(emo_draw.PolygonPatch(reject_circle, fc='grey', ec=(0.5, 0.5, 0.5, 0.3), alpha=0.1, zorder=-2))
            ax.add_artist(plt.Circle((0, 0), mmax, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            inner_white = sg.Point(0, 0).buffer(mmin)
            ax.add_patch(emo_draw.PolygonPatch(inner_white, fc='white', ec=(0.5, 0.5, 0.5, 0), zorder=-1))
            ax.add_artist(plt.Circle((0, 0), mmin, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            for i in range(0, 110, 20):
                ax.add_artist(plt.Circle((0, 0), 0.15 + i / 100, color='grey', alpha=0.3, fill=False, zorder=-20))
            for emo in emotions:
                emo_draw._draw_emotion_petal(ax=ax, emotion_score=scores[emo], emotion=emo, font='sans-serif', fontweight='light', fontsize=15, highlight='all', show_intensity_levels='none', show_coordinates=True, height_width_ratio=1, reject_range=reject, rescale=unified_rescale)
            center_circle = sg.Point(0, 0).buffer(0.15)
            ax.add_patch(emo_draw.PolygonPatch(center_circle, fc='white', ec=(0.5, 0.5, 0.5, 0.3), alpha=1, zorder=15))
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.axis('off')
        draw_flower_via_emoatlas(axes[0], fav_scores)
        axes[0].set_title(f'Math Lovers\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[1], hat_scores)
        axes[1].set_title(f'Math Haters\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[2], llm_scores)
        axes[2].set_title(f'LLM\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        plt.suptitle(f'Emotional Profile Comparison: {target_model[:25]}', fontsize=20, y=1.05, fontweight='bold')
        plt.tight_layout()
        if not out_dir:
            plt.show()
        else:
            san_model = target_model.replace('/', '-')
            #plt.savefig(out_dir.joinpath(f'{san_model}_Q{target_question}_Triple.png'), bbox_inches='tight')
    plot_final_plutchik_comparison_triple(fav_df_agg, hat_df_agg, llm_df_agg, 'grok-4-1-fast-reasoning', 1)
    return (plot_final_plutchik_comparison_triple,)


@app.cell
def _(
    fav_df_agg,
    hat_df_agg,
    llm_df_agg,
    plot_final_plutchik_comparison_triple,
):
    plot_final_plutchik_comparison_triple(fav_df_agg, hat_df_agg, llm_df_agg, 'grok-4-1-fast-reasoning', 2)
    return


@app.cell
def _(
    fav_df_agg,
    hat_df_agg,
    llm_df_agg,
    plot_final_plutchik_comparison_triple,
):
    plot_final_plutchik_comparison_triple(fav_df_agg, hat_df_agg, llm_df_agg, 'grok-4-1-fast-reasoning', 3)
    return


@app.cell
def _(
    fav_df_agg,
    hat_df_agg,
    llm_df_agg,
    plot_final_plutchik_comparison_triple,
):
    plot_final_plutchik_comparison_triple(fav_df_agg, hat_df_agg, llm_df_agg, 'grok-4-1-fast-reasoning', 7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mistral Small 3.2
    """)
    return


@app.cell
def _(
    fav_df_agg,
    hat_df_agg,
    llm_df_agg,
    plot_final_plutchik_comparison_triple,
):
    plot_final_plutchik_comparison_triple(fav_df_agg, hat_df_agg, llm_df_agg, 'Mistral Small 3.2', 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison Human Math Lovers vs Human Math Haters vs LLm (AGGREGATED QUESTIOSN 1-2-3-7)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mistral Small 3.2
    """)
    return


@app.cell
def _(Optional, Path, emo_draw, fav_df_agg, hat_df_agg, llm_df_agg, plt, sg):
    from typing import List

    def plot_final_plutchik_comparison_triple2(fav_df, hat_df, llm_df, target_model: str, target_questions: List[int], out_dir: Optional[Path]=None):
        fav_rows = fav_df[(fav_df['model_name_x'] == target_model) & fav_df['question_number'].isin(target_questions)]
        hat_rows = hat_df[(hat_df['model_name_x'] == target_model) & hat_df['question_number'].isin(target_questions)]
        llm_rows = llm_df[(llm_df['model_name'] == target_model) & llm_df['question_number'].isin(target_questions)]
        if fav_rows.empty or hat_rows.empty or llm_rows.empty:
            return print(f'Data missing for {target_model} Qs: {target_questions}!')
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        fav_scores = {emo: fav_rows[f'z_scores_{emo}'].mean() for emo in emotions}
        hat_scores = {emo: hat_rows[f'z_scores_{emo}'].mean() for emo in emotions}
        llm_scores = {emo: llm_rows[f'z_scores_{emo}'].mean() for emo in emotions}
        all_vals = list(fav_scores.values()) + list(hat_scores.values()) + list(llm_scores.values())
        abs_limit = max(abs(min(all_vals)), abs(max(all_vals))) + 0.5
        unified_rescale = (-abs_limit, abs_limit)
        reject = (-1.645, 1.645)
        q_title_str = ', '.join(map(str, target_questions))
        q_file_str = '_'.join(map(str, target_questions))
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        def draw_flower_via_emoatlas(ax, scores):
            mmin = (reject[0] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            mmax = (reject[1] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            reject_circle = sg.Point(0, 0).buffer(mmax)
            ax.add_patch(emo_draw.PolygonPatch(reject_circle, fc='grey', ec=(0.5, 0.5, 0.5, 0.3), alpha=0.1, zorder=-2))
            ax.add_artist(plt.Circle((0, 0), mmax, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            inner_white = sg.Point(0, 0).buffer(mmin)
            ax.add_patch(emo_draw.PolygonPatch(inner_white, fc='white', ec=(0.5, 0.5, 0.5, 0), zorder=-1))
            ax.add_artist(plt.Circle((0, 0), mmin, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            for i in range(0, 110, 20):
                ax.add_artist(plt.Circle((0, 0), 0.15 + i / 100, color='grey', alpha=0.3, fill=False, zorder=-20))
            for emo in emotions:
                emo_draw._draw_emotion_petal(ax=ax, emotion_score=scores[emo], emotion=emo, font='sans-serif', fontweight='light', fontsize=15, highlight='all', show_intensity_levels='none', show_coordinates=True, height_width_ratio=1, reject_range=reject, rescale=unified_rescale)
            center_circle = sg.Point(0, 0).buffer(0.15)
            ax.add_patch(emo_draw.PolygonPatch(center_circle, fc='white', ec=(0.5, 0.5, 0.5, 0.3), alpha=1, zorder=15))
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.axis('off')
        draw_flower_via_emoatlas(axes[0], fav_scores)
        axes[0].set_title(f'Math Lovers\n({target_model}, Qs: {q_title_str})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[1], hat_scores)
        axes[1].set_title(f'Math Haters\n({target_model}, Qs: {q_title_str})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[2], llm_scores)
        axes[2].set_title(f'LLM\n({target_model}, Qs: {q_title_str})', fontsize=14, fontweight='bold')
        plt.suptitle(f'Emotional Profile Comparison: {target_model[:25]}', fontsize=20, y=1.05, fontweight='bold')
        plt.tight_layout()
        if not out_dir:
            plt.show()
        else:
            san_model = target_model.replace('/', '-')
            plt.savefig(out_dir.joinpath(f'{san_model}_Qs_{q_file_str}_Triple.png'), bbox_inches='tight')
    plot_final_plutchik_comparison_triple2(fav_df_agg, hat_df_agg, llm_df_agg, 'Mistral Small 3.2', [1, 2, 3, 7])
    return (List,)


@app.cell
def _(
    List,
    Optional,
    Path,
    emo_draw,
    fav_df_agg,
    hat_df_agg,
    llm_df_agg,
    plt,
    sg,
):
    def plot_final_plutchik_comparison_triple_5(fav_df, hat_df, llm_df, target_model: str, target_questions: List[int], out_dir: Optional[Path]=None):
        fav_rows = fav_df[(fav_df['model_name_x'] == target_model) & fav_df['question_number'].isin(target_questions)]
        hat_rows = hat_df[(hat_df['model_name_x'] == target_model) & hat_df['question_number'].isin(target_questions)]
        llm_rows = llm_df[(llm_df['model_name'] == target_model) & llm_df['question_number'].isin(target_questions)]
        if fav_rows.empty or hat_rows.empty or llm_rows.empty:
            print(f'Data missing for {target_model} Qs: {target_questions}! Skipping...')
            return
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        fav_scores = {emo: fav_rows[f'z_scores_{emo}'].mean() for emo in emotions}
        hat_scores = {emo: hat_rows[f'z_scores_{emo}'].mean() for emo in emotions}
        llm_scores = {emo: llm_rows[f'z_scores_{emo}'].mean() for emo in emotions}
        all_vals = list(fav_scores.values()) + list(hat_scores.values()) + list(llm_scores.values())
        abs_limit = max(abs(min(all_vals)), abs(max(all_vals))) + 0.5
        unified_rescale = (-abs_limit, abs_limit)
        reject = (-1.645, 1.645)
        q_title_str = ', '.join(map(str, target_questions))
        q_file_str = '_'.join(map(str, target_questions))
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        def draw_flower_via_emoatlas(ax, scores):
            mmin = (reject[0] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            mmax = (reject[1] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            reject_circle = sg.Point(0, 0).buffer(mmax)
            ax.add_patch(emo_draw.PolygonPatch(reject_circle, fc='grey', ec=(0.5, 0.5, 0.5, 0.3), alpha=0.1, zorder=-2))
            ax.add_artist(plt.Circle((0, 0), mmax, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            inner_white = sg.Point(0, 0).buffer(mmin)
            ax.add_patch(emo_draw.PolygonPatch(inner_white, fc='white', ec=(0.5, 0.5, 0.5, 0), zorder=-1))
            ax.add_artist(plt.Circle((0, 0), mmin, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            for i in range(0, 110, 20):
                ax.add_artist(plt.Circle((0, 0), 0.15 + i / 100, color='grey', alpha=0.3, fill=False, zorder=-20))
            for emo in emotions:
                emo_draw._draw_emotion_petal(ax=ax, emotion_score=scores[emo], emotion=emo, font='sans-serif', fontweight='light', fontsize=15, highlight='all', show_intensity_levels='none', show_coordinates=True, height_width_ratio=1, reject_range=reject, rescale=unified_rescale)
            center_circle = sg.Point(0, 0).buffer(0.15)
            ax.add_patch(emo_draw.PolygonPatch(center_circle, fc='white', ec=(0.5, 0.5, 0.5, 0.3), alpha=1, zorder=15))
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.axis('off')
        draw_flower_via_emoatlas(axes[0], fav_scores)
        axes[0].set_title(f'Math Lovers\n({target_model}, Qs: {q_title_str})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[1], hat_scores)
        axes[1].set_title(f'Math Haters\n({target_model}, Qs: {q_title_str})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[2], llm_scores)
        axes[2].set_title(f'LLM\n({target_model}, Qs: {q_title_str})', fontsize=14, fontweight='bold')
        plt.suptitle(f'Emotional Profile Comparison: {target_model[:25]}', fontsize=20, y=1.05, fontweight='bold')
        plt.tight_layout()
        if not out_dir:
            plt.show()
        else:
            san_model = target_model.replace('/', '-')
            svg_filename = f'{san_model}_Qs_{q_file_str}_Triple.svg'
            save_path = out_dir.joinpath(svg_filename)
            plt.savefig(save_path, dpi=300, format='svg', bbox_inches='tight')
        plt.close(fig)
    all_models = ['Mistral Small 4', 'DeepSeek Chat', 'grok-4-1-fast-reasoning', 'Qwen3 4B (Thinking)', 'Ministral 3B', 'Mistral Small 3.2', 'Anita 24B (Uncensored)', 'Qwen3 4B (Uncensored)', 'Granite 4 Tiny', 'Qwen3.5 9B', 'Qwen3 4B', 'Ministral 14B (Reasoning)', 'Phi-4 (Reasoning+)', 'Magistral Small']
    target_qs = [1, 2, 3, 7]
    output_folder = Path('./visualizations')
    output_folder.mkdir(parents=True, exist_ok=True)
    for _model in all_models:
        print(f'Generating SVG for: {_model}...')
        plot_final_plutchik_comparison_triple_5(fav_df=fav_df_agg, hat_df=hat_df_agg, llm_df=llm_df_agg, target_model=_model, target_questions=target_qs, out_dir=output_folder)
    print(f"Done! Check the '{output_folder}' directory for your 14 SVG files.")
    return


@app.cell
def _(
    Path,
    QUESTIONS_TO_CONSIDER,
    fav_df_agg,
    hat_df_agg,
    plot_final_plutchik_comparison,
):
    if False:
        flower_dir = Path('./emotional_flowers')
        for _model in fav_df_agg['model_name_x'].unique():
            for question in QUESTIONS_TO_CONSIDER:
                plot_final_plutchik_comparison(fav_df_agg, hat_df_agg, _model, question, flower_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizations
    """)
    return


@app.cell
def _(df):
    _z_score_cols = [col for col in df.columns if col.startswith('z_scores_')]
    df_long = df.melt(id_vars=['model_name', 'question_number'], value_vars=_z_score_cols, var_name='emotion', value_name='z_score')
    df_long['emotion'] = df_long['emotion'].str.replace('z_scores_', '').str.capitalize()
    return (df_long,)


@app.cell
def _(df_long, plt, sns):
    # Create a grid of plots, one for each emotion
    _g = sns.FacetGrid(df_long, col='emotion', col_wrap=4, height=3.5, sharey=True)
    _g.map_dataframe(sns.lineplot, x='question_number', y='z_score', hue='model_name', marker='o')
    # Map a lineplot to each grid, showing question_number vs z_score, colored by model
    _g.add_legend(title='Model Name', bbox_to_anchor=(1.05, 0.5), loc='center left')
    _g.set_axis_labels('Question Number', 'Mean Z-Score')
    # Add a single legend outside the grid
    _g.set_titles(col_template='{col_name}')
    plt.subplots_adjust(top=0.9)
    _g.fig.suptitle('Emotion Z-Scores Across Questions by Model')
    plt.show()
    return


@app.cell
def _(df_long, plt, sns):
    # Create a pivot table suited for a faceted heatmap
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot_table(index=args[1], columns=args[0], values=args[2], aggfunc='median')
        sns.heatmap(d, cmap='RdBu_r', center=0, annot=True, fmt='.2f', cbar=False, **kwargs)  # CHANGED: Use pivot_table instead of pivot, and add aggfunc='mean'
    _g = sns.FacetGrid(df_long, col='model_name', col_wrap=2, height=5, sharex=True, sharey=True)
    _g.map_dataframe(draw_heatmap, 'emotion', 'question_number', 'z_score')
    _g.set_axis_labels('Emotion', 'Question Number')
    _g.set_titles(col_template='{col_name}')
    plt.show()
    return


@app.cell
def _(Optional, Path, fav_df, hat_df, llm_df, plt, sns):
    def plot_emotion_kde_distributions(fav_df, hat_df, llm_df, out_dir: Optional[Path]=None):
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        groups = [('Math Lovers', fav_df, 'model_name_x'), ('Math Haters', hat_df, 'model_name_x'), ('LLMs', llm_df, 'model_name')]
        for group_name, df, model_col in groups:
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            axes = axes.flatten()
            models = df[model_col].dropna().unique()
            for i, emo in enumerate(emotions):
                _ax = axes[i]
                score_col = f'z_scores_{emo}'
                if score_col not in df.columns:
                    print(f'Warning: {score_col} not found in {group_name} dataframe.')
                    continue
                for _model in models:
                    subset = df[df[model_col] == _model]
                    valid_data = subset[score_col].dropna()
                    if not valid_data.empty and len(valid_data) > 1:
                        sns.kdeplot(data=valid_data, ax=_ax, label=_model, linewidth=1.5, fill=False)
                _ax.set_title(f'{emo.capitalize()}', fontsize=16, fontweight='bold')
                _ax.set_xlabel('Z-Score', fontsize=12)
                _ax.set_ylabel('Density', fontsize=12)
                _ax.axvline(0, color='black', linestyle='--', alpha=0.3)
            plt.suptitle(f'Z-Score KDE Distributions by Emotion: {group_name}', fontsize=22, fontweight='bold', y=1.02)
            plt.tight_layout()
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.05), fontsize=12)
            if not out_dir:
                plt.show()
            else:
                safe_name = group_name.replace(' ', '_').lower()
                plt.savefig(out_dir.joinpath(f'KDE_Distributions_{safe_name}.png'), bbox_inches='tight')
            plt.close(fig)
    plot_emotion_kde_distributions(fav_df, hat_df, llm_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TFMN Extracted
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualization For Anxiety
    """)
    return


@app.cell
def _(df_anxiety_zscore):
    df_anxiety_zscore.head()
    return


@app.cell
def _(Optional, Path, df_anxiety_zscore, emo_draw, llm_df_agg, plt, sg):
    def plot_final_plutchik_comparison_triple_6(human_df, llm_df, target_model: str, target_question, out_dir: Optional[Path]=None):
        fav_row = human_df[(human_df['model_name_x'] == target_model) & (human_df['question_number'] == target_question) & (human_df['anxiety_level'] == 'low_anxiety')]
        hat_row = human_df[(human_df['model_name_x'] == target_model) & (human_df['question_number'] == target_question) & (human_df['anxiety_level'] == 'high_anxiety')]
        llm_row = llm_df[(llm_df['model_name'] == target_model) & (llm_df['question_number'] == target_question)]
        if fav_row.empty or hat_row.empty or llm_row.empty:
            return print(f'Data missing for {target_model} Q{target_question}!')
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        fav_scores = {emo: fav_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        hat_scores = {emo: hat_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        llm_scores = {emo: llm_row.iloc[0][f'z_scores_{emo}'] for emo in emotions}
        all_vals = list(fav_scores.values()) + list(hat_scores.values()) + list(llm_scores.values())
        abs_limit = max(abs(min(all_vals)), abs(max(all_vals))) + 0.5
        unified_rescale = (-abs_limit, abs_limit)
        reject = (-1.645, 1.645)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        def draw_flower_via_emoatlas(ax, scores):
            mmin = (reject[0] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            mmax = (reject[1] - unified_rescale[0]) / (unified_rescale[1] - unified_rescale[0]) + 0.15
            reject_circle = sg.Point(0, 0).buffer(mmax)
            ax.add_patch(emo_draw.PolygonPatch(reject_circle, fc='grey', ec=(0.5, 0.5, 0.5, 0.3), alpha=0.1, zorder=-2))
            ax.add_artist(plt.Circle((0, 0), mmax, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            inner_white = sg.Point(0, 0).buffer(mmin)
            ax.add_patch(emo_draw.PolygonPatch(inner_white, fc='white', ec=(0.5, 0.5, 0.5, 0), zorder=-1))
            ax.add_artist(plt.Circle((0, 0), mmin, color='grey', alpha=0.4, fill=False, zorder=-1, linestyle='--'))
            for i in range(0, 110, 20):
                ax.add_artist(plt.Circle((0, 0), 0.15 + i / 100, color='grey', alpha=0.3, fill=False, zorder=-20))
            for emo in emotions:
                emo_draw._draw_emotion_petal(ax=ax, emotion_score=scores[emo], emotion=emo, font='sans-serif', fontweight='light', fontsize=15, highlight='all', show_intensity_levels='none', show_coordinates=True, height_width_ratio=1, reject_range=reject, rescale=unified_rescale)
            center_circle = sg.Point(0, 0).buffer(0.15)
            ax.add_patch(emo_draw.PolygonPatch(center_circle, fc='white', ec=(0.5, 0.5, 0.5, 0.3), alpha=1, zorder=15))
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.axis('off')
        draw_flower_via_emoatlas(axes[0], fav_scores)
        axes[0].set_title(f'Low Anxiety\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[1], hat_scores)
        axes[1].set_title(f'High Anxiety\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        draw_flower_via_emoatlas(axes[2], llm_scores)
        axes[2].set_title(f'LLM\n({target_model}, Q{target_question})', fontsize=14, fontweight='bold')
        plt.suptitle(f'Emotional Profile Comparison: {target_model[:25]} question {target_question}', fontsize=20, y=1.05, fontweight='bold')
        plt.tight_layout()
        if not out_dir:
            plt.show()
        else:
            san_model = target_model.replace('/', '-')
            plt.savefig(out_dir.joinpath(f'{san_model}_Q{target_question}_Triple.png'), bbox_inches='tight')

    # Define the path OUTSIDE the function, with zero indentation
    output_directory = Path(__file__).parent.parent / "figures"

    # Call the function
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'grok-4-1-fast-reasoning', 1, out_dir=output_directory)
    return output_directory, plot_final_plutchik_comparison_triple_6


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'grok-4-1-fast-reasoning', 1)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'grok-4-1-fast-reasoning', 3)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'grok-4-1-fast-reasoning', 7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Anita 24B (Uncensored)
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Anita 24B (Uncensored)', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Granite 4 Tiny
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Granite 4 Tiny', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Magistral Small
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Magistral Small', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ministral 3B
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Ministral 3B', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ministral 14B (Reasoning)
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Ministral 14B (Reasoning)', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mistral Small 3.2
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Mistral Small 3.2', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phi-4 (Reasoning +)
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Phi-4 (Reasoning+)', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3.5 9B
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3.5 9B', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 4B (Uncensored)
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3 4B (Uncensored)', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 4B
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3 4B', 1, out_dir=output_directory)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 4B (Thinking)
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3 4B (Thinking)', 1, out_dir=output_directory)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3 4B (Thinking)', 2)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3 4B (Thinking)', 3)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Qwen3 4B (Thinking)', 7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mistral Small 4
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Mistral Small 4', 1, out_dir=output_directory)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Mistral Small 4', 2)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Mistral Small 4', 3)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'Mistral Small 4', 7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DeepSeek Chat
    """)
    return


@app.cell
def _(
    df_anxiety_zscore,
    llm_df_agg,
    output_directory,
    plot_final_plutchik_comparison_triple_6,
):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'DeepSeek Chat', 2, out_dir=output_directory)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'DeepSeek Chat', 1)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'DeepSeek Chat', 3)
    return


@app.cell
def _(df_anxiety_zscore, llm_df_agg, plot_final_plutchik_comparison_triple_6):
    plot_final_plutchik_comparison_triple_6(df_anxiety_zscore, llm_df_agg, 'DeepSeek Chat', 7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summary Table For Anxiety Levels
    """)
    return


@app.cell
def _(df_anxiety_zscore):
    df_anxiety_zscore.head()
    return


@app.cell
def _(main_table):
    main_table.head()
    return


@app.cell
def _(df_anxiety_zscore):
    df_anxiety_zscore.columns
    return


@app.cell
def _(df_anxiety_zscore):
    # version without question grouping
    main_table = df_anxiety_zscore.groupby(["model_name_x", "question_number", "anxiety_level"]).agg(
        mean_z_score_anger = ("z_scores_anger", "mean"),
        mean_z_score_trust = ("z_scores_trust", "mean"),
        mean_z_score_surprise = ("z_scores_surprise", "mean"),
        mean_z_score_disgust = ("z_scores_disgust", "mean"),
        mean_z_score_joy = ("z_scores_joy", "mean"),
        mean_z_score_sadness = ("z_scores_sadness", "mean"),
        mean_z_score_fear = ("z_scores_fear", "mean"),
        mean_z_score_anticipation = ("z_scores_anticipation", "mean"),
    ).reset_index()
    return (main_table,)


@app.cell
def _(df_anxiety_zscore):
    main_table_gender = df_anxiety_zscore.groupby(["model_name_x", "question_number", "gender_level"]).agg(
        mean_z_score_anger = ("z_scores_anger", "mean"),
        mean_z_score_trust = ("z_scores_trust", "mean"),
        mean_z_score_surprise = ("z_scores_surprise", "mean"),
        mean_z_score_disgust = ("z_scores_disgust", "mean"),
        mean_z_score_joy = ("z_scores_joy", "mean"),
        mean_z_score_sadness = ("z_scores_sadness", "mean"),
        mean_z_score_fear = ("z_scores_fear", "mean"),
        mean_z_score_anticipation = ("z_scores_anticipation", "mean"),
    ).reset_index()
    main_table_gender.head(20)
    return (main_table_gender,)


@app.cell
def _(df_anxiety_zscore):
    main_table_gender_anx = df_anxiety_zscore.groupby(["model_name_x", "question_number", "gender_level", "anxiety_level"]).agg(
        mean_z_score_anger = ("z_scores_anger", "mean"),
        mean_z_score_trust = ("z_scores_trust", "mean"),
        mean_z_score_surprise = ("z_scores_surprise", "mean"),
        mean_z_score_disgust = ("z_scores_disgust", "mean"),
        mean_z_score_joy = ("z_scores_joy", "mean"),
        mean_z_score_sadness = ("z_scores_sadness", "mean"),
        mean_z_score_fear = ("z_scores_fear", "mean"),
        mean_z_score_anticipation = ("z_scores_anticipation", "mean"),
    ).reset_index()
    main_table_gender_anx.head()
    return (main_table_gender_anx,)


@app.cell
def _(main_table):
    main_table
    return


@app.cell
def _(main_table):
    main_table[main_table["question_number"] == 1]
    return


@app.cell
def _(main_table):
    main_table[main_table["question_number"] == 2]
    return


@app.cell
def _(main_table):
    main_table[main_table["question_number"] == 3]
    return


@app.cell
def _(main_table):
    main_table[main_table["question_number"] == 7]
    return


@app.cell
def _(main_table, plt, sns):
    def create_df_viz_diff_z(q):
        # 1. Isolate the data for Question 7 (as shown in your screenshot)
        df_q7 = main_table[main_table["question_number"] == q].copy()

        # 2. Identify all emotion columns dynamically
        emotion_cols = [col for col in df_q7.columns if "mean_z_score_" in col]

        # 3. Melt the data from "wide" to "long" format for Seaborn
        df_long_final = df_q7.melt(
            id_vars=["model_name_x", "anxiety_level"],
            value_vars=emotion_cols,
            var_name="emotion",
            value_name="z_score"
        )

        # 4. Clean up the emotion labels for better presentation (e.g., 'mean_z_score_anger' -> 'Anger')
        df_long_final["emotion"] = df_long_final["emotion"].str.replace("mean_z_score_", "").str.title()
        df_long_final["anxiety_level"] = df_long_final["anxiety_level"].str.replace("_", " ").str.title()

        return df_long_final

    def plot_difference_z(q, func, hue, palette, markers=["o", "s"]):

        df = func(q)

        # 5. Set up the plotting theme
        sns.set_theme(style="whitegrid", context="talk")

        # 6. Create the FacetGrid with categorical dot plots (pointplot) 
        # Pointplots are excellent for showing shifts (deltas) between two states without cluttering the graph with heavy bars.
        _g = sns.catplot(
            data=df,
            x="z_score",
            y="model_name_x",
            hue=hue,
            col="emotion",
            col_wrap=3,          # Adjust depending on how many emotions you have
            kind="point",        # Using points and lines to show the "shift" between low and high
            join=False,          # Do not connect different models with lines
            dodge=0.4,           # Separate the high/low dots slightly on the y-axis
            palette=palette,     # Color-blind friendly palette
            markers=markers,  # Different shapes for accessibility
            height=5,
            aspect=1.2,
            sharex=False         # Allows each emotion's X-axis to scale independently to better show internal variance
        )

        # 7. Polish the aesthetics
        _g.set_titles("{col_name}", weight="bold")
        _g.set_axis_labels("Mean Z-Score", "")
        _g.legend.set_title("Condition")

        # Add a vertical reference line at Z = 0 for every subplot
        for _ax in _g.axes.flat:
            _ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Tweak the overall layout to make room for a main title
        plt.subplots_adjust(top=0.9)
        _g.fig.suptitle(f"Emotion Z-Scores by Model and Anxiety Level (Question {q})", fontsize=20, weight="bold")

        plt.show()

    _palette = {"High Anxiety": "#d95f02", "Low Anxiety": "#1b9e77"}
    plot_difference_z(7, create_df_viz_diff_z, "anxiety_level", _palette)
    return (create_df_viz_diff_z,)


@app.cell
def _(Path, create_df_viz_diff_z, pd, plt, sns):
    def plot_difference_z_new(q, func, hue, palette, title, markers=["o", "s"], path=None):
        df = func(q)

        # 5. Set up the plotting theme
        sns.set_theme(style="whitegrid", context="talk")

        # 6. Create the FacetGrid with categorical dot plots (pointplot) 
        g = sns.catplot(
            data=df,
            x="z_score",
            y="model_name_x",
            hue=hue,
            col="emotion",
            col_wrap=3,          
            kind="point",        
            join=False,          
            dodge=0.4,           
            palette=palette,
            markers=markers,  
            height=5,
            aspect=1.2,
            sharex=False          
        )

        # 7. Polish the aesthetics
        g.set_titles("{col_name}", weight="bold")
        g.set_axis_labels("Mean Z-Score", "")
        g.legend.set_title("Condition")

        # Loop over all subplots to add regions, axes, and lines
        for ax in g.axes.flat:
            # A. Fix the X-axis limits explicitly from -2.6 to 2.6
            ax.set_xlim(-2.6, 2.6)

            # B. Remove default vertical grid lines
            ax.xaxis.grid(False)

            # C. Add the Shaded Rejection Regions
            # Left rejection region (-2.6 to -1.645)
            ax.axvspan(-1.645, 1.645, color='gray', alpha=0.08, zorder=0) # Faint background
            ax.axvspan(-1.645, 1.645, facecolor='none', edgecolor='gray', hatch='///', alpha=0.4, zorder=0) # Shading/hatch

            # Right rejection region (1.645 to 2.6)
           # ax.axvspan(1.645, 2.6, color='gray', alpha=0.08, zorder=0) # Faint background
            #ax.axvspan(1.645, 2.6, facecolor='none', edgecolor='gray', hatch='///', alpha=0.4, zorder=0) # Shading/hatch

            # D. Solid vertical axis at 0
            ax.axvline(0, color='black', linestyle='-', linewidth=1.5, zorder=3)

            # E. Add the dashed lines from 0 to the point
            for line in ax.lines:
                if line.get_marker() in ['o', 's']:
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    marker_color = line.get_color() 

                    for x, y in zip(x_data, y_data):
                        if pd.notna(x) and pd.notna(y):
                            ax.hlines(y=y, xmin=0, xmax=x, color=marker_color, 
                                      linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)

        # Tweak the overall layout to make room for a main title
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"{title} (Question {q})", fontsize=20, weight="bold")

        if path:
            path = Path(path).resolve().absolute()
            plt.savefig(path)
        plt.show()

    _palette = {"High Anxiety": "#d95f02", "Low Anxiety": "#fdae6b"}
    plot_difference_z_new(1, create_df_viz_diff_z, "anxiety_level", _palette, path="code/figures/z_scores_anxiety_levels",title="Emotion Z-Scores by Model and Anxiety Level")
    return (plot_difference_z_new,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization z-scores for Gender
    """)
    return


@app.cell
def _(main_table_gender, plot_difference_z_new):
    def create_df_viz_diff_z_gender(q):
        # 1. Isolate the data for Question 7 (as shown in your screenshot)
        df_q7 = main_table_gender[main_table_gender["question_number"] == q].copy()

        # 2. Identify all emotion columns dynamically
        emotion_cols = [col for col in df_q7.columns if "mean_z_score_" in col]

        # 3. Melt the data from "wide" to "long" format for Seaborn
        df_long_final = df_q7.melt(
            id_vars=["model_name_x", "gender_level"],
            value_vars=emotion_cols,
            var_name="emotion",
            value_name="z_score"
        )

        # 4. Clean up the emotion labels for better presentation (e.g., 'mean_z_score_anger' -> 'Anger')
        df_long_final["emotion"] = df_long_final["emotion"].str.replace("mean_z_score_", "").str.title()
        df_long_final["gender_level"] = df_long_final["gender_level"].str.replace("_", " ").str.title()

        df_long_final = df_long_final[df_long_final["gender_level"].str.lower().isin(["male", "female"])]

        return df_long_final

    _palette = {"Female": "#ff47da", "Male": "#238FFB", "Other": "orange"}
    plot_difference_z_new(1, create_df_viz_diff_z_gender, "gender_level", _palette, path="code/figures/z_scores_gender_levels.png", title="Emotion Z-Scores by Model and Gender Group")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizations z-scores for gender and anxiety level
    """)
    return


@app.cell
def _(main_table_gender_anx, plot_difference_z_new):
    def create_df_viz_diff_z_gender_male(q):
        # 1. Isolate the data for Question 7 (as shown in your screenshot)
        df_q7 = main_table_gender_anx[main_table_gender_anx["question_number"] == q].copy()

        # 2. Identify all emotion columns dynamically
        emotion_cols = [col for col in df_q7.columns if "mean_z_score_" in col]

        # 3. Melt the data from "wide" to "long" format for Seaborn
        df_long_final = df_q7.melt(
            id_vars=["model_name_x", "gender_level", "anxiety_level"],
            value_vars=emotion_cols,
            var_name="emotion",
            value_name="z_score"
        )

        map_legend = {"high_anxiety": "High Anxiety Male", "low_anxiety": "Low Anxiety Male"}
        # 4. Clean up the emotion labels for better presentation (e.g., 'mean_z_score_anger' -> 'Anger')
        df_long_final["emotion"] = df_long_final["emotion"].str.replace("mean_z_score_", "").str.title()
        df_long_final["gender_level"] = df_long_final["gender_level"].str.replace("_", " ").str.title()

        df_long_final = df_long_final[df_long_final["gender_level"].str.lower().isin(["male"])]
        df_long_final["anxiety_level"] = df_long_final["anxiety_level"].replace(map_legend)

        return df_long_final

    _palette = {"High Anxiety Male": "#238FFB", "Low Anxiety Male": "#87C2FD"}
    plot_difference_z_new(1, create_df_viz_diff_z_gender_male, "anxiety_level", _palette, path="code/figures/z_scores_anxiety_male.png", title="Emotion Z-Scores by Model and Anxiety Level for Male Gender")
    return


@app.cell
def _(main_table_gender_anx, plot_difference_z_new):
    def create_df_viz_diff_z_gender_female(q):
        # 1. Isolate the data for Question 7 (as shown in your screenshot)
        df_q7 = main_table_gender_anx[main_table_gender_anx["question_number"] == q].copy()

        # 2. Identify all emotion columns dynamically
        emotion_cols = [col for col in df_q7.columns if "mean_z_score_" in col]

        # 3. Melt the data from "wide" to "long" format for Seaborn
        df_long_final = df_q7.melt(
            id_vars=["model_name_x", "gender_level", "anxiety_level"],
            value_vars=emotion_cols,
            var_name="emotion",
            value_name="z_score"
        )

        map_legend = {"high_anxiety": "High Anxiety Female", "low_anxiety": "Low Anxiety Female"}
        # 4. Clean up the emotion labels for better presentation (e.g., 'mean_z_score_anger' -> 'Anger')
        df_long_final["emotion"] = df_long_final["emotion"].str.replace("mean_z_score_", "").str.title()
        df_long_final["gender_level"] = df_long_final["gender_level"].str.replace("_", " ").str.title()

        df_long_final = df_long_final[df_long_final["gender_level"].str.lower().isin(["female"])]
        df_long_final["anxiety_level"] = df_long_final["anxiety_level"].replace(map_legend)

        return df_long_final

    _palette = {"High Anxiety Female": "#F72684", "Low Anxiety Female": "#FB9DC7"}
    plot_difference_z_new(1, create_df_viz_diff_z_gender_female, "anxiety_level", _palette, path="code/figures/z_scores_anxiety_female.png", title="Emotion Z-Scores by Model and Anxiety Level for Female Gender")
    return


@app.cell
def _(main_table_gender_anx, plot_difference_z_new):
    def create_df_viz_diff_z_gender_high_anx(q):
        # 1. Isolate the data for Question 7 (as shown in your screenshot)
        df_q7 = main_table_gender_anx[main_table_gender_anx["question_number"] == q].copy()

        # 2. Identify all emotion columns dynamically
        emotion_cols = [col for col in df_q7.columns if "mean_z_score_" in col]

        # 3. Melt the data from "wide" to "long" format for Seaborn
        df_long_final = df_q7.melt(
            id_vars=["model_name_x", "gender_level", "anxiety_level"],
            value_vars=emotion_cols,
            var_name="emotion",
            value_name="z_score"
        )

        map_legend = {"Female": "High Anxiety Female", "Male": "High Anxiety Male"}
        # 4. Clean up the emotion labels for better presentation (e.g., 'mean_z_score_anger' -> 'Anger')
        df_long_final["emotion"] = df_long_final["emotion"].str.replace("mean_z_score_", "").str.title()
        df_long_final["gender_level"] = df_long_final["gender_level"].str.replace("_", " ").str.title()

        df_long_final = df_long_final[(df_long_final["gender_level"].str.lower().isin(["male", "female"])) & df_long_final["anxiety_level"].isin(["high_anxiety"])]
        df_long_final["gender_level"] = df_long_final["gender_level"].replace(map_legend)

        return df_long_final

    _palette = {"High Anxiety Male": "#333ebd", "High Anxiety Female": "#FF4080"}
    plot_difference_z_new(1, create_df_viz_diff_z_gender_high_anx, "gender_level", _palette, path="code/figures/High_Anxiety_Gender.png", title="Emotion Z-Scores by Model and Gender Groups for High Anxiety Level")
    return


@app.cell
def _(main_table_gender_anx, plot_difference_z_new):
    def create_df_viz_diff_z_gender_low_anx(q):
        # 1. Isolate the data for Question 7 (as shown in your screenshot)
        df_q7 = main_table_gender_anx[main_table_gender_anx["question_number"] == q].copy()

        # 2. Identify all emotion columns dynamically
        emotion_cols = [col for col in df_q7.columns if "mean_z_score_" in col]

        # 3. Melt the data from "wide" to "long" format for Seaborn
        df_long_final = df_q7.melt(
            id_vars=["model_name_x", "gender_level", "anxiety_level"],
            value_vars=emotion_cols,
            var_name="emotion",
            value_name="z_score"
        )

        map_legend = {"Female": "Low Anxiety Female", "Male": "Low Anxiety Male"}
        # 4. Clean up the emotion labels for better presentation (e.g., 'mean_z_score_anger' -> 'Anger')
        df_long_final["emotion"] = df_long_final["emotion"].str.replace("mean_z_score_", "").str.title()
        df_long_final["gender_level"] = df_long_final["gender_level"].str.replace("_", " ").str.title()

        df_long_final = df_long_final[(df_long_final["gender_level"].str.lower().isin(["male", "female"])) & df_long_final["anxiety_level"].isin(["high_anxiety"])]
        df_long_final["gender_level"] = df_long_final["gender_level"].replace(map_legend)

        return df_long_final

    _palette = {"Low Anxiety Male": "#87C2FD", "Low Anxiety Female": "#FB9DC7"}
    plot_difference_z_new(1, create_df_viz_diff_z_gender_low_anx, "gender_level", _palette, path="code/figures/Low_Anxiety_Gender.png", title="Emotion Z-Scores by Model and Gender Groups for Low Anxiety Level")
    return


@app.cell
def _(main_table, plt, sns):
    _df_q7 = main_table[main_table['question_number'] == 3].copy()
    heatmap_data = _df_q7.set_index(['model_name_x', 'anxiety_level'])
    _z_score_cols = [col for col in heatmap_data.columns if col.startswith('mean_z_score_')]
    heatmap_data = heatmap_data[_z_score_cols]
    heatmap_data.columns = [col.replace('mean_z_score_', '').title() for col in heatmap_data.columns]
    common_args = {'cmap': 'coolwarm', 'vmin': -3, 'vmax': 3, 'annot': False, 'yticklabels': True}
    g_high = sns.clustermap(data=heatmap_data.xs('high_anxiety', level='anxiety_level'), **common_args)
    g_high.fig.suptitle('Emotional Profiles: High Anxiety Condition (Q7)', y=1.05, weight='bold')
    g_low = sns.clustermap(data=heatmap_data.xs('low_anxiety', level='anxiety_level'), **common_args)
    g_low.fig.suptitle('Emotional Profiles: Low Anxiety Condition (Q7)', y=1.05, weight='bold')
    plt.show()
    return


@app.cell
def _(df_anxiety_zscore):
    # version with question grouping
    df_anxiety_zscore.groupby(["model_name_x", "anxiety_level"]).agg(
        mean_z_score_anger = ("z_scores_anger", "mean"),
        mean_z_score_trust = ("z_scores_trust", "mean"),
        mean_z_score_surprise = ("z_scores_surprise", "mean"),
        mean_z_score_disgust = ("z_scores_disgust", "mean"),
        mean_z_score_joy = ("z_scores_joy", "mean"),
        mean_z_score_sadness = ("z_scores_sadness", "mean"),
        mean_z_score_fear = ("z_scores_fear", "mean"),
        mean_z_score_anticipation = ("z_scores_anticipation", "mean"),
    ).reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ridgeline Plots
    """)
    return


@app.cell
def _(Path, df_call2, df_socio_demo, map_if_mseaq, np, pd):
    FOLDER_NAME_MAPPING = {
        'MANX_LLM_anitamistral': 'Anita 24B (Uncensored)',
        'MANX_LLM_DeepSeekLarge': 'DeepSeek Chat',
        'MANX_LLM_granite4h': 'Granite 4 Tiny',
        'MANX_LLM_Grok41FastReasoning': 'Grok 4.1 Fast (Reasoning)',
        'MANX_LLM_magistralsmall': 'Magistral Small',
        'MANX_LLM_ministral3b': 'Ministral 3B',
        'MANX_LLM_ministral14b': 'Ministral 14B (Reasoning)',
        'MANX_LLM_mistralsmall': 'Mistral Small 3.2',
        'MANX_LLM_MistralSmall4': 'Mistral Small 4',
        'MANX_LLM_phi4reasoning': 'Phi-4 (Reasoning+)',
        'MANX_LLM_qwen4bthink': 'Qwen3 4B (Thinking)',
        'MANX_LLM_qwen4bunce': 'Qwen3 4B (Uncensored)',
        'MANX_LLM_qwen34binstruct': 'Qwen3 4B',
        'MANX_LLM_qwen35_9b': 'Qwen3.5 9B'
    }
    call2_path_new = Path("code/Task2/call2_dataset.csv").resolve().absolute()
    df_call2_new = pd.read_csv(call2_path_new)
    df_call2_new["Model"] = df_call2_new["Model"].replace(FOLDER_NAME_MAPPING)
    df_call2_new_humans = df_call2_new[df_call2["mode"] == "human"].merge(df_socio_demo[["run_id", "gender"]], on="run_id")
    df_call2_new_humans["gender"] = np.where(df_call2_new_humans["gender"] == "man", "male", np.where(df_call2_new_humans["gender"] == "woman", "female", "other"))

    mapped_df_call2 = map_if_mseaq(df_call2_new_humans)
    return (mapped_df_call2,)


@app.cell
def _(mapped_df_call2):
    mapped_df_call2.head()
    return


@app.cell
def _(mapped_df_call2):
    df_viz = mapped_df_call2.groupby(["Model", "scale", "gender", "run_id"]).agg(
            sum_of_scores = ("rating", "sum")
        ).reset_index()
    return (df_viz,)


@app.cell
def _():
    # all the items that are part of the anxiety subscale 
    anxiety_items = [str(x) for x in range(8,29)]

    # all the items that are part of the self-efficacy subscale
    self_efficacy_items = [str(x) for x in range(1,8)]
    return anxiety_items, self_efficacy_items


@app.cell
def _(anxiety_items, mapped_df_call2, self_efficacy_items):
    anxiety_filter = (mapped_df_call2["scale"] == "mseaq") & (mapped_df_call2["item number"].astype(str).isin(anxiety_items))
    self_efficacy_filter = (mapped_df_call2["scale"] == "mseaq") & (mapped_df_call2["item number"].astype(str).isin(self_efficacy_items))

    mseaq_anxiety_dataset = mapped_df_call2[anxiety_filter]
    mseaq_efficacy_dataset = mapped_df_call2[self_efficacy_filter]
    return mseaq_anxiety_dataset, mseaq_efficacy_dataset


@app.cell
def _(mseaq_anxiety_dataset, mseaq_efficacy_dataset):
    df_viz_anxiety =  mseaq_anxiety_dataset.groupby(["Model", "scale", "gender", "run_id"]).agg(
        sum_of_scores = ("rating", "sum")
    ).reset_index()

    df_viz_self_efficacy = mseaq_efficacy_dataset.groupby(["Model", "scale", "gender", "run_id"]).agg(
        sum_of_scores = ("rating", "sum")
    ).reset_index()
    return df_viz_anxiety, df_viz_self_efficacy


@app.cell
def _(Path, np, plt):
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde


    color_human = '#87C2FD' 
    color_llm = '#fb6f92'

    def create_custom_ridgeline(
            df, 
            target_scale, 
            scale_col="scale", 
            model_col="Model", 
            mode_col="mode", 
            score_col="rating",
            save_path = None,
            custom_name = None):
            """
            Creates a clean, overlapping ridgeline plot on a single axis using KDEs,
            filtered to show only data for a specific scale.
            """
            # Filter the dataframe for the target scale
            df_filtered = df[df[scale_col] == target_scale].copy()

            if df_filtered.empty:
                print(f"Warning: No data found for scale '{target_scale}'. Check your spelling or dataframe.")
                return

            # Setup and configurations
            models = df_filtered[model_col].unique()
            n_models = len(models)

            # SPACING and HEIGHT_SCALE can be used to control how much the distributions overlap
            SPACING = 0.6
            HEIGHT_SCALE = 1.0 

            fig, ax = plt.subplots(figsize=(12, 6))

            # Define a smooth x-axis range based on your filtered data's min/max
            x_min, x_max = df_filtered[score_col].min(), df_filtered[score_col].max()
            x_smooth = np.linspace(x_min - 0.5, x_max + 0.5, 500)

            # Iterate and plot each model on the same axis
            for i, model in enumerate(models):
                base = i * SPACING

                # z-order: lower rows are drawn last so they overlap the rows behind them
                z_fill = n_models - i
                z_line = z_fill + 0.1

                # --- HUMAN MODE (Blue) ---
                human_data = df_filtered[(df_filtered[model_col] == model) & (df_filtered[mode_col].str.lower() == "male")][score_col].dropna().values

                # Ensure we have enough variance to compute a KDE
                if len(human_data) > 1 and np.var(human_data) > 0:
                    kde_human = gaussian_kde(human_data, bw_method=0.3)
                    y_human = kde_human(x_smooth)
                    y_human = (y_human / y_human.max()) * HEIGHT_SCALE 

                    ax.fill_between(x_smooth, base, base + y_human, color=color_human, alpha=0.85, zorder=z_fill, linewidth=0)
                    ax.plot(x_smooth, base + y_human, color="black", linewidth=1, zorder=z_line)

                # --- LLM MODE (Orange) ---
                llm_data = df_filtered[(df_filtered[model_col] == model) & (df_filtered[mode_col].str.lower() == "female")][score_col].dropna().values

                if len(llm_data) > 1 and np.var(llm_data) > 0:
                    kde_llm = gaussian_kde(llm_data, bw_method=0.3)
                    y_llm = kde_llm(x_smooth)
                    y_llm = (y_llm / y_llm.max()) * HEIGHT_SCALE

                    ax.fill_between(x_smooth, base, base + y_llm, color=color_llm, alpha=0.85, zorder=z_fill - 0.05, linewidth=0)
                    ax.plot(x_smooth, base + y_llm, color="black", linewidth=1, zorder=z_line)

            # Axis formatting
            ax.set_yticks(np.arange(n_models) * SPACING)
            ax.set_yticklabels(models, fontsize=14)

            ax.set_xlabel("Score", fontsize=14)
            # ax.set_ylabel("Models", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)

            # Uncomment this to show title of the scale.
            # ax.set_title(f"Ridgeline Distribution: {str(target_scale).upper()}", fontsize=14, fontweight="bold", pad=15)

            # Remove the bulky borders for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0) 

            # Custom Legend
            legend_handles = [
                Patch(facecolor=color_human, edgecolor="black", label="Male"),
                Patch(facecolor=color_llm, edgecolor="black", label="Female")
            ]

            ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=14)

            plt.tight_layout()

            if save_path:
                if custom_name:
                    path = Path(save_path).resolve().absolute()
                    plt.savefig(path.joinpath(str(custom_name)+"_joyplot.png"), format = "png")

            plt.show()

    return (create_custom_ridgeline,)


@app.cell
def _(create_custom_ridgeline, df_viz):
    create_custom_ridgeline(df_viz, target_scale="amas", scale_col="scale", model_col="Model", mode_col="gender", score_col="sum_of_scores", save_path="code/figures",custom_name="AMAS_gender")
    return


@app.cell
def _(create_custom_ridgeline, df_viz):
    create_custom_ridgeline(df_viz, target_scale="maes", scale_col="scale", model_col="Model", mode_col="gender", score_col="sum_of_scores", save_path="code/figures",custom_name="MAES_gender") 
    return


@app.cell
def _(create_custom_ridgeline, df_viz_anxiety):
    create_custom_ridgeline(df_viz_anxiety, target_scale="mseaq", scale_col="scale", model_col="Model", mode_col="gender", score_col="sum_of_scores",save_path="code/figures",custom_name="MSEAQ_anxiety_gender") 
    return


@app.cell
def _(create_custom_ridgeline, df_viz_self_efficacy):
    create_custom_ridgeline(df_viz_self_efficacy, target_scale="mseaq", scale_col="scale", model_col="Model", mode_col="gender", score_col="sum_of_scores", save_path="code/figures",custom_name="MSEAQ_self_efficacy_gender")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summary Table
    """)
    return


@app.cell
def _(df_viz, pd):
    from scipy.stats import mannwhitneyu

    def generate_gender_stats_table_np(df, scale_col="scale", model_col="Model", mode_col="gender", score_col="sum_of_scores"):
        """
        Generates a summary table of mean differences and Mann-Whitney U statistics 
        split by gender across different models/samples, specifically for the "Amas" scale.
        """
        results = []

        # Filter the DataFrame to ONLY include the "Amas" scale.
        # Using .str.lower() ensures it catches "Amas", "AMAS", or "amas".
        df_amas = df[df[scale_col].str.lower() == "amas"]

        # Iterate through unique samples (e.g., Joint, Russian, Chinese) 
        for model in df_amas[model_col].unique():
            # Iterate through the scale(s) (which will now only be the filtered Amas)
            for scale in df_amas[scale_col].unique():

                # Filter for the specific sample and scale
                subset = df_amas[(df_amas[model_col] == model) & (df_amas[scale_col] == scale)]

                # Extract male and female distributions
                males = subset[subset[mode_col].str.lower() == "male"][score_col].dropna()
                females = subset[subset[mode_col].str.lower() == "female"][score_col].dropna()

                # Skip if data is missing for either group
                if len(males) == 0 or len(females) == 0:
                    continue

                # Calculate descriptive statistics
                mean_males = males.mean()
                mean_females = females.mean()
                diff = mean_males - mean_females

                # Perform Mann-Whitney U test (Non-parametric equivalent for 2 groups)
                u_stat, p_val = mannwhitneyu(males, females, alternative='two-sided')

                # Format p-value with significance markers
                if p_val < 0.001:
                    p_val_str = "<0.001**"
                elif p_val < 0.01:
                    p_val_str = f"{p_val:.3f}**"
                elif p_val < 0.05:
                    p_val_str = f"{p_val:.3f}*"
                else:
                    p_val_str = f"{p_val:.3f}"

                results.append({
                    "Sample": model,
                    "Scale": scale,
                    "Males": round(mean_males, 2),
                    "Females": round(mean_females, 2),
                    "Difference": round(diff, 2),
                    "U-statistic": round(u_stat, 2),
                    "p-value": p_val_str
                })

        # Compile into a DataFrame
        stats_df = pd.DataFrame(results)

        # Apply a MultiIndex to replicate the grouped visual layout
        if not stats_df.empty:
            stats_df = stats_df.set_index(["Sample", "Scale"])

        return stats_df

    # Example usage:
    summary_table_np = generate_gender_stats_table_np(df_viz)
    summary_table_np
    return (summary_table_np,)


@app.cell
def _(summary_table_np):
    print(summary_table_np.to_latex(index=True,
                      formatters={"name": str.upper},
                      float_format="{:.1f}".format,
                      ))
    return


if __name__ == "__main__":
    app.run()
