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
    matplotlib.use('Agg')  # Force non-interactive backend
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from scipy import stats
    from pathlib import Path

    return Path, np, pd, plt


@app.cell
def _():
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
    return (FOLDER_NAME_MAPPING,)


@app.cell
def _(Path, pd):
    aggregated_file = pd.read_csv(Path("code/Task3/aggregated_wide_format_dataset/NEW_aggregated_final_wide_format_dataset.csv").resolve())
    return (aggregated_file,)


@app.cell
def _(FOLDER_NAME_MAPPING, aggregated_file):
    aggregated_file["model_name"] = aggregated_file["model_name"].replace(FOLDER_NAME_MAPPING)
    return


@app.cell
def _(aggregated_file):
    aggregated_file.head()
    return


@app.cell
def _(Path, pd):
    socio_demo = pd.read_csv(Path("code/Task1/individual_tfmn/persona_dataset.csv").resolve())
    return (socio_demo,)


@app.cell
def _(socio_demo):
    socio_demo.head()
    return


@app.cell
def _(Path, np, pd, socio_demo):
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

    persona_merged = socio_demo.merge(pivoted_dataset, on="run_id", how="inner")

    persona_merged["anxiety_score"] = persona_merged["amas_score"] + persona_merged["mseaq_score"]

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
def _(aggregated_file, persona_merged_1):
    aggregated_new = aggregated_file.merge(persona_merged_1[["run_id", "gender", "anxiety_level"]], on="run_id")
    return (aggregated_new,)


@app.cell
def _(aggregated_file):
    filtered = aggregated_file["cue_word"]=="mathematic"
    filtered_dataset = aggregated_file[filtered]
    new_data=filtered_dataset.groupby(["model_name", "cue_word"])
    return filtered_dataset, new_data


@app.cell
def _(new_data):
    _proportions = new_data.value_counts(normalize=True)
    _proportions
    return


@app.cell
def _(filtered_dataset):
    # 3. Calculate the overall proportions (multiplying by 100 for percentages)
    overall_proportions = filtered_dataset['valence_cue_word'].value_counts(normalize=True) * 100
    print('--- Overall Proportions ---')
    print(overall_proportions)
    _model_proportions = filtered_dataset.groupby('model_name')['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    print('\n--- Proportions by Model ---')
    # 4. Calculate proportions per individual model
    print(_model_proportions)
    return


@app.cell
def _(filtered_dataset, plt):
    # 3. Calculate proportions per individual model
    # unstack(fill_value=0) creates a nice grid, and * 100 converts to percentages
    _model_proportions = filtered_dataset.groupby('model_name')['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _model_proportions = _model_proportions.sort_index()
    # Sort the index (models) alphabetically so the bars are in a reliable sorted order
    _ax = _model_proportions.plot(kind='bar', figsize=(10, 6), color=['#fe1f1f', '#a3a3a3', '#1b84fe'], edgecolor='black')
    plt.title('Proportions of Valence Scores for "mathematic" by Model', pad=20)
    # 4. Generate the bar plot
    # We use pandas built-in plotting, passing color hex codes for Negative, Neutral, and Positive
    plt.xlabel('Model Name', labelpad=15)
    plt.ylabel('Proportion (%)', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Valence Score', labels=['Negative (-1)', 'Neutral (0)', 'Positive (1)'])
    # 5. Format the plot ensuring labels are readable
    # Rotate x-axis labels by 45 degrees so the long model names don't overlap
    # Add a descriptive legend
    #plt.tight_layout() #ensures everything fits without getting cropped

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(aggregated_new, plt):
    df_1 = aggregated_new
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[df_1['cue_word'].isin(_target_words)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
        _axes[_i].set_title(f'"{_word.capitalize()}" Valence by Model', pad=15)
        _axes[_i].set_xlabel('Model Name', labelpad=10)
        _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].set_ylabel('Proportion (%)', labelpad=10)
    _axes[1].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.close('all')
    return (df_1,)


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    male_condition = df_1["gender"] == "man"
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & male_condition)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
    
        # --- ADDED THIS LINE ---
        _axes[_i].set_facecolor('#caf0f8') 
    
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_male.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    female_condition = df_1["gender"] == "woman"
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & female_condition)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
    
        # --- ADDED THIS LINE ---
        _axes[_i].set_facecolor('#ffe5ec')
    
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_female.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    high_anx_condition = df_1["anxiety_level"] == "high_anxiety"
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & high_anx_condition)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
    
        # --- ADDED THIS LINE ---
        _axes[_i].set_facecolor('#ffba78')
    
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_high_anx.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    low_anx_condition = df_1["anxiety_level"] == "low_anxiety"
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & low_anx_condition)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
    
        # --- ADDED THIS LINE ---
        _axes[_i].set_facecolor('#ffd991')
    
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_low_anx.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    low_anx_condition_woman = (df_1["anxiety_level"] == "low_anxiety") & (df_1["gender"] == "woman")
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & low_anx_condition_woman)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_low_anx_woman.png').resolve(), dpi=300, format='png')
    plt.gca()
    return (low_anx_condition_woman,)


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    low_anx_condition_man = (df_1["anxiety_level"] == "low_anxiety") & (df_1["gender"] == "man")
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & low_anx_condition_man)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_low_anx_man.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell
def _(Path, df_1, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    high_anx_condition_woman = (df_1["anxiety_level"] == "high_anxiety") & (df_1["gender"] == "woman")
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & high_anx_condition_woman)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_high_anx_woman.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell
def _(Path, df_1, low_anx_condition_woman, plt):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold'})
    high_anx_condition_man = (df_1["anxiety_level"] == "high_anxiety") & (df_1["gender"] == "man")
    _target_words = ['mathematic', 'science']
    _filtered_df = df_1[(df_1['cue_word'].isin(_target_words) & low_anx_condition_woman)]
    _proportions = _filtered_df.groupby(['model_name', 'cue_word'])['valence_cue_word'].value_counts(normalize=True).unstack(fill_value=0) * 100
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    _colors = ['#fe1f1f', '#a3a3a3', '#1b84fe']
    _labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    for _i, _word in enumerate(_target_words):
        _word_data = _proportions.xs(_word, level='cue_word')
        _word_data.plot(kind='bar', ax=_axes[_i], color=_colors, edgecolor='black', legend=False)
        _axes[_i].grid(False)
        _axes[_i].text(0.5, 0.95, f'"{_word.capitalize()}" Valence by Model', transform=_axes[_i].transAxes, fontsize=14, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        _axes[_i].set_ylabel('Proportion (%)', labelpad=10)
        _axes[_i].set_xlabel('')
        if _i == 0:
            _axes[_i].set_xticklabels([])
        else:
            _axes[_i].set_xticklabels(_word_data.index, rotation=45, ha='right')
    _axes[0].legend(title='Valence Score', labels=_labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(Path('code/Task1/visualizations/Keywords_valence_comparison_horizontal_final_high_anx_man.png').resolve(), dpi=300, format='png')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Valences Auras
    """)
    return


@app.cell
def _(FOLDER_NAME_MAPPING, Path, aggregated_file, pd, plt):
    import seaborn as sns
    df_wide = aggregated_file
    df_aura = pd.read_csv(Path('code/Task3/auras/aggregated_aura_valences.csv').resolve())
    df_aura['model_name'] = df_aura['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    _agg_wide = df_wide.groupby(['model_name', 'cue_word'])['valence_cue_word'].mean().reset_index()
    merged = pd.merge(_agg_wide, df_aura, on=['model_name', 'cue_word'], how='inner')
    plt.figure(figsize=(14, 10))
    _ax = sns.scatterplot(data=merged, x='valence_cue_word', y='aura_net_valence', hue='model_name', style='cue_word', s=300, palette='Set1', edgecolor='black', alpha=0.9)
    t_neg, t_pos = (-0.05, 0.05)
    plt.axvline(t_neg, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(t_pos, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(t_neg, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(t_pos, color='gray', linestyle='--', alpha=0.5)
    _ax.set_xlim(-0.3, 0.7)
    _ax.set_ylim(-0.3, 0.7)
    regions = {'Neg Keyword\nPos Aura': (-0.175, 0.325), 'Neu Keyword\nPos Aura': (0, 0.325), 'Pos Keyword\nPos Aura': (0.375, 0.325), 'Neg Keyword\nNeu Aura': (-0.175, 0), 'Neu Keyword\nNeu Aura': (0, 0), 'Pos Keyword\nNeu Aura': (0.375, 0), 'Neg Keyword\nNeg Aura': (-0.175, -0.175), 'Neu Keyword\nNeg Aura': (0, -0.175), 'Pos Keyword\nNeg Aura': (0.375, -0.175)}
    for text, (x, y) in regions.items():
        plt.text(x, y, text, alpha=0.15, fontsize=16, ha='center', va='center', weight='bold')
    plt.title('3x3 Matrix: Keyword Valence vs. Aura Valence\n(Mapping Models and Concepts across 9 Behavioral Cases)', pad=20, fontsize=16)
    plt.xlabel('Keyword Valence (Mean Score)', fontsize=14)
    plt.ylabel('Aura Net Valence (Positive Fraction - Negative Fraction)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Models & Concepts', fontsize=12)
    plt.tight_layout()
    plt.savefig('./visualizations/Keywords_Aura_comparison_matrix.svg', dpi=300, format='svg')
    plt.gca()
    return (sns,)


@app.cell
def _(FOLDER_NAME_MAPPING, Path, aggregated_file, pd, plt, sns):
    df_wide_2 = aggregated_file
    df_aura_2 = pd.read_csv(Path('code/Task3/auras/aggregated_aura_valences.csv').resolve())
    df_aura_2['model_name'] = df_aura_2['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    _agg_wide = df_wide_2.groupby(['model_name', 'cue_word'])['valence_cue_word'].mean().reset_index()
    merged_2 = pd.merge(_agg_wide, df_aura_2, on=['model_name', 'cue_word'], how='inner')
    merged_2['model_name_short'] = merged_2['model_name'].str.replace('MANX_LLM_', '')
    merged_2['cue_word_disp'] = merged_2['cue_word'].str.capitalize()
    _pivot_kw = merged_2.pivot(index='model_name_short', columns='cue_word_disp', values='valence_cue_word')
    _pivot_aura = merged_2.pivot(index='model_name_short', columns='cue_word_disp', values='aura_net_valence')
    _pivot_kw = _pivot_kw.sort_index(axis=0).sort_index(axis=1)
    _pivot_aura = _pivot_aura.sort_index(axis=0).sort_index(axis=1)
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    _cmap = sns.diverging_palette(10, 240, as_cmap=True)
    _vmin = min(_pivot_kw.min().min(), _pivot_aura.min().min())
    _vmax = max(_pivot_kw.max().max(), _pivot_aura.max().max())
    _limit = max(abs(_vmin), abs(_vmax))
    _limit = round(_limit + 0.05, 2)
    sns.heatmap(_pivot_kw, ax=_axes[0], cmap=_cmap, center=0, vmin=-_limit, vmax=_limit, annot=True, fmt='.3f', cbar_kws={'label': 'Mean Valence'}, linewidths=0.5)
    _axes[0].set_title('Mean Keyword Valence', fontsize=14, pad=15)
    _axes[0].set_xlabel('Cue Word', fontsize=12)
    _axes[0].set_ylabel('Model', fontsize=12)
    _axes[0].tick_params(axis='y', rotation=0)
    sns.heatmap(_pivot_aura, ax=_axes[1], cmap=_cmap, center=0, vmin=-_limit, vmax=_limit, annot=True, fmt='.3f', cbar_kws={'label': 'Net Aura Valence'}, linewidths=0.5)
    _axes[1].set_title('Mean Aura Net Valence', fontsize=14, pad=15)
    _axes[1].set_xlabel('Cue Word', fontsize=12)
    _axes[1].set_ylabel('')
    plt.suptitle('Comparison: Keyword Valence vs. Aura Valence by Model', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('./visualizations/Keywords_Aura_comparison_heatmap.svg', dpi=300, format='svg')
    plt.gca()
    #print('Plot successfully generated and saved!')
    return


@app.cell
def _(FOLDER_NAME_MAPPING, Path, aggregated_file, pd, plt, sns):
    df_wide_3 = aggregated_file
    df_aura_3 = pd.read_csv(Path('code/Task3/auras/aggregated_aura_valences.csv').resolve())
    _subjects = ['mathematic', 'science']
    df_aura_3['model_name'] = df_aura_3['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    df_aura_3 = df_aura_3[df_aura_3['cue_word'].isin(_subjects)]
    _agg_wide = df_wide_3.groupby(['model_name', 'cue_word'])['valence_cue_word'].mean().reset_index()
    merged_3 = pd.merge(_agg_wide, df_aura_3, on=['model_name', 'cue_word'], how='inner')
    merged_3['model_name_short'] = merged_3['model_name'].str.replace('MANX_LLM_', '')
    merged_3['cue_word_disp'] = merged_3['cue_word'].str.capitalize()
    _pivot_kw = merged_3.pivot(index='cue_word_disp', columns='model_name_short', values='valence_cue_word')
    _pivot_aura = merged_3.pivot(index='cue_word_disp', columns='model_name_short', values='aura_net_valence')
    _pivot_kw = _pivot_kw.sort_index(axis=0).sort_index(axis=1)
    _pivot_aura = _pivot_aura.sort_index(axis=0).sort_index(axis=1)
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    _cmap = sns.diverging_palette(10, 240, as_cmap=True)
    _vmin = min(_pivot_kw.min().min(), _pivot_aura.min().min())
    _vmax = max(_pivot_kw.max().max(), _pivot_aura.max().max())
    _limit = max(abs(_vmin), abs(_vmax))
    _limit = round(_limit + 0.05, 2)
    sns.heatmap(_pivot_kw, ax=_axes[0], cmap=_cmap, center=0, vmin=-_limit, vmax=_limit, annot=True, fmt='.3f', cbar_kws={'label': 'Mean Valence'}, linewidths=0.5)
    _axes[0].set_title('Mean Keyword Valence', fontsize=14, pad=15)
    _axes[0].set_ylabel('Cue Word', fontsize=12)
    _axes[0].set_xlabel('')
    _axes[0].tick_params(axis='y', rotation=0)
    sns.heatmap(_pivot_aura, ax=_axes[1], cmap=_cmap, center=0, vmin=-_limit, vmax=_limit, annot=True, fmt='.3f', cbar_kws={'label': 'Net Aura Valence'}, linewidths=0.5)
    _axes[1].set_title('Mean Aura Net Valence', fontsize=14, pad=15)
    _axes[1].set_ylabel('Cue Word', fontsize=12)
    _axes[1].set_xlabel('Model', fontsize=12)
    _axes[1].tick_params(axis='y', rotation=0)
    _axes[1].set_xticklabels(_axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.suptitle('Comparison: Keyword Valence vs. Aura Valence by Model', fontsize=16, y=1.02)
    plt.tight_layout()
    #plt.savefig('./visualizations/Keywords_Aura_comparison_heatmap_horizontal.svg', dpi=300, format='svg')
    plt.gca()
    #print('Plot successfully generated and saved!')
    return df_aura_3, df_wide_3, merged_3


@app.cell
def _(df_wide_3):
    df_wide_3.head()
    return


@app.cell
def _(FOLDER_NAME_MAPPING, Path, aggregated_file, pd, plt, sns, socio_demo):
    _df_wide_4 = aggregated_file
    _df_aura_4 = pd.read_csv(Path('code/Task3/auras/aggregated_aura_valences.csv').resolve())
    _subjects = ['mathematic', 'science']
    _df_aura_4['model_name'] = _df_aura_4['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    # Add gender, filter by man
    _df_wide_4 = _df_wide_4.merge(socio_demo[["run_id", "gender"]], on="run_id").query("gender == 'man'")
    # print(_df_wide_4.head())
    _df_aura_4 = _df_aura_4[_df_aura_4['cue_word'].isin(_subjects)]
    # df_wide_4 filtering propagates here
    _agg_wide = _df_wide_4.groupby(['model_name', 'cue_word'])['valence_cue_word'].mean().reset_index()
    # however due to how merging is performed here this filtering stops. 
    _merged_4 = pd.merge(_agg_wide, _df_aura_4, on=['model_name', 'cue_word'], how='inner')
    _merged_4['model_name_short'] = _merged_4['model_name'].str.replace('MANX_LLM_', '')
    _merged_4['cue_word_disp'] = _merged_4['cue_word'].str.capitalize()
    _pivot_kw = _merged_4.pivot(index='cue_word_disp', columns='model_name_short', values='valence_cue_word')
    _pivot_aura = _merged_4.pivot(index='cue_word_disp', columns='model_name_short', values='aura_net_valence')
    _pivot_kw = _pivot_kw.sort_index(axis=0).sort_index(axis=1)
    _pivot_aura = _pivot_aura.sort_index(axis=0).sort_index(axis=1)
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    _cmap = sns.diverging_palette(10, 240, as_cmap=True)
    _vmin = min(_pivot_kw.min().min(), _pivot_aura.min().min())
    _vmax = max(_pivot_kw.max().max(), _pivot_aura.max().max())
    _limit = max(abs(_vmin), abs(_vmax))
    _limit = round(_limit + 0.05, 2)
    sns.heatmap(_pivot_kw, ax=_axes[0], cmap=_cmap, center=0, vmin=-_limit, vmax=_limit, annot=True, fmt='.3f', cbar_kws={'label': 'Mean Valence'}, linewidths=0.5)
    _axes[0].set_title('Mean Keyword Valence', fontsize=14, pad=15)
    _axes[0].set_ylabel('Cue Word', fontsize=12)
    _axes[0].set_xlabel('')
    _axes[0].tick_params(axis='y', rotation=0)
    sns.heatmap(_pivot_aura, ax=_axes[1], cmap=_cmap, center=0, vmin=-_limit, vmax=_limit, annot=True, fmt='.3f', cbar_kws={'label': 'Net Aura Valence'}, linewidths=0.5)
    _axes[1].set_title('Mean Aura Net Valence', fontsize=14, pad=15)
    _axes[1].set_ylabel('Cue Word', fontsize=12)
    _axes[1].set_xlabel('Model', fontsize=12)
    _axes[1].tick_params(axis='y', rotation=0)
    _axes[1].set_xticklabels(_axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.suptitle('Comparison: Keyword Valence vs. Aura Valence by Model', fontsize=16, y=1.02)
    #plt.tight_layout()
    #plt.savefig('./visualizations/Keywords_Aura_comparison_heatmap_horizontal.svg', dpi=300, format='svg')
    plt.gca()
    #print('Plot successfully generated and saved!')
    return


@app.cell
def _(df_aura_3):
    df_aura_3.head()
    return


@app.cell
def _(df_wide_3):
    df_wide_3.head()
    return


@app.cell
def _(df_aura_3):
    df_aura_3.head()
    return


@app.cell
def _(merged_3):
    merged_3.head()
    return


@app.cell
def _(FOLDER_NAME_MAPPING, Path, aggregated_file, mo, np, pd, plt):
    import math
    df_wide_4 = aggregated_file
    df_aura_4 = pd.read_csv(Path('code/Task3/auras/run_level_aura_valences.csv').resolve())
    _subjects = ['mathematic', 'science']
    df_aura_4['model_name_short'] = df_aura_4['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    df_wide_4['model_name_short'] = df_wide_4['model_name'].str.replace('MANX_LLM_', '').replace(FOLDER_NAME_MAPPING)
    merged_4 = pd.merge(df_wide_4, df_aura_4, on=['model_name_short', 'cue_word', 'run_id'], how='inner')
    merged_4 = merged_4[merged_4['cue_word'].isin(_subjects)]

    def _categorize_valence(val):
        if val > 0:
            return 'P'
        elif val < 0:
            return 'N'
        else:
            return 'Ne'
    merged_4['kw_cat'] = merged_4['valence_cue_word'].apply(_categorize_valence)
    merged_4['aura_cat'] = merged_4['aura_net_valence'].apply(_categorize_valence)
    merged_4['valence_pair'] = merged_4['kw_cat'] + merged_4['aura_cat']
    _categories = ['NN', 'NP', 'NNe', 'PN', 'PP', 'PNe', 'NeN', 'NeP', 'NeNe']
    _colors = ['#f47f7f', '#7cbdec', '#a6a6a6'] * 3

    def _plot_keyword_combinations(df, keyword, row_layout=[5, 5, 4]):
        """
    # Combine them into the 9 categories (e.g., 'N' + 'N' = 'NN')
        Plots the combinations. 
        row_layout controls exactly how many models appear on each line.
    # The exact 9 categories from the image in order
        """
        df_kw = df[df['cue_word'] == keyword]
        _proportions = df_kw.groupby('model_name_short')['valence_pair'].value_counts(normalize=True).unstack(fill_value=0)
        for cat in _categories:
            if cat not in _proportions.columns:
                _proportions[cat] = 0
        _proportions = _proportions[_categories]
        models = sorted(_proportions.index.tolist())
        models_rows = []
        start_idx = 0
        for count in row_layout:
            end_idx = start_idx + count
            chunk = models[start_idx:end_idx]
            if chunk:
                models_rows.append(chunk)
            start_idx = end_idx
        num_rows = len(models_rows)
        print(models_rows)
        max_models_in_row = max((len(r) for r in models_rows))
        max_x_limit = max_models_in_row * (len(_categories) + 3)
        _fig, _axes = plt.subplots(num_rows, 1, figsize=(18, 3.5 * num_rows), sharey=True)
        if num_rows == 1:
            _axes = [_axes]
        for _i, _ax in enumerate(_axes):
            current_models = models_rows[_i]
            major_ticks = []
            minor_ticks = []
            minor_labels = []
            for j, model in enumerate(current_models):
                model_data = _proportions.loc[model]
                x_start = j * (len(_categories) + 3)
                x_pos = np.arange(x_start, x_start + len(_categories))
                for k, cat in enumerate(_categories):
                    _ax.bar(x_pos[k], model_data[cat], width=1.0, color=_colors[k], edgecolor='white', linewidth=0.5)
                minor_ticks.extend(x_pos)
                minor_labels.extend(_categories)
                major_ticks.append(x_start + len(_categories) / 2 - 0.5)
            _ax.set_xticks(minor_ticks, minor=True)
            _ax.set_xticklabels(minor_labels, minor=True, fontsize=12, rotation=90, ha='right', rotation_mode='anchor')
            _ax.tick_params(axis='x', which='minor', length=0)
            _ax.set_xticks(major_ticks)
            _ax.set_xticklabels(current_models, fontsize=16, fontweight='bold')
            _ax.tick_params(axis='x', which='major', pad=30, length=0)
            _ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=14)
            _ax.set_ylabel('% of Words', fontsize=16)
            _ax.set_xlim(-1.5, max_x_limit - 1.5)
            if _i == 0:
                _ax.text(0.5, 0.95, f'"{keyword.capitalize()}" Valence by Model', transform=_ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        plt.tight_layout(h_pad=0.5)
        plt.savefig(f'./visualizations/{keyword}_Aura_comparison.svg', dpi=300, format='svg')
        return _fig

    # 1. Create a list to store the generated figures
    generated_figs = []

    for _kw in _subjects:
        # 2. Append the returned figure to your list
        _fig = _plot_keyword_combinations(merged_4, _kw, row_layout=[5, 5, 4])
        generated_figs.append(_fig)

    # 3. Stack and display the figures as the final expression in the cell
    mo.vstack(generated_figs)
    return df_wide_4, merged_4


@app.cell
def _(FOLDER_NAME_MAPPING, df_wide_4, merged_4, mo, np, pd, plt):
    import matplotlib.patches as mpatches
    plt.rcParams.update({'font.size': 20, 'axes.labelweight': 'bold', 'font.weight': 'bold'})
    df_aura_5 = pd.read_csv('/Users/luisaporzio/Projects/teachme-project/src/data_validation/aggregated/run_level_aura_valences.csv')
    _subjects = ['mathematic', 'science']
    df_aura_5['model_name_short'] = df_aura_5['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    df_wide_4['model_name_short'] = df_wide_4['model_name'].str.replace('MANX_LLM_', '').replace(FOLDER_NAME_MAPPING)
    merged_5 = pd.merge(df_wide_4, df_aura_5, on=['model_name_short', 'cue_word', 'run_id'], how='inner')
    merged_5 = merged_5[merged_5['cue_word'].isin(_subjects)]

    def _categorize_valence(val):
        if val > 0:
            return 'P'
        elif val < 0:
            return 'N'
        else:
            return 'Ne'
    merged_5['kw_cat'] = merged_5['valence_cue_word'].apply(_categorize_valence)
    merged_5['aura_cat'] = merged_5['aura_net_valence'].apply(_categorize_valence)
    merged_5['valence_pair'] = merged_5['kw_cat'] + merged_5['aura_cat']
    _categories = ['NN', 'NP', 'NNe', 'PN', 'PP', 'PNe', 'NeN', 'NeP', 'NeNe']
    _colors = ['#f47f7f', '#7cbdec', '#a6a6a6'] * 3

    def _plot_keyword_combinations(df, keyword, row_layout=[5, 5, 4]):
        df_kw = df[df['cue_word'] == keyword]
        _proportions = df_kw.groupby('model_name_short')['valence_pair'].value_counts(normalize=True).unstack(fill_value=0)
        for cat in _categories:
            if cat not in _proportions.columns:
                _proportions[cat] = 0
        _proportions = _proportions[_categories]
        models = sorted(_proportions.index.tolist())
        models_rows = []
        start_idx = 0
        for count in row_layout:
            end_idx = start_idx + count
            chunk = models[start_idx:end_idx]
            if chunk:
                models_rows.append(chunk)
            start_idx = end_idx
        num_rows = len(models_rows)
        max_models_in_row = max((len(r) for r in models_rows))
        max_x_limit = max_models_in_row * (len(_categories) + 3)
        _fig, _axes = plt.subplots(num_rows, 1, figsize=(18, 3.5 * num_rows), sharey=True)
        if num_rows == 1:
            _axes = [_axes]
        for _i, _ax in enumerate(_axes):
            _ax.grid(False)
            current_models = models_rows[_i]
            category_ticks = []
            category_labels = []
            for j, model in enumerate(current_models):
                model_data = _proportions.loc[model]
                x_start = j * (len(_categories) + 3)
                x_pos = np.arange(x_start, x_start + len(_categories))
                for k, cat in enumerate(_categories):
                    _ax.bar(x_pos[k], model_data[cat], width=1.0, color=_colors[k], edgecolor='white', linewidth=0.5)
                category_ticks.extend(x_pos)
                category_labels.extend(_categories)
                model_center = x_start + len(_categories) / 2 - 0.5
                _ax.text(model_center, 0.9, model, fontsize=17, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            _ax.set_xticks(category_ticks)
            if _i == 0:
                _ax.set_xticklabels([])
            else:
                _ax.set_xticklabels(category_labels, fontsize=16, rotation=90, ha='center', va='top')
            _ax.tick_params(axis='x', which='major', length=0)
            _ax.tick_params(axis='y', which='major', direction='in', right=True, labelsize=16)
            _ax.set_ylabel('% of Words', fontsize=20, fontweight='bold')
            _ax.set_xlim(-1.5, max_x_limit - 1.5)
            _ax.set_ylim(0, 1.0)
            if _i == 0:
                _ax.text(0.5, 1.05, f'"{keyword.capitalize()}" Valence by Model', transform=_ax.transAxes, fontsize=20, fontweight='bold', va='bottom', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        legend_patches = [mpatches.Patch(color='#f47f7f', label='Negative (N)'), mpatches.Patch(color='#7cbdec', label='Positive (P)'), mpatches.Patch(color='#a6a6a6', label='Neutral (Ne)')]
        leg = _axes[-1].legend(handles=legend_patches, loc='lower right', title='Aura Valence', prop={'size': 20, 'weight': 'bold'})
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(20)
        plt.tight_layout(h_pad=0.5)
        if num_rows > 1:
            _fig.canvas.draw()
            pos = _axes[0].get_position()
            _axes[0].set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height])
        return _fig

    # 1. Create a list to store the generated figures
    _generated_figs = []

    for _kw in _subjects:
        # 2. Append the returned figure to your list
        _fig = _plot_keyword_combinations(merged_4, _kw, row_layout=[5, 5, 4])
        _generated_figs.append(_fig)

    # 3. Stack and display the figures as the final expression in the cell
    mo.vstack(_generated_figs)
    return (mpatches,)


@app.cell
def _(FOLDER_NAME_MAPPING, df_wide_4, merged_4, mo, mpatches, np, pd, plt):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 20, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20, 'legend.title_fontsize': 20})
    df_aura_6 = pd.read_csv('/Users/luisaporzio/Projects/teachme-project/src/data_validation/aggregated/run_level_aura_valences.csv')
    _subjects = ['mathematic', 'science']
    df_aura_6['model_name_short'] = df_aura_6['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    df_wide_4['model_name_short'] = df_wide_4['model_name'].str.replace('MANX_LLM_', '').replace(FOLDER_NAME_MAPPING)
    merged_6 = pd.merge(df_wide_4, df_aura_6, on=['model_name_short', 'cue_word', 'run_id'], how='inner')
    merged_6 = merged_6[merged_6['cue_word'].isin(_subjects)]

    def _categorize_valence(val):
        if val > 0:
            return 'P'
        elif val < 0:
            return 'N'
        else:
            return 'Ne'
    merged_6['kw_cat'] = merged_6['valence_cue_word'].apply(_categorize_valence)
    merged_6['aura_cat'] = merged_6['aura_net_valence'].apply(_categorize_valence)
    merged_6['valence_pair'] = merged_6['kw_cat'] + merged_6['aura_cat']
    _categories = ['NN', 'NP', 'NNe', 'PN', 'PP', 'PNe', 'NeN', 'NeP', 'NeNe']
    _colors = ['#fe1f1f', '#1b84fe', '#a3a3a3'] * 3

    def _plot_keyword_combinations(df, keyword, row_layout=[5, 5, 4]):
        df_kw = df[df['cue_word'] == keyword]
        _proportions = df_kw.groupby('model_name_short')['valence_pair'].value_counts(normalize=True).unstack(fill_value=0)
        for cat in _categories:
            if cat not in _proportions.columns:
                _proportions[cat] = 0
        _proportions = _proportions[_categories]
        models = sorted(_proportions.index.tolist())
        models_rows = []
        start_idx = 0
        for count in row_layout:
            end_idx = start_idx + count
            chunk = models[start_idx:end_idx]
            if chunk:
                models_rows.append(chunk)
            start_idx = end_idx
        num_rows = len(models_rows)
        max_models_in_row = max((len(r) for r in models_rows))
        max_x_limit = max_models_in_row * (len(_categories) + 3)
        _fig, _axes = plt.subplots(num_rows, 1, figsize=(18, 3.5 * num_rows), sharey=True)
        if num_rows == 1:
            _axes = [_axes]
        for _i, _ax in enumerate(_axes):
            _ax.grid(False)
            current_models = models_rows[_i]
            category_ticks = []
            category_labels = []
            for j, model in enumerate(current_models):
                model_data = _proportions.loc[model]
                x_start = j * (len(_categories) + 3)
                x_pos = np.arange(x_start, x_start + len(_categories))
                for k, cat in enumerate(_categories):
                    _ax.bar(x_pos[k], model_data[cat], width=1.0, color=_colors[k], edgecolor='white', linewidth=0.5)
                category_ticks.extend(x_pos)
                category_labels.extend(_categories)
                model_center = x_start + len(_categories) / 2 - 0.5
                _ax.text(model_center, 0.9, model, fontsize=17, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            _ax.set_xticks(category_ticks)
            _ax.set_xticklabels(category_labels, fontsize=17, rotation=90, ha='center', va='top')
            _ax.tick_params(axis='x', which='major', length=0)
            _ax.tick_params(axis='y', which='major', direction='in', right=True, labelsize=19)
            _ax.set_ylabel('% of Words', fontsize=19, fontweight='bold')
            _ax.set_xlim(-1.5, max_x_limit - 1.5)
            _ax.set_ylim(0, 1.0)
            if _i == 0:
                _ax.text(0.5, 1.05, f'"{keyword.capitalize()}" Valence by Model', transform=_ax.transAxes, fontsize=20, fontweight='bold', va='bottom', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        legend_patches = [mpatches.Patch(color='#fe1f1f', label='Negative (N)'), mpatches.Patch(color='#1b84fe', label='Positive (P)'), mpatches.Patch(color='#a3a3a3', label='Neutral (Ne)')]
        leg = _axes[-1].legend(handles=legend_patches, loc='lower right', title='Aura Valence', prop={'size': 20, 'weight': 'bold'})
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(20)
        plt.tight_layout(h_pad=0.5)
        plt.savefig(f'./visualizations/OFFICIAL_{keyword}_Aura_comparison.svg', dpi=300, format='svg')
        return _fig

    # 1. Create a list to store the generated figures
    _generated_figs = []

    for _kw in _subjects:
        # 2. Append the returned figure to your list
        _fig = _plot_keyword_combinations(merged_4, _kw, row_layout=[5, 5, 4])
        _generated_figs.append(_fig)

    # 3. Stack and display the figures as the final expression in the cell
    mo.vstack(_generated_figs)
    return


@app.cell
def _(FOLDER_NAME_MAPPING, df_wide_4, merged_4, mo, mpatches, np, pd, plt):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 20, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20, 'legend.title_fontsize': 20})
    df_aura_7 = pd.read_csv('/Users/luisaporzio/Projects/teachme-project/src/data_validation/aggregated/run_level_aura_valences.csv')
    _subjects = ['mathematic', 'science']
    df_aura_7['model_name_short'] = df_aura_7['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    df_wide_4['model_name_short'] = df_wide_4['model_name'].str.replace('MANX_LLM_', '').replace(FOLDER_NAME_MAPPING)
    merged_7 = pd.merge(df_wide_4, df_aura_7, on=['model_name_short', 'cue_word', 'run_id'], how='inner')
    merged_7 = merged_7[merged_7['cue_word'].isin(_subjects)]

    def _categorize_valence(val):
        if val > 0:
            return 'P'
        elif val < 0:
            return 'N'
        else:
            return 'Ne'
    merged_7['kw_cat'] = merged_7['valence_cue_word'].apply(_categorize_valence)
    merged_7['aura_cat'] = merged_7['aura_net_valence'].apply(_categorize_valence)
    merged_7['valence_pair'] = merged_7['kw_cat'] + merged_7['aura_cat']
    _categories = ['NN', 'NP', 'NNe', 'PN', 'PP', 'PNe', 'NeN', 'NeP', 'NeNe']
    _colors = ['#fe1f1f', '#1b84fe', '#a3a3a3'] * 3

    def _plot_keyword_combinations(df, keyword, row_layout=[5, 5, 4]):
        df_kw = df[df['cue_word'] == keyword]
        _proportions = df_kw.groupby('model_name_short')['valence_pair'].value_counts(normalize=True).unstack(fill_value=0)
        for cat in _categories:
            if cat not in _proportions.columns:
                _proportions[cat] = 0
        _proportions = _proportions[_categories]
        models = sorted(_proportions.index.tolist())
        models_rows = []
        start_idx = 0
        for count in row_layout:
            end_idx = start_idx + count
            chunk = models[start_idx:end_idx]
            if chunk:
                models_rows.append(chunk)
            start_idx = end_idx
        num_rows = len(models_rows)
        max_models_in_row = max((len(r) for r in models_rows))
        max_x_limit = max_models_in_row * (len(_categories) + 3)
        _fig, _axes = plt.subplots(num_rows, 1, figsize=(18, 3.5 * num_rows), sharey=True)
        if num_rows == 1:
            _axes = [_axes]
        for _i, _ax in enumerate(_axes):
            _ax.grid(False)
            current_models = models_rows[_i]
            category_ticks = []
            category_labels = []
            for j, model in enumerate(current_models):
                model_data = _proportions.loc[model]
                x_start = j * (len(_categories) + 3)
                x_pos = np.arange(x_start, x_start + len(_categories))
                for k, cat in enumerate(_categories):
                    _ax.bar(x_pos[k], model_data[cat], width=1.0, color=_colors[k], edgecolor='black', linewidth=1.0)
                category_ticks.extend(x_pos)
                category_labels.extend(_categories)
                model_center = x_start + len(_categories) / 2 - 0.5
                _ax.text(model_center, 0.9, model, fontsize=17, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            _ax.set_xticks(category_ticks)
            _ax.set_xticklabels(category_labels, fontsize=17, rotation=90, ha='center', va='top')
            _ax.tick_params(axis='x', which='major', length=0)
            _ax.tick_params(axis='y', which='major', direction='in', right=True, labelsize=19)
            _ax.set_ylabel('% of Words', fontsize=19, fontweight='bold')
            _ax.set_xlim(-1.5, max_x_limit - 1.5)
            _ax.set_ylim(0, 1.0)
        legend_patches = [mpatches.Patch(facecolor='#fe1f1f', edgecolor='black', linewidth=1.3, label='Negative (N)'), mpatches.Patch(facecolor='#1b84fe', edgecolor='black', linewidth=1.3, label='Positive (P)'), mpatches.Patch(facecolor='#a3a3a3', edgecolor='black', linewidth=1.3, label='Neutral (Ne)')]
        leg = _axes[-1].legend(handles=legend_patches, loc='lower right', title=f'"{keyword.capitalize()}" Aura Valence', prop={'size': 20, 'weight': 'bold'})
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(20)
        plt.tight_layout(h_pad=0.5)
        plt.savefig(f'./visualizations/OFFICIAL_{keyword}_Aura_comparison.svg', dpi=300, format='svg')
        return _fig

    # 1. Create a list to store the generated figures
    _generated_figs = []

    for _kw in _subjects:
        # 2. Append the returned figure to your list
        _fig = _plot_keyword_combinations(merged_4, _kw, row_layout=[5, 5, 4])
        _generated_figs.append(_fig)

    # 3. Stack and display the figures as the final expression in the cell
    mo.vstack(_generated_figs)
    return


@app.cell
def _(FOLDER_NAME_MAPPING, df_wide_4, merged_4, mo, mpatches, np, pd, plt):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 20, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20, 'legend.title_fontsize': 20})
    df_aura_8 = pd.read_csv('/Users/luisaporzio/Projects/teachme-project/src/data_validation/aggregated/run_level_aura_valences.csv')
    _subjects = ['mathematic', 'science']
    df_aura_8['model_name_short'] = df_aura_8['model_name'].str.replace('compatible_dataset_', '').replace(FOLDER_NAME_MAPPING)
    df_wide_4['model_name_short'] = df_wide_4['model_name'].str.replace('MANX_LLM_', '').replace(FOLDER_NAME_MAPPING)
    merged_8 = pd.merge(df_wide_4, df_aura_8, on=['model_name_short', 'cue_word', 'run_id'], how='inner')
    merged_8 = merged_8[merged_8['cue_word'].isin(_subjects)]

    def _categorize_valence(val):
        if val > 0:
            return 'P'
        elif val < 0:
            return 'N'
        else:
            return 'Ne'
    merged_8['kw_cat'] = merged_8['valence_cue_word'].apply(_categorize_valence)
    merged_8['aura_cat'] = merged_8['aura_net_valence'].apply(_categorize_valence)
    merged_8['valence_pair'] = merged_8['kw_cat'] + merged_8['aura_cat']
    _categories = ['NN', 'NP', 'NNe', 'PN', 'PP', 'PNe', 'NeN', 'NeP', 'NeNe']
    _colors = ['#f47f7f', '#7cbdec', '#a6a6a6'] * 3

    def _plot_keyword_combinations(df, keyword, row_layout=[5, 5, 4]):
        df_kw = df[df['cue_word'] == keyword]
        _proportions = df_kw.groupby('model_name_short')['valence_pair'].value_counts(normalize=True).unstack(fill_value=0)
        for cat in _categories:
            if cat not in _proportions.columns:
                _proportions[cat] = 0
        _proportions = _proportions[_categories]
        models = sorted(_proportions.index.tolist())
        models_rows = []
        start_idx = 0
        for count in row_layout:
            end_idx = start_idx + count
            chunk = models[start_idx:end_idx]
            if chunk:
                models_rows.append(chunk)
            start_idx = end_idx
        num_rows = len(models_rows)
        max_models_in_row = max((len(r) for r in models_rows))
        max_x_limit = max_models_in_row * (len(_categories) + 3)
        _fig, _axes = plt.subplots(num_rows, 1, figsize=(18, 3.5 * num_rows), sharey=True)
        if num_rows == 1:
            _axes = [_axes]
        for _i, _ax in enumerate(_axes):
            _ax.grid(False)
            current_models = models_rows[_i]
            category_ticks = []
            category_labels = []
            for j, model in enumerate(current_models):
                model_data = _proportions.loc[model]
                x_start = j * (len(_categories) + 3)
                x_pos = np.arange(x_start, x_start + len(_categories))
                for k, cat in enumerate(_categories):
                    _ax.bar(x_pos[k], model_data[cat], width=1.0, color=_colors[k], edgecolor='white', linewidth=0.5)
                category_ticks.extend(x_pos)
                category_labels.extend(_categories)
                model_center = x_start + len(_categories) / 2 - 0.5
                _ax.text(model_center, 0.9, model, fontsize=17, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            _ax.set_xticks(category_ticks)
            if _i == 0:
                _ax.set_xticklabels([])
            else:
                _ax.set_xticklabels(category_labels, fontsize=20, rotation=90, ha='center', va='top')
            _ax.tick_params(axis='x', which='major', length=0)
            _ax.tick_params(axis='y', which='major', direction='in', right=True, labelsize=16)
            _ax.set_ylabel('% of Words', fontsize=20, fontweight='bold')
            _ax.set_xlim(-1.5, max_x_limit - 1.5)
            _ax.set_ylim(0, 1.0)
            if _i == 0:
                _ax.text(0.5, 1.05, f'"{keyword.capitalize()}" Valence by Model', transform=_ax.transAxes, fontsize=20, fontweight='bold', va='bottom', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        legend_patches = [mpatches.Patch(color='#f47f7f', label='Negative (N)'), mpatches.Patch(color='#7cbdec', label='Positive (P)'), mpatches.Patch(color='#a6a6a6', label='Neutral (Ne)')]
        leg = _axes[-1].legend(handles=legend_patches, loc='lower right', title='Aura Valence', prop={'size': 20, 'weight': 'bold'})
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(20)
        plt.tight_layout(h_pad=0.5)
        if num_rows > 1:
            _fig.canvas.draw()
            pos0 = _axes[0].get_position()
            pos1 = _axes[1].get_position()
            target_gap_inches = 8.0 / 25.4
            fig_height_inches = _fig.get_figheight()
            target_gap_rel = target_gap_inches / fig_height_inches
            current_bottom_0 = pos0.y0
            current_top_1 = pos1.y0 + pos1.height
            current_gap = current_bottom_0 - current_top_1
            shift = current_gap - target_gap_rel
            _axes[0].set_position([pos0.x0, pos0.y0 - shift, pos0.width, pos0.height])
        return _fig

    # 1. Create a list to store the generated figures
    _generated_figs = []

    for _kw in _subjects:
        # 2. Append the returned figure to your list
        _fig = _plot_keyword_combinations(merged_4, _kw, row_layout=[5, 5, 4])
        _generated_figs.append(_fig)

    # 3. Stack and display the figures as the final expression in the cell
    mo.vstack(_generated_figs)
    return


if __name__ == "__main__":
    app.run()
