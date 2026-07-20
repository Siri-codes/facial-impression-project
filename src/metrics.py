import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from config import N_SPLITS, SEED, SPEARMAN_BROWN

def pearsonr_correlation(df_1, df_2):

    # 1. Merge the two dataframes on the 'stimulus' column
    # Added suffixes to tell which column is which (e.g., trustworthy_human vs trustworthy_gemini)
    merged_df = pd.merge(df_1, df_2, on='stimulus', suffixes=('_1', '_2'))

    # The list of baseline attributes we want to loop through
    attributes = [col for col in df_2.columns if col != 'stimulus']

    # A dictionary to store our final correlation results
    correlation_results = {}

    for attr in attributes:
        # Construct the exact column names we generated during the merge
        col_1 = f"{attr}_1"
        col_2 = f"{attr}_2"

        # Calculate the correlation between the human and gemini columns
        current_r = merged_df[col_1].corr(merged_df[col_2])

        # Store the result in dictionary
        correlation_results[attr] = current_r

    # Convert to a DataFrame for easy plotting later
    corr_df = pd.DataFrame(list(correlation_results.items()), columns=['Attribute', 'Correlation'])
    corr_df = corr_df.sort_values(by="Correlation", ascending=False)

    return corr_df

def calculate_human_reliability(df_human, attribute_name, n_splits=N_SPLITS, seed=SEED, spearman_brown=SPEARMAN_BROWN):
    """
    Calculates the split-half reliability (R^2) for a single facial attribute accross 100 random splits.
    """
    #set random generator
    rng = np.random.default_rng(seed)  

    # 1. Filter the data
    df_attr = df_human[df_human['attribute'] == attribute_name]

    split_r2s = []
    
    for split in range(n_splits):
        group_a = []
        group_b = []

        # Group by stimulus to shuffle ratings locally per image
        for stim, group in df_attr.groupby('stimulus'):
            ratings = group['rating'].values.copy() # Make a copy to avoid in-place corruption

            # Shuffle the ratings
            rng.shuffle(ratings)       

            len_ratings = len(ratings)
            mid = len_ratings // 2

            if mid == 0:
                continue

            # Extract the split-halves and calculate their means
            mean_a = np.mean(ratings[:mid])
            mean_b = np.mean(ratings[mid:])

            group_a.append({'stimulus': stim, 'mean_rating_a': mean_a})
            group_b.append({'stimulus': stim, 'mean_rating_b': mean_b})

        # Convert lists to temporary DataFrames
        df_a = pd.DataFrame(group_a)
        df_b = pd.DataFrame(group_b)

        if df_a.empty or df_b.empty:
            continue
            
        # Merge on stimulus to ensure matching row alignment
        merged_splits = pd.merge(df_a, df_b, on='stimulus')

        # Calculate Pearson correlation (returns tuple: (r_value, p_value))
        r_split, _ = pearsonr(merged_splits['mean_rating_a'], merged_splits['mean_rating_b'])

        if spearman_brown:
            r_split = 2 * r_split / (1 + r_split)   # correct r, then square

        # Convert r to R^2 and store it in split_r2s
        split_r2s.append(r_split ** 2)

    # Return the mean of all successful splits
    return np.mean(split_r2s) if split_r2s else np.nan

def compute_self_reliability(rep1_df, rep2_df, rep3_df):
    """
    Computes per-attribute self-reliability (R^2) for a model, averaged across
    all three pairwise comparisons of three repeated measurement runs.
    """
    corr_12 = pearsonr_correlation(rep1_df, rep2_df)
    corr_13 = pearsonr_correlation(rep1_df, rep3_df)
    corr_23 = pearsonr_correlation(rep2_df, rep3_df)

    merged = corr_12.merge(corr_13, on='Attribute', suffixes=('_12', '_13'))
    merged = merged.merge(corr_23, on='Attribute')
    merged = merged.rename(columns={'Correlation': 'Correlation_23'})

    # Average raw r across the three pairings, then square to match R^2 scale
    merged['Reliability_r'] = merged[['Correlation_12', 'Correlation_13', 'Correlation_23']].mean(axis=1)
    merged['self_reliability_r2'] = merged['Reliability_r'] ** 2

    return merged[['Attribute', 'self_reliability_r2']].rename(columns={'Attribute': 'attribute'})

def build_evaluation_dataset(df_human_means, df_human_raw, df_ai):
    """
    Loops through overlapping attributes, calculates AI and Human R^2 metrics,
    and returns a sorted DataFrame ready for visualization.
    """

    # Find matching attributes present in both datasets
    ai_attributes = [col for col in df_ai.columns if col != 'stimulus']
    human_attributes = df_human_raw['attribute'].unique()
    attributes_to_plot = list(set(ai_attributes).intersection(set(human_attributes)))
    
    results = []
    
    ai_correlations = pearsonr_correlation(df_human_means, df_ai)
    
    # Turn the 'Attribute' column into the row indices
    ai_lookup = ai_correlations.set_index('Attribute')

    for attr in attributes_to_plot:
        print(f"Calculating metrics for: {attr}...")
        
        # 1. Compute Human Intersubject Reliability 
        human_r2 = calculate_human_reliability(df_human_raw, attr, n_splits=100)
        
        # 2. Compute AI Performance R^2
        ai_r = ai_lookup.loc[attr, 'Correlation']
        ai_r2 = ai_r ** 2
        
        # Append to our structured results list
        results.append({
            'attribute': attr,
            'human_reliability_r2': human_r2,
            'ai_performance_r2': ai_r2
        })
        
    # Turn results into a DataFrame and sort by human ceiling (mimicking PNAS)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='human_reliability_r2', ascending=True).reset_index(drop=True)
    return df_results