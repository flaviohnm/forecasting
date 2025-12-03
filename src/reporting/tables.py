# File: src/reporting/tables.py
import pandas as pd

def generate_consolidated_matrix(summary_df, metric='MAPE'):
    if summary_df.empty: return "Sem dados."
    pivot_df = summary_df.pivot_table(index=['comparison_group', 'model_type'], columns='dataset', values=metric, aggfunc='first')
    sort_mapping = {'benchmark_statistical': 0, 'benchmark_standalone_dl': 1, 'hybrid_direct': 2, 'hybrid_mimo': 3, 'hybrid_recursive': 4}
    pivot_df = pivot_df.sort_index(key=lambda x: x.map(sort_mapping) if x.name == 'comparison_group' else x)
    
    def highlight_min(s):
        is_min = s == s.min()
        return [f"**{v:.4f}**" if is_min_val and pd.notna(v) else f"{v:.4f}" if pd.notna(v) else "-" for v, is_min_val in zip(s, is_min)]

    formatted_df = pivot_df.apply(highlight_min, axis=0).reset_index()
    formatted_df.columns.name = None 
    formatted_df.rename(columns={'comparison_group': 'Grupo', 'model_type': 'Modelo'}, inplace=True)
    return formatted_df.to_markdown(index=False)

def generate_champions_table(summary_df):
    try:
        winners_idx = summary_df.groupby('dataset')['MAPE'].idxmin()
        winners = summary_df.loc[winners_idx].sort_values('dataset')[['comparison_group', 'model_type', 'dataset', 'MAPE', 'MASE', 'RMSSE']]
        winners.columns = ['Grupo', 'Modelo', 'Dataset', 'MAPE', 'MASE', 'RMSSE']
        for col in ['MAPE', 'MASE', 'RMSSE']:
            winners[col] = winners[col].apply(lambda x: f"**{x:.4f}**")
        return winners.to_markdown(index=False)
    except: return ""

def generate_ranking_table(ds_df, metric, sort_mapping):
    best_val = ds_df[metric].min()
    ds_df = ds_df.copy()
    ds_df['sort'] = ds_df['comparison_group'].map(sort_mapping).fillna(99)
    sorted_df = ds_df.sort_values(by=['sort', 'model_type'])
    
    view = sorted_df[['comparison_group', 'model_type', 'dataset', metric]].copy()
    view.columns = ['Grupo', 'Modelo', 'Dataset', metric]
    view[metric] = view[metric].apply(lambda x: f"{x:.4f}")
    
    def bold(row):
        val_orig = sorted_df.loc[row.name, metric]
        if abs(val_orig - best_val) < 1e-9: return [f"**{x}**" for x in row]
        return row.tolist()
    
    final_tbl = pd.DataFrame(view.apply(bold, axis=1).tolist(), columns=view.columns)
    return final_tbl.to_markdown(index=False)