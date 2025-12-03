# File: src/reporting/data_loader.py
import os
import pandas as pd

def load_all_data(main_config, successful_executions):
    metrics_path = main_config['results_paths']['metrics']
    plots_path = main_config['results_paths']['plots']
    all_metrics, all_forecasts = [], []

    for ds, mod in successful_executions:
        ename = f"{ds['name']}_{mod['model_name']}"
        m_file = os.path.join(metrics_path, f"metrics_{ename}.csv")
        f_file = os.path.join(plots_path, f"forecasts_{ename}.csv")
        grp = mod.get('comparison_group', 'other')

        if os.path.exists(m_file):
            try: d=pd.read_csv(m_file); d['comparison_group']=grp; all_metrics.append(d)
            except: pass
        if os.path.exists(f_file):
            try:
                d=pd.read_csv(f_file); d.rename(columns={d.columns[0]: 'date_index'}, inplace=True)
                try: d['date_index'] = pd.to_datetime(d['date_index'])
                except: pass
                d['model_type']=mod['model_type']; d['dataset']=ds['name']; d['comparison_group']=grp
                all_forecasts.append(d)
            except: pass
            
    summary = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    forecasts = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    return summary, forecasts

def filter_models(summary, forecasts, exclude):
    if not summary.empty: summary = summary[~summary['model_type'].isin(exclude)]
    if not forecasts.empty: forecasts = forecasts[~forecasts['model_type'].isin(exclude)]
    return summary, forecasts