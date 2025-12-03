# File: src/reporting/reporter.py
import os
from datetime import datetime
import pandas as pd
from . import data_loader, statistics, visualizer, tables

def generate_report(main_config: dict, successful_executions: list):
    print("Iniciando geração do relatório refatorado...")
    report_dir = "reports"; plot_dir = os.path.join(report_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    summary, forecasts = data_loader.load_all_data(main_config, successful_executions)
    summary, forecasts = data_loader.filter_models(summary, forecasts, ['ETS', 'NAIVE', 'SEASONAL_NAIVE'])
    
    if summary.empty: print("Sem dados."); return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = f"# Relatório Final - {now}\n\n"

    md += "## 1. Resumo Global\n\n"
    md += tables.generate_champions_table(summary) + "\n\n"
    
    md += "### Matrizes Consolidadas\n"
    for m in ['MAPE', 'MASE', 'RMSSE']:
        if m in summary.columns:
            md += f"**{m}**\n" + tables.generate_consolidated_matrix(summary, m) + "\n\n"

    md += "### Análise Global\n"
    if box := visualizer.generate_rank_boxplot(summary, plot_dir): md += f"![Boxplot](plots/{box})\n\n"
    if cd := visualizer.generate_cd_diagram(summary, plot_dir): md += f"![CD Diagram](plots/{cd})\n\n"
    md += "---\n"

    md += "## 2. Detalhes por Dataset\n"
    sort_map = {'benchmark_statistical': 0, 'benchmark_standalone_dl': 1, 'hybrid_direct': 2, 'hybrid_mimo': 3, 'hybrid_recursive': 4}
    
    for ds in sorted(summary['dataset'].unique()):
        md += f"### {ds}\n"
        ds_sum = summary[summary['dataset'] == ds]
        ds_forc = forecasts[forecasts['dataset'] == ds] if not forecasts.empty else pd.DataFrame()

        if pd_plot := visualizer.generate_pd_plot(ds_sum, plot_dir, ds): md += f"![PD](plots/{pd_plot})\n\n"

        if not ds_forc.empty:
            winner = ds_sum.loc[ds_sum['MAPE'].idxmin()]['model_type']
            try:
                w_data = ds_forc[ds_forc['model_type'] == winner].sort_values('date_index')
                real, base_p = w_data['real'].values, w_data['previsao'].values
                dm_list = []
                for m in ds_sum['model_type'].unique():
                    if m == winner: continue
                    c_data = ds_forc[ds_forc['model_type'] == m].sort_values('date_index')
                    if len(c_data) == len(real):
                        s, p = statistics.calculate_dm_statistic(real, base_p, c_data['previsao'].values)
                        dm_list.append({'Modelo': m, 'DM': f"{s:.3f}", 'p-Val': f"{p:.2e}"})
                if dm_list: md += f"**DM (Ref: {winner})**\n" + pd.DataFrame(dm_list).to_markdown(index=False) + "\n\n"
            except: pass

            if prop := visualizer.generate_proposal_vs_reference_plot(ds_forc, plot_dir, ds): md += f"![PropRef](plots/{prop})\n\n"
            if panel := visualizer.generate_actual_vs_predicted_plots(ds_forc, plot_dir, ds): md += f"![Panel](plots/{panel})\n\n"
        
        md += "---\n"

    with open(f"reports/relatorio_final_{datetime.now().strftime('%Y%m%d_%H%M')}.md", 'w', encoding='utf-8') as f: f.write(md)
    print("Relatório gerado.")