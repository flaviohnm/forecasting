import collections
import glob
import logging
import os
import re
import shutil

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importa o teste estatístico
try:
    from src.analysis.statistical_tests import run_significance_analysis
    HAS_STATS = True
except ImportError:
    logging.warning("Módulo 'src.analysis.statistical_tests' não encontrado. Testes estatísticos serão pulados.")
    HAS_STATS = False

# --- CONFIGURAÇÃO DE CORES ---
MODEL_COLORS = {
    "Real": "black",
    "y": "black",
    "ARIMA": "#1f77b4",
    "ETS": "#ff7f0e",
    "Naive": "#2ca02c",
    "SeasonalNaive": "#d62728",
    "NBEATS": "#9467bd",
    "NHITS": "#8c564b",
    "Informer": "#e377c2",
    "TFT": "#7f7f7f",
    "AutoARIMA": "#bcbd22",
}

def get_model_color(model_name):
    for key, color in MODEL_COLORS.items():
        if key.upper() in model_name.upper():
            return color
    return "#333333"

def parse_model_info(row):
    exec_name = row["model"]
    ds_name = row["dataset"]
    clean_name = exec_name.replace(f"{ds_name}_", "")
    clean_name = re.sub(r'_h\d+$', '', clean_name)
    
    if "hybrid_" in clean_name:
        group = "Hybrid Methods"
        parts = clean_name.replace("hybrid_", "").split("_on_")
        if len(parts) == 2:
            model_display = f"{parts[0].upper()} + {parts[1].upper()}"
        else:
            model_display = clean_name.upper()
    elif "base_" in clean_name:
        core_name = clean_name.replace("base_", "")
        if core_name in ["nbeats", "nhits", "informer", "tft"]:
            group = "Deep Learning"
        else:
            group = "Statistical"
        model_display = core_name.upper()
    else:
        group = "Other"
        model_display = clean_name.upper()
        
    return pd.Series([group, model_display])


def generate_report(main_config, successful_runs):
    logging.info("--- [REPORT] Gerando Relatório Markdown ---")

    metrics_path = main_config["results_paths"]["metrics"]
    reports_path = main_config["results_paths"]["reports"]
    statistical_path = main_config["results_paths"]["statistical"]
    plots_path = main_config["results_paths"]["plots"]
    pd_path = os.path.join(plots_path, "percentage_difference")
    
    # AUTOLIMPEZA: Evita fantasmas de gráficos gerados em execuções anteriores
    if os.path.exists(pd_path):
        shutil.rmtree(pd_path)
    os.makedirs(pd_path, exist_ok=True)
    os.makedirs(reports_path, exist_ok=True)

    # 1. Carregar Métricas
    all_files = glob.glob(os.path.join(metrics_path, "*.csv"))
    if not all_files:
        logging.warning("Nenhuma métrica encontrada para gerar o relatório.")
        return
    df_list = [pd.read_csv(f) for f in all_files]
    df_full = pd.concat(df_list, ignore_index=True)
    df_full.to_csv(os.path.join(reports_path, "consolidated_metrics.csv"), index=False)

    # 2. Executar Testes Estatísticos
    if HAS_STATS:
        try:
            run_significance_analysis(main_config, metrics_df=df_full)
        except Exception as e:
            logging.error(f"Erro nos testes estatísticos: {e}")

    # 3. Processar Dados para a Tabela HTML
    df_full[["Grupo", "Modelo"]] = df_full.apply(parse_model_info, axis=1)

    group_order = ["Statistical", "Deep Learning", "Hybrid Methods", "Other"]
    df_full["Grupo"] = pd.Categorical(df_full["Grupo"], categories=group_order, ordered=True)

    pivot_df = df_full.pivot_table(
        index=["Grupo", "Modelo"],
        columns="horizon",
        values=["mae", "mase", "mape", "smape"]
    )

    metrics_order = ["mae", "mase", "mape", "smape"]
    available_metrics = [m for m in metrics_order if m in pivot_df.columns.get_level_values(0)]
    pivot_df = pivot_df.reindex(columns=available_metrics, level=0)

    rename_dict = {"mae": "MAE", "mase": "MASE", "mape": "MAPE (%)", "smape": "sMAPE (%)"}
    pivot_df = pivot_df.rename(columns=rename_dict, level=0)
    pivot_df.columns = pivot_df.columns.set_levels([f"H={h}" for h in pivot_df.columns.levels[1]], level=1)

    html_table = pivot_df.to_html(classes="table", justify="center", float_format="%.4f", border=1)

    # --- LÓGICA DE PERCENTAGE DIFFERENCE (PD) AGREGADA (FIG 13) ---
    logging.info("Calculando PD agregado para a Fig 13 do artigo...")
    
    # O observed=True impede o Pandas de inventar linhas NaN que esvaziavam o gráfico
    df_mean_mape = df_full.dropna(subset=["mape"]).groupby(["dataset", "Grupo", "Modelo"], observed=True)["mape"].mean().reset_index()
    hybrid_models = df_mean_mape[df_mean_mape["Grupo"] == "Hybrid Methods"]
    
    for _, hyb_row in hybrid_models.iterrows():
        ds = hyb_row["dataset"]
        hyb_name = hyb_row["Modelo"]
        hyb_mape = hyb_row["mape"]
        
        df_benchmarks = df_mean_mape[(df_mean_mape["dataset"] == ds) & (df_mean_mape["Grupo"] != "Hybrid Methods")].copy()
        df_benchmarks = df_benchmarks.dropna(subset=["mape"])
        
        if not df_benchmarks.empty:
            df_benchmarks["PD"] = 100 * ((df_benchmarks["mape"] - hyb_mape) / df_benchmarks["mape"])
            df_benchmarks = df_benchmarks.sort_values(by="PD", ascending=True)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_benchmarks, x="Modelo", y="PD", color="#a9a9a9")
            plt.title(f"Percentage Difference (PD) in terms of MAPE between the {hyb_name}\nand other literature models regarding all horizons", pad=15)
            plt.ylabel("PD (%)")
            plt.xlabel("Model")
            plt.axhline(0, color="black", linewidth=1.2)
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            safe_name = hyb_name.replace(" + ", "_").replace(" ", "")
            plt.savefig(os.path.join(pd_path, f"PD_Fig13_{ds}_{safe_name}.png"), dpi=150)
            plt.close()

    # 4. Escrever Relatório Markdown
    md_output = os.path.join(reports_path, "relatorio_final.md")
    with open(md_output, "w", encoding="utf-8") as f:
        f.write("# Relatório Final de Experimentos\n\n")
        f.write("## 🏆 Matriz de Resultados Abrangente\n\n")
        f.write(html_table)
        f.write("\n\n<br><br>\n\n")

        # --- SEÇÃO: GRÁFICOS DE FAMÍLIAS ---
        f.write("## 📈 Dinâmica de Previsão por Famílias\n\n")
        family_plots = glob.glob(os.path.join(plots_path, "families", "*.png"))
        if family_plots:
            grouped_plots = collections.defaultdict(dict)
            pattern = re.compile(r"^family_(?P<dataset>.+)_(?P<base>[^_]+)_(?P<dl>[^_]+)_h(?P<horizon>\d+)\.png$")
            
            for fp in sorted(family_plots):
                basename = os.path.basename(fp)
                match = pattern.match(basename)
                if match:
                    dataset = match.group("dataset")
                    base = match.group("base")
                    dl = match.group("dl")
                    horizon = int(match.group("horizon"))
                    rel_path = os.path.relpath(fp, start=reports_path)
                    grouped_plots[(dataset, base, dl)][horizon] = rel_path

            for (dataset, base, dl), horizons_dict in grouped_plots.items():
                f.write(f"### Grupo: {base.upper()} x {dl.upper()} x Híbrido ({dataset})\n\n")
                def get_img(h):
                    return f'<img src="{horizons_dict[h]}" width="100%" style="width:100%; max-width:100%; height:auto;">' if h in horizons_dict else "<span style='color:gray;'><i>Não executado</i></span>"

                html_grid = f'<table width="100%" style="width:100%; table-layout:fixed; border:none; text-align:center;"><tr><td style="width:50%; border:none; padding:5px; vertical-align:top;"><b>H=96</b><br>{get_img(96)}</td><td style="width:50%; border:none; padding:5px; vertical-align:top;"><b>H=192</b><br>{get_img(192)}</td></tr><tr><td style="width:50%; border:none; padding:5px; vertical-align:top;"><b>H=336</b><br>{get_img(336)}</td><td style="width:50%; border:none; padding:5px; vertical-align:top;"><b>H=720</b><br>{get_img(720)}</td></tr></table>'
                f.write(html_grid + "\n\n<br>\n\n")
        
        f.write("---\n\n")

        # --- SEÇÃO: PERCENTAGE DIFFERENCE (PD) (Apenas as imagens, sem tabelas) ---
        f.write("## 🚀 Percentage Difference (PD) Global\n\n")
        f.write("Este gráfico apresenta a Diferença Percentual (PD) em termos de MAPE entre o modelo Híbrido proposto e os modelos da literatura, **agregando todos os horizontes de previsão**.\n\n")
        
        pd_plots = glob.glob(os.path.join(pd_path, "PD_Fig13_*.png"))
        if pd_plots:
            for pp in sorted(pd_plots):
                rel_path = os.path.relpath(pp, start=reports_path)
                f.write(f"![Percentage Difference]({rel_path})\n\n")

        f.write("---\n\n")

        # --- SEÇÃO: TESTES ESTATÍSTICOS ---
        f.write("## 🔬 Teste Estatístico de Diebold-Mariano (DM)\n\n")
        f.write("A tabela abaixo apresenta a estatística DM e o p-Value comparando o modelo Híbrido com os modelos da literatura para todos os horizontes agregados. Um *p-Value < 0.05* indica diferença estatística significativa a favor do modelo proposto.\n\n")
        
        dm_csv = os.path.join(statistical_path, "dm_test_results.csv")
        if os.path.exists(dm_csv):
            df_dm = pd.read_csv(dm_csv)
            dm_html = df_dm.to_html(classes="table", justify="center", index=False, border=1)
            f.write(dm_html + "\n\n<br>\n\n")

    logging.info(f"Relatório Markdown salvo com sucesso em: {md_output}")


def generate_plots(main_config, successful_runs):
    logging.info("--- [PLOTS] Gerando Gráficos de Previsão ---")

    forecasts_path = main_config["results_paths"]["forecasts"]
    plots_path = main_config["results_paths"]["plots"]
    family_path = os.path.join(plots_path, "families")

    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(family_path, exist_ok=True)

    all_runs_dict = {}

    for exec_name in successful_runs:
        file_path = os.path.join(forecasts_path, f"forecast_{exec_name}.csv")

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["ds"] = pd.to_datetime(df["ds"])
                all_runs_dict[exec_name] = df
            except Exception as e:
                logging.error(f"Erro ao carregar forecast para {exec_name}: {e}")

    logging.info("Gerando gráficos comparativos de Família...")

    hybrid_pattern = re.compile(r"^(?P<dataset>.+)_hybrid_(?P<dl>.+)_on_(?P<base>.+)_h(?P<horizon>\d+)$")

    for exec_name, df_hybrid in all_runs_dict.items():
        match = hybrid_pattern.match(exec_name)
        if match:
            dataset = match.group("dataset")
            dl_model = match.group("dl")
            base_model = match.group("base")
            horizon = match.group("horizon")

            base_exec = f"{dataset}_base_{base_model}_h{horizon}"
            dl_exec = f"{dataset}_base_{dl_model}_h{horizon}"

            if base_exec in all_runs_dict and dl_exec in all_runs_dict:
                df_base = all_runs_dict[base_exec]
                df_dl = all_runs_dict[dl_exec]

                zoom_size = min(len(df_hybrid), 300)

                plt.figure(figsize=(14, 7))

                plt.plot(df_hybrid["ds"].tail(zoom_size), df_hybrid["y"].tail(zoom_size), label="Real (Ground Truth)", color="black", linewidth=2.5, alpha=0.5, zorder=1)
                plt.plot(df_base["ds"].tail(zoom_size), df_base["y_hat"].tail(zoom_size), label=f"Estatístico ({base_model.upper()})", color="#1f77b4", linewidth=1.5, alpha=0.8, zorder=2, linestyle="--")
                plt.plot(df_dl["ds"].tail(zoom_size), df_dl["y_hat"].tail(zoom_size), label=f"Neural ({dl_model.upper()})", color="#ff7f0e", linewidth=1.5, alpha=0.8, zorder=3, linestyle="-.")
                plt.plot(df_hybrid["ds"].tail(zoom_size), df_hybrid["y_hat"].tail(zoom_size), label=f"Híbrido ({dl_model.upper()} + {base_model.upper()})", color="#d62728", linewidth=2.5, alpha=1.0, zorder=4)

                plt.title(f"Comparativo de Desempenho: {base_model.upper()} vs {dl_model.upper()} vs Híbrido | Dataset: {dataset} | Horizonte: {horizon}")
                plt.xlabel("Tempo")
                plt.ylabel("Valores")
                plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
                plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                plt.tight_layout()

                plt.savefig(os.path.join(family_path, f"family_{dataset}_{base_model}_{dl_model}_h{horizon}.png"), dpi=150)
                plt.close()

    logging.info(f"Relatórios concluídos com sucesso.")