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
    from src.evaluation.statistical_tests import run_significance_analysis

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


def parse_model_info(row):
    exec_name = row["model"]
    ds_name = row.get("dataset", "ETTh1")
    clean_name = exec_name.replace(f"{ds_name}_", "")
    clean_name = re.sub(r"_h\d+$", "", clean_name)

    if "hybrid" in clean_name.lower():
        group = "Hybrid Methods"
        model_display = clean_name.upper().replace("HYBRID_", "").replace("_ON_", " + ")
    elif "base_" in clean_name.lower():
        core_name = clean_name.replace("base_", "").lower()
        group = "Deep Learning" if core_name in ["nbeats", "nhits", "informer", "tft"] else "Statistical"
        model_display = core_name.upper()
    else:
        group = "Other"
        model_display = clean_name.upper()

    return pd.Series([group, model_display])


def generate_pd_plot(main_config, df_full):
    """
    Gera o gráfico de barras de Percentage Difference (PD) baseado no artigo de referência.
    Calcula a melhoria relativa do melhor modelo Híbrido contra a literatura.
    """
    plots_path = main_config["results_paths"]["plots"]
    pd_path = os.path.join(plots_path, "percentage_difference")
    os.makedirs(pd_path, exist_ok=True)

    try:
        # 1. Calcula o MAPE médio global de cada modelo (agregando todos os horizontes)
        df_mean = df_full.groupby(["Grupo", "Modelo"])["mape"].mean().reset_index()
        df_mean = df_mean.dropna()

        if df_mean.empty:
            return

        # 2. Identifica o "Modelo Proposto" (O Híbrido com menor MAPE médio)
        hybrids = df_mean[df_mean["Grupo"] == "Hybrid Methods"]
        if hybrids.empty:
            logging.warning("Nenhum modelo híbrido encontrado para gerar o gráfico de PD.")
            return

        best_hybrid = hybrids.loc[hybrids["mape"].idxmin()]
        best_hybrid_name = best_hybrid["Modelo"]
        best_hybrid_mape = best_hybrid["mape"]

        # 3. Isola os modelos da literatura (Bases Estatísticas e Deep Learning Puros)
        baselines = df_mean[df_mean["Grupo"].isin(["Statistical", "Deep Learning"])].copy()
        if baselines.empty:
            return

        # 4. Fórmula de Melhoria Percentual: (Baseline - Proposed) / Baseline * 100
        baselines["PD"] = ((baselines["mape"] - best_hybrid_mape) / baselines["mape"]) * 100
        baselines = baselines.sort_values(by="PD", ascending=True)

        # 5. Plotagem do Gráfico (A estética do artigo: barras cinzas ascendentes)
        plt.figure(figsize=(12, 7))
        bars = plt.bar(baselines["Modelo"], baselines["PD"], color="#a9a9a9", width=0.65)

        plt.ylabel("PD(%)", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.title(
            f"Percentage Difference (PD) in terms of mean MAPE between\n{best_hybrid_name} and literature models",
            fontsize=14,
        )

        plt.xticks(rotation=45, ha="right", fontsize=11)
        plt.grid(axis="y", linestyle="-", alpha=0.3)
        plt.tight_layout()

        plot_file = os.path.join(pd_path, "pd_global.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        logging.info(f"Gráfico PD gerado com sucesso em: {plot_file}")

    except Exception as e:
        logging.error(f"Erro ao gerar o gráfico de Percentage Difference: {e}")


def generate_report(main_config, successful_runs):
    logging.info("--- [REPORT] Gerando Relatório Markdown ---")

    metrics_path = main_config["results_paths"]["metrics"]
    reports_path = main_config["results_paths"]["reports"]
    statistical_path = main_config["results_paths"]["statistical"]
    plots_path = main_config["results_paths"]["plots"]
    pd_path = os.path.join(plots_path, "percentage_difference")

    os.makedirs(pd_path, exist_ok=True)
    os.makedirs(reports_path, exist_ok=True)

    all_files = glob.glob(os.path.join(metrics_path, "*.csv"))
    if not all_files:
        logging.warning("Nenhum arquivo de métricas encontrado para gerar o relatório.")
        return

    df_list = [pd.read_csv(f) for f in all_files]
    df_full = pd.concat(df_list, ignore_index=True)
    df_full.to_csv(os.path.join(reports_path, "consolidated_metrics.csv"), index=False)

    # 1. Executar Testes Estatísticos (DM Test e Friedman)
    if HAS_STATS:
        try:
            run_significance_analysis(main_config, metrics_df=df_full)
        except Exception as e:
            logging.error(f"Erro nos testes estatísticos: {e}")

    # 2. Gerar Gráfico de Percentage Difference (PD)
    df_full_parsed = df_full.copy()
    df_full_parsed[["Grupo", "Modelo"]] = df_full_parsed.apply(parse_model_info, axis=1)
    generate_pd_plot(main_config, df_full_parsed)

    # 3. Processar Dados para a Tabela HTML (Matriz de Resultados)
    df_full[["Grupo", "Modelo"]] = df_full.apply(parse_model_info, axis=1)
    group_order = ["Statistical", "Deep Learning", "Hybrid Methods", "Other"]
    df_full["Grupo"] = pd.Categorical(df_full["Grupo"], categories=group_order, ordered=True)

    pivot_df = df_full.pivot_table(
        index=["Grupo", "Modelo"], columns="horizon", values=["mae", "mase", "mape", "smape"], observed=True
    )

    metrics_order = ["mae", "mase", "mape", "smape"]
    available_metrics = [m for m in metrics_order if m in pivot_df.columns.get_level_values(0)]
    pivot_df = pivot_df.reindex(columns=available_metrics, level=0)

    rename_dict = {"mae": "MAE", "mase": "MASE", "mape": "MAPE (%)", "smape": "sMAPE (%)"}
    pivot_df = pivot_df.rename(columns=rename_dict, level=0)
    pivot_df.columns = pivot_df.columns.set_levels([f"H={h}" for h in pivot_df.columns.levels[1]], level=1)

    html_table = pivot_df.to_html(classes="table", justify="center", float_format="%.4f", border=1)

    # ESCRITA DO MARKDOWN BLINDADA
    md_output = os.path.join(reports_path, "relatorio_final.md")
    with open(md_output, "w", encoding="utf-8") as f:
        f.write("# Relatório Final de Experimentos\n\n")
        f.write("## 🏆 Matriz de Resultados Abrangente\n\n")
        f.write(html_table)
        f.write("\n\n<br><br>\n\n")

        # --- SEÇÃO ESTATÍSTICA (INJEÇÃO DO FRIEDMAN E DM) ---
        f.write("## 🔬 Testes de Significância Estatística\n\n")

        friedman_csv = os.path.join(statistical_path, "friedman_ranks.csv")
        friedman_txt = os.path.join(statistical_path, "friedman_summary.txt")

        if os.path.exists(friedman_csv) and os.path.exists(friedman_txt):
            f.write("### Teste Não-Paramétrico de Friedman\n\n")
            f.write(
                "Avaliação de superioridade global (MAE) utilizando múltiplos horizontes como blocos experimentais.\n\n"
            )

            with open(friedman_txt, "r", encoding="utf-8") as txt:
                lines = txt.readlines()
                if len(lines) >= 2:
                    f.write(f"- **Estatística Chi-Square:** `{lines[0].strip()}`\n")
                    f.write(f"- **P-Value:** `{lines[1].strip()}`\n\n")

            df_ranks = pd.read_csv(friedman_csv)
            df_ranks[["Grupo", "Modelo"]] = df_ranks.apply(
                lambda r: parse_model_info(pd.Series({"model": r["Modelo_Core"], "dataset": "ETTh1"})), axis=1
            )
            df_ranks = df_ranks[["Grupo", "Modelo", "Ranking Médio"]]

            f.write("**Tabela de Ranking Médio (Menor é Melhor):**\n\n")
            f.write(df_ranks.to_html(classes="table", index=False, border=1, justify="center", float_format="%.2f"))
            f.write("\n\n<br><br>\n\n")

        dm_csv = os.path.join(statistical_path, "dm_test_results.csv")
        if os.path.exists(dm_csv):
            f.write("### Teste de Diebold-Mariano (Par a Par)\n\n")
            df_dm = pd.read_csv(dm_csv)
            f.write(df_dm.to_html(classes="table", index=False, border=1, justify="center"))
            f.write("\n\n<br><br>\n\n")
        f.write("---\n\n")

        # --- SEÇÃO FAMÍLIAS (GRID 2x2) ---
        f.write("## 📈 Dinâmica de Previsão por Famílias\n\n")
        family_path = os.path.join(plots_path, "families")
        family_plots = glob.glob(os.path.join(family_path, "*.png"))

        if not family_plots:
            f.write("> *Aviso: Nenhum gráfico de família foi encontrado.*\n\n")
        else:
            families = collections.defaultdict(dict)
            pattern = re.compile(r"family_(.+?)_(.+?)_(.+?)_h(\d+)\.png")

            for plot in family_plots:
                basename = os.path.basename(plot)
                match = pattern.match(basename)
                if match:
                    dataset, base, dl, horizon = match.groups()
                    key = f"Comparativo: {base.upper()} + {dl.upper()}"
                    families[key][int(horizon)] = os.path.relpath(plot, start=reports_path)

            for family_name, horizons_dict in sorted(families.items()):
                f.write(f"### {family_name}\n\n")

                img_96 = horizons_dict.get(96, "")
                img_192 = horizons_dict.get(192, "")
                img_336 = horizons_dict.get(336, "")
                img_720 = horizons_dict.get(720, "")

                grid_html = f"""
<table style="width:100%; text-align:center; border:none;">
  <tr>
    <td style="width:50%; border:none;"><b>H = 96</b><br><img src="{img_96}" alt="H=96"></td>
    <td style="width:50%; border:none;"><b>H = 192</b><br><img src="{img_192}" alt="H=192"></td>
  </tr>
  <tr>
    <td style="width:50%; border:none;"><b>H = 336</b><br><img src="{img_336}" alt="H=336"></td>
    <td style="width:50%; border:none;"><b>H = 720</b><br><img src="{img_720}" alt="H=720"></td>
  </tr>
</table>
<br>

"""
                f.write(grid_html)

        # --- SEÇÃO PERCENTAGE DIFFERENCE (NOVO GRÁFICO!) ---
        f.write("\n\n---\n\n## 🚀 Percentage Difference (PD) Global\n\n")
        f.write(
            "Este gráfico apresenta a Diferença Percentual (PD) em termos do erro médio (MAPE) entre o melhor modelo Híbrido proposto e os modelos da literatura, agregando todos os horizontes de previsão.\n\n"
        )

        pd_plots = glob.glob(os.path.join(pd_path, "*.png"))
        if not pd_plots:
            f.write("> *Aviso: Nenhum gráfico PD encontrado.*\n\n")
        else:
            for plot in sorted(pd_plots):
                rel_path = os.path.relpath(plot, start=reports_path)
                f.write(f"![Percentage Difference]({rel_path})\n\n")

    logging.info(f"Relatório Markdown atualizado em: {md_output}")


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

                plt.plot(
                    df_hybrid["ds"].tail(zoom_size),
                    df_hybrid["y"].tail(zoom_size),
                    label="Real (Ground Truth)",
                    color="black",
                    linewidth=2.5,
                    alpha=0.5,
                    zorder=1,
                )
                plt.plot(
                    df_base["ds"].tail(zoom_size),
                    df_base["y_hat"].tail(zoom_size),
                    label=f"Estatístico ({base_model.upper()})",
                    color="#1f77b4",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=2,
                    linestyle="--",
                )
                plt.plot(
                    df_dl["ds"].tail(zoom_size),
                    df_dl["y_hat"].tail(zoom_size),
                    label=f"Neural ({dl_model.upper()})",
                    color="#ff7f0e",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                    linestyle="-.",
                )
                plt.plot(
                    df_hybrid["ds"].tail(zoom_size),
                    df_hybrid["y_hat"].tail(zoom_size),
                    label=f"Híbrido ({dl_model.upper()} + {base_model.upper()})",
                    color="#d62728",
                    linewidth=2.5,
                    alpha=1.0,
                    zorder=4,
                )

                plt.title(
                    f"Comparativo de Desempenho: {base_model.upper()} vs {dl_model.upper()} vs Híbrido | Dataset: {dataset} | Horizonte: {horizon}"
                )
                plt.xlabel("Tempo")
                plt.ylabel("Valores")
                plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
                plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                plt.tight_layout()

                plt.savefig(
                    os.path.join(family_path, f"family_{dataset}_{base_model}_{dl_model}_h{horizon}.png"), dpi=150
                )
                plt.close()

    logging.info("Relatórios concluídos com sucesso.")
