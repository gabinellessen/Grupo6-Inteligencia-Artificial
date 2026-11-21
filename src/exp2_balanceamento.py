"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/exp2_balanceamento.py
Síntese:
    Experimento 2 - tratamento do desbalanceamento de classes. Compara, com Random
    Forest e XGBoost (configuração da N1), cinco estratégias aplicadas SOMENTE ao
    conjunto de treino: sem tratamento, ponderação de classes, subamostragem aleatória
    (1 fraude : 10 legítimas), SMOTE (fraudes = 10% das legítimas) e SMOTE seguido de
    subamostragem (razão final 1:2). Validação e teste mantêm a proporção real.

Uso:
    python src/exp2_balanceamento.py A|B [--refazer]

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import comum as c

ESTRATEGIAS = ["Sem tratamento", "Ponderação de classes", "Subamostragem 1:10", "SMOTE 10%", "SMOTE + subamostragem"]


def reamostrar(estrategia, X, y):
    if estrategia == "Subamostragem 1:10":
        return RandomUnderSampler(sampling_strategy=0.1, random_state=c.SEMENTE).fit_resample(X, y)
    if estrategia == "SMOTE 10%":
        return SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=c.SEMENTE).fit_resample(X, y)
    if estrategia == "SMOTE + subamostragem":
        X2, y2 = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=c.SEMENTE).fit_resample(X, y)
        return RandomUnderSampler(sampling_strategy=0.5, random_state=c.SEMENTE).fit_resample(X2, y2)
    return X, y


def construir(modelo, estrategia, y_tr):
    ponderar = estrategia == "Ponderação de classes"
    if modelo == "XGBoost":
        return c.xgb_padrao(y_tr, scale_pos_weight=c.peso_positivo(y_tr) if ponderar else 1.0)
    rf = c.rf_padrao()
    rf.set_params(class_weight="balanced_subsample" if ponderar else None)
    return rf


def executar(chave, refazer=False, n_boot=None):
    n_boot = n_boot or (1000 if chave == "B" else 500)  # Base A: 1,7 milhão de linhas no teste
    saida = c.RESULTADOS / f"exp2_metricas_{chave}.csv"
    if saida.exists() and not refazer:
        return pd.read_csv(saida)
    partes, todos = c.carregar(chave)
    atributos = c.atributos_finais(chave, todos)  # conjunto escolhido no Experimento 1
    tr, val, te = partes["treino"], partes["validacao"], partes["teste"]
    y_val, y_te = val["is_fraud"].to_numpy(), te["is_fraud"].to_numpy()
    linhas, preds = [], {}
    for estrategia in ESTRATEGIAS:
        X_r, y_r = reamostrar(estrategia, tr[atributos], tr["is_fraud"].to_numpy())
        for modelo in ["Random Forest", "XGBoost"]:
            with c.Cronometro() as t:
                m = construir(modelo, estrategia, y_r)
                c.ajustar_com_parada(m, X_r, y_r, val[atributos], y_val)
                p_val, p_te = c.pontuar(m, val[atributos]), c.pontuar(m, te[atributos])
            limiar = c.melhor_limiar(y_val, p_val)
            met = c.metricas(y_te, p_te, limiar)
            nome = f"{modelo} | {estrategia}"
            preds[nome] = (p_te, limiar)
            linhas.append({"base": c.BASES[chave]["nome"], "modelo": modelo, "estrategia": estrategia,
                           "linhas_treino": len(y_r), "fraudes_treino": int(np.sum(y_r)),
                           "tempo_s": round(t.s, 1), **met})
            print(f"[{chave}] {nome:45s} F1={met['f1']:.3f} AUPRC={met['pr_auc']:.3f} "
                  f"P={met['precision']:.3f} R={met['recall']:.3f} ({t.s:.0f}s)", flush=True)
        del X_r, y_r
    df = pd.DataFrame(linhas)
    boot = c.bootstrap(y_te, preds, n_boot=n_boot)
    for i, r in df.iterrows():
        nome = f"{r['modelo']} | {r['estrategia']}"
        ref = f"{r['modelo']} | Ponderação de classes"
        for m in ("f1", "pr_auc"):
            df.loc[i, f"{m}_ic_inf"], df.loc[i, f"{m}_ic_sup"] = c.ic(boot[f"{nome}|{m}"])
            dif = boot[f"{nome}|{m}"] - boot[f"{ref}|{m}"]  # diferença pareada em relação à ponderação
            df.loc[i, f"dif_{m}_vs_ponderacao"] = dif.mean()
            df.loc[i, f"dif_{m}_ic_inf"], df.loc[i, f"dif_{m}_ic_sup"] = c.ic(dif)
    df.to_csv(saida, index=False, float_format="%.6f")
    return df


def figura(chaves=("A", "B")):
    plt = c.preparar_matplotlib()
    fig, eixos = plt.subplots(1, len(chaves), figsize=(11, 4.0), sharey=True)
    for ax, chave in zip(np.atleast_1d(eixos), chaves):
        df = executar(chave)
        x = np.arange(len(ESTRATEGIAS))
        for k, modelo in enumerate(["Random Forest", "XGBoost"]):
            d = df[df["modelo"] == modelo].set_index("estrategia").loc[ESTRATEGIAS]
            ax.bar(x + (k - 0.5) * 0.38, d["f1"], width=0.36, color=c.CORES[modelo], label=modelo)
            ax.errorbar(x + (k - 0.5) * 0.38, d["f1"], yerr=[(d["f1"] - d["f1_ic_inf"]).clip(lower=0), (d["f1_ic_sup"] - d["f1"]).clip(lower=0)],
                        fmt="none", ecolor=c.TINTA, elinewidth=0.9, capsize=2)
        ax.set_xticks(x, [e.replace(" + ", " +\n").replace("Ponderação de classes", "Ponderação\nde classes")
                          .replace("Subamostragem 1:10", "Subamostragem\n1:10").replace("Sem tratamento", "Sem\ntratamento")
                          for e in ESTRATEGIAS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-score no teste (IC 95%)", color=c.TINTA_2)
        ax.set_title(c.BASES[chave]["nome"], fontsize=11, loc="left", color=c.TINTA)
        c.estilo_eixo(ax, eixos_numericos="y")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(c.RESULTADOS / "fig_exp2_balanceamento.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    executar(sys.argv[1], refazer="--refazer" in sys.argv)
