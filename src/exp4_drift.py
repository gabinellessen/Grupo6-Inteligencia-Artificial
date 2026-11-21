"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/exp4_drift.py
Síntese:
    Experimento 4 - mudança de conceito (concept drift) e retreinamento na Base A.
    Avalia, mês a mês ao longo de 2019, três estratégias com o modelo final da Base A:
      1. estático (histórico até 2016): treinado só com o padrão antigo de fraude;
      2. estático (janela recente): treinado uma vez, de jan/2017 a set/2018;
      3. retreino mensal: no início de cada mês, treina com os 21 meses anteriores
         (exceto os 3 últimos, usados como validação para parada e limiar).
    Supõe que os rótulos de fraude de um mês estão disponíveis ao fim dele.

Uso:
    python src/exp4_drift.py [--refazer]

Histórico de alterações:
    2026-10-28 - Milton Almeida Leoncio - Comparação treino até 2016 x janela recente (N1).
    2026-11-20 - Milton Almeida Leoncio - Avaliação mensal e retreinamento com janela deslizante (N2).
"""
import gc
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import comum as c
from exp3_otimizacao import carregar_final, treinar

ESTRATEGIAS = ["Estático (histórico até 2016)", "Estático (janela recente)", "Retreino mensal"]


def resumo(y, p, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(y), "fraudes": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "pr_auc": average_precision_score(y, p) if y.sum() else np.nan}


def executar(refazer=False):
    saida = c.RESULTADOS / "exp4_mensal.csv"
    if saida.exists() and not refazer:
        return pd.read_csv(saida), pd.read_csv(c.RESULTADOS / "exp4_total.csv")
    _, meta = carregar_final("A")
    familia, params, atributos = meta["familia"], meta["params"], meta["atributos"]

    # 1) padrão antigo: treino com o histórico até 2016 (10% das legítimas, por memória)
    ph, _ = c.carregar("A", split_col="split_hist", fracao_legitimas=0.10)
    m = treinar(familia, params, "A", atributos, ph["treino"], ph["validacao"])
    lim_hist = c.melhor_limiar(ph["validacao"]["is_fraud"].to_numpy(), c.pontuar(m, ph["validacao"][atributos]))
    teste = ph["teste"]
    p_hist = c.pontuar(m, teste[atributos])
    del ph, m
    gc.collect()

    # 2) janela recente (modelo final do Experimento 3) e 3) retreino mensal
    partes, _ = c.carregar("A")
    teste = partes["teste"]
    assert len(teste) == len(p_hist)
    m_final, _ = carregar_final("A")
    p_rec = c.pontuar(m_final, teste[atributos])
    lim_rec = meta["limiar"]
    todos = pd.concat([partes["treino"], partes["validacao"], teste], ignore_index=True)
    del partes
    gc.collect()

    mes_teste = teste["ts"].dt.to_period("M")
    p_ret = np.zeros(len(teste))
    pred_ret = np.zeros(len(teste), dtype=np.int8)
    limiares = {}
    for mes in sorted(mes_teste.unique()):
        ini_mes = mes.to_timestamp()
        ini_tr, ini_val = ini_mes - pd.DateOffset(months=24), ini_mes - pd.DateOffset(months=3)
        tr = todos[(todos["ts"] >= ini_tr) & (todos["ts"] < ini_val)]
        val = todos[(todos["ts"] >= ini_val) & (todos["ts"] < ini_mes)]
        with c.Cronometro() as t:
            mm = treinar(familia, params, "A", atributos, tr, val)
        lim = c.melhor_limiar(val["is_fraud"].to_numpy(), c.pontuar(mm, val[atributos]))
        idx = np.nonzero((mes_teste == mes).to_numpy())[0]
        p_ret[idx] = c.pontuar(mm, teste.iloc[idx][atributos])
        pred_ret[idx] = (p_ret[idx] >= lim).astype(np.int8)
        limiares[str(mes)] = lim
        print(f"retreino {mes}: treino {len(tr):,} ({int(tr['is_fraud'].sum())} fraudes), "
              f"validação {len(val):,}; limiar {lim:.3f} ({t.s:.0f}s)", flush=True)
        del tr, val, mm
        gc.collect()

    y = teste["is_fraud"].to_numpy()
    estrategias = {ESTRATEGIAS[0]: (p_hist, (p_hist >= lim_hist).astype(np.int8)),
                   ESTRATEGIAS[1]: (p_rec, (p_rec >= lim_rec).astype(np.int8)),
                   ESTRATEGIAS[2]: (p_ret, pred_ret)}
    linhas, totais = [], []
    fraude = teste["is_fraud"] == 1
    for mes in sorted(mes_teste.unique()):
        sel = (mes_teste == mes).to_numpy()
        canal = {"frac_exterior": teste.loc[sel & fraude, "exterior"].mean(),
                 "frac_online": teste.loc[sel & fraude, "online"].mean()}
        for nome, (p, pred) in estrategias.items():
            linhas.append({"mes": str(mes), "estrategia": nome, **resumo(y[sel], p[sel], pred[sel]), **canal})
    for nome, (p, pred) in estrategias.items():
        totais.append({"estrategia": nome, **resumo(y, p, pred)})
    mensal, total = pd.DataFrame(linhas), pd.DataFrame(totais)
    mensal.to_csv(saida, index=False, float_format="%.6f")
    total.to_csv(c.RESULTADOS / "exp4_total.csv", index=False, float_format="%.6f")
    c.salvar_json({"limiares_retreino": limiares, "limiar_hist": lim_hist, "limiar_recente": lim_rec},
                  c.RESULTADOS / "exp4_limiares.json")
    return mensal, total


def figura():
    plt = c.preparar_matplotlib()
    mensal, _ = executar()
    cores = {ESTRATEGIAS[0]: "#898781", ESTRATEGIAS[1]: c.CORES["Random Forest"], ESTRATEGIAS[2]: c.CORES["XGBoost"]}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    meses = sorted(mensal["mes"].unique())
    # meses sem nenhuma fraude rotulada (nov. e dez./2019 na Base A) têm F1 indefinido e ficam fora do gráfico
    com_fraude = set(mensal.loc[mensal["fraudes"] > 0, "mes"])
    meses = [m for m in meses if m in com_fraude]
    x = np.arange(len(meses))
    for nome in ESTRATEGIAS:
        d = mensal[mensal["estrategia"] == nome].set_index("mes").loc[meses]
        ax.plot(x, d["f1"], marker="o", markersize=4, linewidth=2, color=cores[nome], label=nome)
    ax.set_xticks(x, [m[5:] + "/" + m[2:4] for m in meses])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score no mês", color=c.TINTA_2)
    ax.set_xlabel("Mês de 2019 (conjunto de teste da Base A)", color=c.TINTA_2)
    c.estilo_eixo(ax, eixos_numericos="y")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(c.RESULTADOS / "fig_exp4_drift_mensal.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    executar(refazer="--refazer" in sys.argv)
