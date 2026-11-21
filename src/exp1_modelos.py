"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/exp1_modelos.py
Síntese:
    Experimento 1 - comparação de classificadores na divisão temporal de cada base:
    um sistema de regras estáticas (linha de base "tradicional"), Regressão Logística,
    Random Forest, XGBoost, LightGBM, rede neural MLP e, só na Base B, SVM com kernel RBF.
    Para todos, o limiar de decisão é escolhido na validação (maior F1) e as métricas
    são medidas no teste, com intervalos de confiança de 95% por bootstrap.
    Na Base A, todos os modelos são treinados com dois conjuntos de atributos: o completo
    (N1) e o sem perfil estático do titular (renda, dívida, FICO, limite etc.), que se
    comportam como identificadores do cartão. O conjunto final de cada base é o de maior
    AUPRC na VALIDAÇÃO (Base A: sem perfil estático; Base B: completo).

Uso:
    python src/exp1_modelos.py A|B [--refazer]

Histórico de alterações:
    2026-10-28 - Milton Almeida Leoncio - Random Forest e XGBoost com ponderação de classes (N1).
    2026-11-20 - Milton Almeida Leoncio - Regras, Regressão Logística, LightGBM, MLP e SVM; bootstrap;
                                          comparação com e sem perfil estático na Base A.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import comum as c

MODELOS = ["Regras", "Regressão Logística", "Random Forest", "XGBoost", "LightGBM", "MLP", "SVM (RBF)"]


def flags_regras(df, chave, ref):
    """Regras heurísticas usuais em antifraude; os cortes (percentil 99) vêm só do treino (`ref`)."""
    q = lambda col: ref[col].quantile(0.99)
    f = {"valor alto": df["valor"] > q("valor"),
         "valor atípico p/ o cliente": df["valor_sobre_media_hist"] > q("valor_sobre_media_hist"),
         "muitas transações em 24h": df["n_trans_24h"] > q("n_trans_24h"),
         "gasto alto em 24h": df["valor_24h"] > q("valor_24h"),
         "madrugada (22h-3h)": df["hora"].isin([22, 23, 0, 1, 2, 3]),
         "comerciante novo": df["novo_comerciante"] == 1}
    if chave == "A":
        f.update({"compra no exterior": df["exterior"] == 1, "estado diferente": df["estado_diferente"] == 1,
                  "erro de segurança": df[["erro_pin", "erro_cvv", "erro_num_cartao", "erro_validade"]].sum(axis=1) > 0,
                  "compra online": df["online"] == 1})
    else:
        f.update({"categoria nova": df["novo_categoria_hist"] == 1})
    return pd.DataFrame(f).astype(np.int8)


def subamostrar(df, razao, semente=c.SEMENTE):
    """Mantém todas as fraudes e `razao` legítimas por fraude."""
    fraudes = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    n = min(len(legit), razao * len(fraudes))
    return pd.concat([fraudes, legit.sample(n=n, random_state=semente)]).sort_values("ts", kind="stable")


def construir(nome, chave, atributos, y_tr):
    cat = c.BASES[chave]["categoricas"]
    if nome == "Regressão Logística":
        return make_pipeline(c.preprocessador(atributos, cat),
                             LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0))
    if nome == "Random Forest":
        return c.rf_padrao()
    if nome == "XGBoost":
        return c.xgb_padrao(y_tr)
    if nome == "LightGBM":
        return c.lgbm_padrao(y_tr)
    if nome == "MLP":
        return make_pipeline(c.preprocessador(atributos, cat),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                           learning_rate_init=1e-3, max_iter=40, random_state=c.SEMENTE))
    if nome == "SVM (RBF)":
        return make_pipeline(c.preprocessador(atributos, cat),
                             SVC(C=1.0, gamma="scale", class_weight="balanced", cache_size=2000))
    raise ValueError(nome)


def treino_do_modelo(nome, chave, treino):
    """Conjunto de treino de cada modelo (limitações de custo documentadas no artigo)."""
    if nome == "MLP":  # sem ponderação de classes estável: subamostragem 1:50
        return subamostrar(treino, 50)
    if nome == "SVM (RBF)":  # custo O(n^2): subamostragem 1:20
        return subamostrar(treino, 20)
    if nome == "Regressão Logística" and chave == "A":  # memória do one-hot em 3 milhões de linhas
        return subamostrar(treino, 400)
    return treino


def executar(chave, refazer=False, n_boot=None):
    n_boot = n_boot or (1000 if chave == "B" else 500)  # Base A: 1,7 milhão de linhas no teste
    saida = c.RESULTADOS / f"exp1_metricas_{chave}.csv"
    arq_pred = c.pasta_atributos() / f"pred_exp1_{chave}.parquet"
    if saida.exists() and arq_pred.exists() and not refazer:
        return pd.read_csv(saida), pd.read_parquet(arq_pred)
    from sklearn.metrics import average_precision_score
    partes, todos = c.carregar(chave)
    tr, val, te = partes["treino"], partes["validacao"], partes["teste"]
    linhas, pred = [], {"y": te["is_fraud"].to_numpy(np.int8)}
    for conjunto in CONJUNTOS[chave]:
        atributos = c.atributos_conjunto(chave, todos, conjunto)
        for nome in MODELOS:
            if nome == "SVM (RBF)" and chave == "A":
                continue  # inviável em 3 milhões de transações (ver artigo)
            with c.Cronometro() as t:
                if nome == "Regras":
                    p_val = flags_regras(val, chave, tr).sum(axis=1).to_numpy(float)
                    p_te = flags_regras(te, chave, tr).sum(axis=1).to_numpy(float)
                    n_tr, extra = len(tr), {}
                else:
                    dtr = treino_do_modelo(nome, chave, tr)
                    n_tr = len(dtr)
                    m = construir(nome, chave, atributos, dtr["is_fraud"].to_numpy())
                    c.ajustar_com_parada(m, dtr[atributos], dtr["is_fraud"].to_numpy(), val[atributos],
                                         val["is_fraud"].to_numpy())
                    p_val, p_te = c.pontuar(m, val[atributos]), c.pontuar(m, te[atributos])
                    extra = {"n_arvores": c.n_arvores(m)}
            limiar = c.melhor_limiar(val["is_fraud"].to_numpy(), p_val)
            met = c.metricas(te["is_fraud"].to_numpy(), p_te, limiar)
            ident = id_pred(nome, conjunto)
            linhas.append({"base": c.BASES[chave]["nome"], "conjunto": conjunto, "n_atributos": len(atributos),
                           "modelo": nome, "id": ident, "linhas_treino": n_tr, "tempo_s": round(t.s, 1),
                           "val_pr_auc": average_precision_score(val["is_fraud"], p_val), **met, **extra})
            pred[ident] = p_te.astype(np.float64)  # float64: float32 cria empates no limiar em modelos de poucas árvores
            print(f"[{chave}] {conjunto:20s} {nome:20s} val AUPRC={linhas[-1]['val_pr_auc']:.3f} | teste F1={met['f1']:.3f} "
                  f"P={met['precision']:.3f} R={met['recall']:.3f} AUPRC={met['pr_auc']:.3f} ({t.s:.0f}s)", flush=True)
    pred_df = pd.DataFrame(pred)
    met_df = pd.DataFrame(linhas)
    # intervalos de confiança (bootstrap pareado)
    boot = c.bootstrap(pred_df["y"], {r["id"]: (pred_df[r["id"]], r["limiar"]) for r in linhas}, n_boot=n_boot)
    for i, r in met_df.iterrows():
        for m in ("f1", "pr_auc"):
            met_df.loc[i, f"{m}_ic_inf"], met_df.loc[i, f"{m}_ic_sup"] = c.ic(boot[f"{r['id']}|{m}"])
            if r["conjunto"] != "completo":  # ganho pareado em relação ao conjunto completo
                dif = boot[f"{r['id']}|{m}"] - boot[f"{id_pred(r['modelo'], 'completo')}|{m}"]
                met_df.loc[i, f"dif_{m}_vs_completo"] = dif.mean()
                met_df.loc[i, f"dif_{m}_ic_inf"], met_df.loc[i, f"dif_{m}_ic_sup"] = c.ic(dif)
    boot.to_csv(c.RESULTADOS / f"exp1_bootstrap_{chave}.csv.gz", index=False, float_format="%.5f")
    met_df.to_csv(saida, index=False, float_format="%.6f")
    pred_df.to_parquet(arq_pred, index=False)
    return met_df, pred_df


CONJUNTOS = {"A": ["completo", "sem perfil estático"], "B": ["completo"]}


def id_pred(modelo, conjunto):
    return modelo if conjunto == "completo" else f"{modelo} [{conjunto}]"


def finais(chave):
    """Linhas do Experimento 1 no conjunto de atributos final da base."""
    met, pred = executar(chave)
    return met[met["conjunto"] == c.BASES[chave]["conjunto_final"]].reset_index(drop=True), pred


def figuras(chaves=("A", "B")):
    from sklearn.metrics import precision_recall_curve
    plt = c.preparar_matplotlib()
    fig, eixos = plt.subplots(1, len(chaves), figsize=(11, 4.4))
    for ax, chave in zip(np.atleast_1d(eixos), chaves):
        met, pred = finais(chave)
        y = pred["y"].to_numpy()
        for _, r in met.iterrows():
            prec, rec, _ = precision_recall_curve(y, pred[r["id"]])
            ax.plot(rec, prec, color=c.CORES[r["modelo"]], linewidth=1.6,
                    label=f"{r['modelo']} ({c.br(r['pr_auc'], 3)})")
        ax.axhline(y.mean(), color="#b9b8b0", linestyle="--", linewidth=1)
        ax.set_title(c.BASES[chave]["nome"], fontsize=11, loc="left", color=c.TINTA)
        ax.set_xlabel("Revocação (recall)", color=c.TINTA_2)
        ax.set_ylabel("Precisão", color=c.TINTA_2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        c.estilo_eixo(ax)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", title="AUPRC no teste", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(c.RESULTADOS / "fig_exp1_curvas_pr.png", dpi=200)
    plt.close(fig)

    fig, eixos = plt.subplots(1, len(chaves), figsize=(11, 3.8))
    for ax, chave in zip(np.atleast_1d(eixos), chaves):
        met, _ = finais(chave)
        met = met.iloc[::-1]
        ypos = np.arange(len(met))
        ax.barh(ypos, met["f1"], color=[c.CORES[m] for m in met["modelo"]], height=0.6)
        ax.errorbar(met["f1"], ypos, xerr=[(met["f1"] - met["f1_ic_inf"]).clip(lower=0), (met["f1_ic_sup"] - met["f1"]).clip(lower=0)],
                    fmt="none", ecolor=c.TINTA, elinewidth=1, capsize=3)
        for yy, v, sup in zip(ypos, met["f1"], met["f1_ic_sup"]):
            ax.text(min(max(v, sup) + 0.02, 0.95), yy, c.br(v, 2), va="center", fontsize=8, color=c.TINTA_2)
        ax.set_yticks(ypos, met["modelo"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1-score no teste (IC 95%)", color=c.TINTA_2)
        ax.set_title(c.BASES[chave]["nome"], fontsize=11, loc="left", color=c.TINTA)
        c.estilo_eixo(ax, eixos_numericos="x")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(c.RESULTADOS / "fig_exp1_f1_ic.png", dpi=200)
    plt.close(fig)

    # efeito do perfil estático na Base A (validação x teste, por modelo)
    if "A" in chaves:
        met, _ = executar("A")
        modelos = [m for m in MODELOS if m in set(met["modelo"]) and m != "Regras"]
        fig, eixos = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        for ax, (col, titulo) in zip(eixos, [("val_pr_auc", "AUPRC na validação (out-dez/2018)"),
                                             ("pr_auc", "AUPRC no teste (2019)")]):
            x = np.arange(len(modelos))
            for k, (conj, cor) in enumerate([("completo", "#c3c2b7"), ("sem perfil estático", c.CORES["XGBoost"])]):
                d = met[met["conjunto"] == conj].set_index("modelo").loc[modelos]
                ax.bar(x + (k - 0.5) * 0.38, d[col], width=0.36, color=cor,
                       label="todos os atributos (N1)" if conj == "completo" else "sem perfil estático do titular")
            ax.set_xticks(x, [m.replace("Regressão Logística", "Reg. Logística") for m in modelos], fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(titulo, fontsize=10.5, loc="left", color=c.TINTA)
            c.estilo_eixo(ax, eixos_numericos="y")
            ax.grid(axis="x", visible=False)
        eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(c.RESULTADOS / "fig_exp1_perfil_estatico_A.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    executar(sys.argv[1], refazer="--refazer" in sys.argv)
