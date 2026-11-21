"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/exp5_etica_xai.py
Síntese:
    Experimento 5 - aspectos éticos e explicabilidade do modelo final de cada base.
      (a) Auditoria de viés: taxa de detecção (recall) e taxa de falsos positivos (FPR)
          por gênero, faixa etária e quartil de renda (Base A) e por faixa etária (Base B).
      (b) Mitigação: o mesmo modelo treinado SEM atributos sensíveis (gênero, idade,
          renda, dívida, score de crédito), comparando desempenho e disparidades.
      (c) Explicabilidade: importância global (média do |contribuição|) e explicação
          local de um acerto e de um falso alarme (ver src/explicacao.py).

Uso:
    python src/exp5_etica_xai.py A|B [--refazer]

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import sys

import numpy as np
import pandas as pd

import comum as c
import explicacao as xai
from exp3_otimizacao import carregar_final, treinar

FAIXAS_IDADE = ([0, 30, 45, 60, 200], ["< 30", "30-44", "45-59", "60 ou mais"])


def grupos(chave, te):
    g = {"Faixa etária": pd.cut(te["idade"], FAIXAS_IDADE[0], labels=FAIXAS_IDADE[1], right=False).astype(str)}
    if chave == "A":
        g["Gênero"] = np.where(te["genero"] == 1, "Feminino", "Masculino")
        usuarios = te.drop_duplicates("usuario")["renda_anual"]
        cortes = usuarios.quantile([0.25, 0.5, 0.75]).to_numpy()
        g["Quartil de renda"] = pd.cut(te["renda_anual"], [-np.inf, *cortes, np.inf],
                                       labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"]).astype(str)
    return g


def por_grupo(y, pred, rotulos):
    linhas = []
    for nome, r in rotulos.items():
        for valor in sorted(pd.unique(r)):
            s = np.asarray(r) == valor
            yy, pp = y[s], pred[s]
            tp = int(((pp == 1) & (yy == 1)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
            neg = int((yy == 0).sum()); pos = int(yy.sum())
            linhas.append({"atributo": nome, "grupo": valor, "n": int(s.sum()), "fraudes": pos,
                           "recall": tp / pos if pos else np.nan, "fpr": fp / neg if neg else np.nan,
                           "precision": tp / (tp + fp) if tp + fp else np.nan,
                           "alertas_10k": 1e4 * (tp + fp) / s.sum()})
    return pd.DataFrame(linhas)


def disparidades(df):
    out = []
    for (modelo, atributo), d in df.groupby(["modelo", "atributo"]):
        # FPR em falsos alarmes por 10 mil transações legítimas; a razão fica indefinida se algum grupo tem FPR = 0
        out.append({"modelo": modelo, "atributo": atributo,
                    "dif_recall_max_min": d["recall"].max() - d["recall"].min(),
                    "fpr_10k_min": 1e4 * d["fpr"].min(), "fpr_10k_max": 1e4 * d["fpr"].max(),
                    "dif_fpr_10k_max_min": 1e4 * (d["fpr"].max() - d["fpr"].min()),
                    "razao_fpr_max_min": d["fpr"].max() / d["fpr"].min() if d["fpr"].min() > 0 else np.nan})
    return pd.DataFrame(out)


def executar(chave, refazer=False):
    saida = c.RESULTADOS / f"exp5_grupos_{chave}.csv"
    if saida.exists() and not refazer:
        return {k: pd.read_csv(c.RESULTADOS / f"exp5_{k}_{chave}.csv") for k in ("grupos", "resumo", "disparidades", "importancia")}
    m, meta = carregar_final(chave)
    partes, _ = c.carregar(chave)
    atributos = meta["atributos"]
    tr, val, te = partes["treino"], partes["validacao"], partes["teste"]
    y = te["is_fraud"].to_numpy()
    rot = grupos(chave, te)

    # (a) modelo final
    sens = c.BASES[chave]["sensiveis"]
    final_tem_sensiveis = any(a in atributos for a in sens)
    rot_final = "Com atributos sensíveis" if final_tem_sensiveis else "Sem atributos sensíveis"
    p = c.pontuar(m, te[atributos])
    pred = (p >= meta["limiar"]).astype(np.int8)
    g1 = por_grupo(y, pred, rot).assign(modelo=rot_final)
    r1 = {"modelo": rot_final, "final": True, "n_atributos": len(atributos), **c.metricas(y, p, meta["limiar"])}

    # (b) variante: mesmos hiperparâmetros, retirando (Base B) ou acrescentando (Base A) os atributos sensíveis
    if final_tem_sensiveis:
        atr2, rot_var = [a for a in atributos if a not in sens], "Sem atributos sensíveis"
    else:
        atr2, rot_var = atributos + sens, "Com atributos sensíveis"
    m2 = treinar(meta["familia"], meta["params"], chave, atr2, tr, val)
    p2_val = c.pontuar(m2, val[atr2])
    lim2 = c.melhor_limiar(val["is_fraud"].to_numpy(), p2_val)
    p2 = c.pontuar(m2, te[atr2])
    g2 = por_grupo(y, (p2 >= lim2).astype(np.int8), rot).assign(modelo=rot_var)
    from sklearn.metrics import average_precision_score
    r2 = {"modelo": rot_var, "final": False, "n_atributos": len(atr2), **c.metricas(y, p2, lim2)}
    r1["val_pr_auc"] = meta["val_pr_auc"]
    r2["val_pr_auc"] = average_precision_score(val["is_fraud"], p2_val)
    boot = c.bootstrap(y, {"final": (p, meta["limiar"]), "variante": (p2, lim2)}, n_boot=1000 if chave == "B" else 500)
    resumo = pd.DataFrame([r1, r2])
    for i, nome in enumerate(["final", "variante"]):
        for mt in ("f1", "pr_auc"):
            resumo.loc[i, f"{mt}_ic_inf"], resumo.loc[i, f"{mt}_ic_sup"] = c.ic(boot[f"{nome}|{mt}"])
    for mt in ("f1", "pr_auc"):
        dif = boot[f"variante|{mt}"] - boot[f"final|{mt}"]
        resumo.loc[1, f"dif_{mt}_vs_final"] = dif.mean()
        resumo.loc[1, f"dif_{mt}_ic_inf"], resumo.loc[1, f"dif_{mt}_ic_sup"] = c.ic(dif)
    df_g = pd.concat([g1, g2], ignore_index=True)
    disp = disparidades(df_g)

    # (c) explicabilidade do modelo final: todas as fraudes do teste + amostra de legítimas
    ref = xai.referencia_legitimas(tr, atributos)
    c.salvar_json(ref.to_dict(), c.MODELOS_DIR / f"referencia_{chave}.json")  # usada pelo protótipo
    rng = np.random.default_rng(c.SEMENTE)
    legit = np.nonzero(y == 0)[0]
    idx = np.r_[np.nonzero(y == 1)[0], rng.choice(legit, size=min(20000, len(legit)), replace=False)]
    contr = xai.contribuicoes(m, te.iloc[idx][atributos], ref)
    imp = contr.abs().mean().sort_values(ascending=False).rename("media_abs_contribuicao").to_frame()
    imp["metodo"] = xai.metodo(m)
    # exemplos locais: o acerto e o falso alarme de maior escore
    tp_i = int(np.argmax(np.where((pred == 1) & (y == 1), p, -1)))
    fp_i = int(np.argmax(np.where((pred == 1) & (y == 0), p, -1)))
    locais = []
    for tipo, i in [("Fraude detectada", tp_i), ("Falso alarme", fp_i)]:
        x = te.iloc[[i]][atributos]
        cl = xai.contribuicoes(m, x, ref).iloc[0]
        for n, v in cl.sort_values(key=np.abs, ascending=False).head(8).items():
            locais.append({"tipo": tipo, "escore": float(p[i]), "atributo": n, "rotulo": c.rotulo(n),
                           "valor": xai.formatar_valor(n, x.iloc[0][n]), "contribuicao": float(v)})

    df_g.to_csv(saida, index=False, float_format="%.6f")
    resumo.to_csv(c.RESULTADOS / f"exp5_resumo_{chave}.csv", index=False, float_format="%.6f")
    disp.to_csv(c.RESULTADOS / f"exp5_disparidades_{chave}.csv", index=False, float_format="%.6f")
    imp.to_csv(c.RESULTADOS / f"exp5_importancia_{chave}.csv", float_format="%.6f")
    pd.DataFrame(locais).to_csv(c.RESULTADOS / f"exp5_local_{chave}.csv", index=False, float_format="%.6f")
    print(resumo[["modelo", "f1", "pr_auc", "precision", "recall"]].to_string(index=False))
    print(disp.to_string(index=False))
    return {"grupos": df_g, "resumo": resumo, "disparidades": disp, "importancia": imp.reset_index()}


def figuras(chaves=("A", "B")):
    plt = c.preparar_matplotlib()
    # importância global
    fig, eixos = plt.subplots(1, len(chaves), figsize=(11, 4.4))
    for ax, chave in zip(np.atleast_1d(eixos), chaves):
        imp = pd.read_csv(c.RESULTADOS / f"exp5_importancia_{chave}.csv", index_col=0)
        meta = c.ler_json(c.MODELOS_DIR / f"modelo_final_{chave}.json")
        top = imp["media_abs_contribuicao"].head(10)[::-1]
        ax.barh([c.rotulo(n) for n in top.index], top.values, color=c.CORES.get(meta["familia"], "#eb6834"), height=0.6)
        ax.set_title(f"{c.BASES[chave]['nome']} - {meta['familia']}", fontsize=11, loc="left", color=c.TINTA)
        ax.set_xlabel(f"Média de |contribuição| em log-odds ({imp['metodo'].iloc[0]})", color=c.TINTA_2)
        c.estilo_eixo(ax, eixos_numericos="x", casas=2)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(c.RESULTADOS / "fig_exp5_importancia_global.png", dpi=200)
    plt.close(fig)
    # explicações locais (Base B e Base A)
    for chave in chaves:
        loc = pd.read_csv(c.RESULTADOS / f"exp5_local_{chave}.csv")
        fig, eixos = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, (tipo, d) in zip(eixos, loc.groupby("tipo", sort=False)):
            d = d.iloc[::-1]
            cores = ["#c0392b" if v > 0 else "#2a78d6" for v in d["contribuicao"]]
            ax.barh([f"{r} = {v}" for r, v in zip(d["rotulo"], d["valor"])], d["contribuicao"], color=cores, height=0.6)
            ax.axvline(0, color=c.TINTA, linewidth=0.8)
            ax.set_title(f"{tipo} (escore {c.br(d['escore'].iloc[0], 3)})", fontsize=10.5, loc="left", color=c.TINTA)
            ax.set_xlabel("Contribuição para o risco (log-odds)", color=c.TINTA_2)
            c.estilo_eixo(ax, fonte=8, eixos_numericos="x")
            ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(c.RESULTADOS / f"fig_exp5_local_{chave}.png", dpi=200)
        plt.close(fig)
    # disparidades por grupo (FPR e recall) na Base A
    if "A" in chaves:
        g = pd.read_csv(c.RESULTADOS / "exp5_grupos_A.csv")
        atributos = ["Gênero", "Faixa etária", "Quartil de renda"]
        fig, eixos = plt.subplots(2, 3, figsize=(11, 5.2), sharey="row")
        for j, atr in enumerate(atributos):
            d = g[g["atributo"] == atr]
            ordem = [g for g in FAIXAS_IDADE[1] if g in set(d["grupo"])] if atr == "Faixa etária" else list(dict.fromkeys(d["grupo"]))
            x = np.arange(len(ordem))
            for k, (modelo, cor) in enumerate([("Com atributos sensíveis", c.CORES["XGBoost"]),
                                               ("Sem atributos sensíveis", c.CORES["Random Forest"])]):
                dd = d[d["modelo"] == modelo].set_index("grupo").loc[ordem]
                eixos[0, j].bar(x + (k - 0.5) * 0.38, dd["recall"], width=0.36, color=cor, label=modelo)
                eixos[1, j].bar(x + (k - 0.5) * 0.38, 100 * dd["fpr"], width=0.36, color=cor, label=modelo)
            for i in range(2):
                eixos[i, j].set_xticks(x, ordem, fontsize=8)
                c.estilo_eixo(eixos[i, j], fonte=8, eixos_numericos="y", casas=2 if i else 1)
                eixos[i, j].grid(axis="x", visible=False)
            eixos[0, j].set_title(atr, fontsize=10.5, loc="left", color=c.TINTA)
        eixos[0, 0].set_ylabel("Recall", color=c.TINTA_2)
        eixos[1, 0].set_ylabel("Falsos positivos (%)", color=c.TINTA_2)
        h, l = eixos[0, 0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8.5, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(c.RESULTADOS / "fig_exp5_grupos_A.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    executar(sys.argv[1], refazer="--refazer" in sys.argv)
