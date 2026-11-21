"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/eda.py
Síntese:
    Análise exploratória das duas bases a partir dos parquets preparados: resumo e
    divisão temporal, qualidade (valores ausentes e duplicados), taxa de fraude por
    hora, canal das fraudes por ano na Base A (mudança de conceito), distribuição dos
    valores e taxa de fraude por categoria de comércio na Base B.

Histórico de alterações:
    2026-10-28 - Milton Almeida Leoncio - Taxa de fraude por hora e canal das fraudes por ano (N1).
    2026-11-20 - Milton Almeida Leoncio - Qualidade dos dados, valores e categorias; módulo único (N2).
"""
import numpy as np
import pandas as pd

import comum as c


def ler(chave, colunas):
    return pd.read_parquet(c.pasta_atributos() / c.BASES[chave]["arquivo"], columns=colunas)


def resumo():
    linhas = {}
    for chave in ("A", "B"):
        cols = ["ts", "is_fraud", "split"] + (["cartao", "usuario"] if chave == "A" else ["cliente"])
        d = ler(chave, cols)
        r = {"Transações (após preparação)": len(d), "Fraudes": int(d["is_fraud"].sum()),
             "% de fraudes": round(100 * d["is_fraud"].mean(), 3),
             "Início": d["ts"].min().date(), "Fim": d["ts"].max().date(),
             "Titulares": d["usuario" if chave == "A" else "cliente"].nunique()}
        for s in ("treino", "validacao", "teste"):
            p = d[d["split"] == s]
            r[f"{s}: transações"] = len(p)
            r[f"{s}: fraudes"] = int(p["is_fraud"].sum())
            r[f"{s}: período"] = f"{p['ts'].min():%m/%Y} a {p['ts'].max():%m/%Y}"
        linhas[c.BASES[chave]["nome"]] = r
    df = pd.DataFrame(linhas)
    df.to_csv(c.RESULTADOS / "eda_resumo_bases.csv")
    return df


def qualidade():
    out = []
    for chave in ("A", "B"):
        d = pd.read_parquet(c.pasta_atributos() / c.BASES[chave]["arquivo"],
                            filters=[("split", "!=", "fora")] if chave == "A" else None)
        atr = [x for x in d.columns if x not in c.NAO_ATRIBUTOS]
        out.append({"base": c.BASES[chave]["nome"], "linhas": len(d), "atributos": len(atr),
                    "valores ausentes": int(d[atr].isna().sum().sum()),
                    "linhas duplicadas": int(d.duplicated().sum()),
                    "valores negativos (estornos)": int((d["valor"] < 0).sum())})
    df = pd.DataFrame(out)
    df.to_csv(c.RESULTADOS / "eda_qualidade.csv", index=False)
    return df


def concentracao_fraudes():
    """Quantos cartões/clientes concentram as fraudes de cada período e quantos se repetem entre treino e teste.

    Evidência do vazamento por identidade: se as fraudes do teste ocorrem em titulares diferentes dos do treino,
    atributos que identificam o titular não ajudam a generalizar.
    """
    linhas = []
    for chave, grupo in (("A", "cartao"), ("B", "cliente")):
        d = pd.read_parquet(c.pasta_atributos() / c.BASES[chave]["arquivo"], columns=[grupo, "is_fraud", "split", "ts"],
                            filters=[("split", "!=", "fora")])
        com_fraude = {}
        for s in ("treino", "validacao", "teste"):
            p = d[d["split"] == s]
            f = p[p["is_fraud"] == 1]
            com_fraude[s] = set(f[grupo])
            surto = f.groupby(grupo)["ts"].agg(lambda x: (x.max() - x.min()).days)
            linhas.append({"base": c.BASES[chave]["nome"], "periodo": s, "fraudes": len(f),
                           "titulares_ativos": p[grupo].nunique(), "titulares_com_fraude": len(com_fraude[s]),
                           "fraudes_por_titular_mediana": float(f.groupby(grupo).size().median()),
                           "duracao_surto_dias_mediana": float(surto.median()),
                           "titulares_com_fraude_tambem_no_treino": len(com_fraude[s] & com_fraude["treino"]) if s != "treino" else None})
    df = pd.DataFrame(linhas)
    df.to_csv(c.RESULTADOS / "eda_concentracao_fraudes.csv", index=False)
    return df


def figuras():
    plt = c.preparar_matplotlib()
    from matplotlib.ticker import FuncFormatter
    # 1. taxa de fraude por hora
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for chave, cor in (("A", c.CORES["Random Forest"]), ("B", c.CORES["XGBoost"])):
        d = ler(chave, ["hora", "is_fraud"])
        t = d.groupby("hora")["is_fraud"].mean()
        ax.plot(t.index, 100 * t.values, color=cor, linewidth=2, marker="o", markersize=4, label=c.BASES[chave]["nome"])
    ax.set_xlabel("Hora do dia", color=c.TINTA_2); ax.set_ylabel("Fraudes (% das transações)", color=c.TINTA_2)
    ax.set_xticks(range(0, 24, 2))
    c.estilo_eixo(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(c.RESULTADOS / "fig_eda_fraude_por_hora.png", dpi=200); plt.close(fig)

    # 2. canal das fraudes por ano (Base A) - evidência da mudança de conceito
    d = ler("A", ["ano", "is_fraud", "online", "exterior"])
    f = d[d["is_fraud"] == 1]
    canal = f.groupby("ano")[["online", "exterior"]].mean()
    canal["fraudes"] = f.groupby("ano").size()
    canal.to_csv(c.RESULTADOS / "eda_canal_fraudes_ano_A.csv")
    cc = canal[canal.index >= 2004]
    partes = {"Compra online": cc["online"], "Compra no exterior": cc["exterior"],
              "Presencial no país": 1 - cc["online"] - cc["exterior"]}
    fig, ax = plt.subplots(figsize=(10, 3.4))
    base = np.zeros(len(cc))
    for (rot, v), cor in zip(partes.items(), ["#2a78d6", "#eb6834", "#c3c2b7"]):
        ax.bar(cc.index, 100 * v, bottom=base, color=cor, width=0.8, label=rot, edgecolor="white", linewidth=0.8)
        base += 100 * v.to_numpy()
    for corte in (2016.5, 2018.5):
        ax.axvline(corte, color=c.TINTA, linestyle="--", linewidth=1)
    ax.text(2016.4, 104, "treino histórico: até 2016", ha="right", va="bottom", fontsize=9, color=c.TINTA_2)
    ax.text(2017.5, 104, "treino/val.", ha="center", va="bottom", fontsize=9, color=c.TINTA_2)
    ax.text(2019, 104, "teste", ha="center", va="bottom", fontsize=9, color=c.TINTA_2)
    ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("% das fraudes do ano", color=c.TINTA_2)
    c.estilo_eixo(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(c.RESULTADOS / "fig_eda_canal_fraudes_A.png", dpi=200); plt.close(fig)

    # 3. valores: fraude x legítima (escala log) e taxa de fraude por categoria (Base B)
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, chave in zip(eixos[:2], ("A", "B")):
        d = ler(chave, ["valor", "is_fraud"])
        d = d[d["valor"] > 0]
        bins = np.logspace(0, np.log10(d["valor"].quantile(0.9995)), 50)
        for rot, cor, sel in (("Legítimas", "#898781", d["is_fraud"] == 0), ("Fraudes", "#c0392b", d["is_fraud"] == 1)):
            ax.hist(d.loc[sel, "valor"], bins=bins, density=True, histtype="step", linewidth=1.6, color=cor, label=rot)
        ax.set_xscale("log")
        ax.set_title(c.BASES[chave]["nome"], fontsize=10.5, loc="left", color=c.TINTA)
        ax.set_xlabel("Valor da transação (US$, escala log)", color=c.TINTA_2)
        ax.set_ylabel("Densidade", color=c.TINTA_2)
        c.estilo_eixo(ax, fonte=8, eixos_numericos="")
        ax.legend(frameon=False, fontsize=8)
    bruto = pd.read_parquet(c.pasta_atributos() / "base_b_sparkov.parquet", columns=["categoria", "is_fraud"])
    t = bruto.groupby("categoria")["is_fraud"].mean().sort_values()
    eixos[2].barh([c.CATEGORIAS_SPARKOV[i] for i in t.index], 100 * t.values, color=c.CORES["XGBoost"], height=0.6)
    eixos[2].set_xlabel("Fraudes (% das transações)", color=c.TINTA_2)
    eixos[2].set_title("Base B - taxa por categoria", fontsize=10.5, loc="left", color=c.TINTA)
    c.estilo_eixo(eixos[2], fonte=8, eixos_numericos="x")
    eixos[2].grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(c.RESULTADOS / "fig_eda_valores_categorias.png", dpi=200); plt.close(fig)
    return canal
