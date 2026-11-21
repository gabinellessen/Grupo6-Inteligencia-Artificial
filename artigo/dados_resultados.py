"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: artigo/dados_resultados.py
Síntese:
    Lê os CSVs de resultados e monta as linhas das tabelas do artigo (com IC 95% e
    marcação de diferenças significativas), além de um dicionário de números usados
    no texto.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import json

import pandas as pd

from docx_sbc import br, milhar

A, B = "Base A - IBM", "Base B - Sparkov"


def ic_txt(v, inf, sup, casas=3):
    return f"{br(v, casas)} [{br(inf, casas)}; {br(sup, casas)}]"


def signif(inf, sup):
    return "*" if (inf > 0 or sup < 0) else ""


def dif_txt(v, inf, sup, casas=3):
    s = f"{'+' if v >= 0 else '−'}{br(abs(v), casas)}"
    return f"{s}{signif(inf, sup)}"


class Resultados:
    def __init__(self, RES):
        self.RES = RES
        self.e1 = {k: pd.read_csv(RES / f"exp1_metricas_{k}.csv") for k in ("A", "B")}
        self.e2 = {k: pd.read_csv(RES / f"exp2_metricas_{k}.csv") for k in ("A", "B")}
        self.e3 = {k: pd.read_csv(RES / f"exp3_metricas_{k}.csv") for k in ("A", "B")}
        self.e4m = pd.read_csv(RES / "exp4_mensal.csv")
        self.e4t = pd.read_csv(RES / "exp4_total.csv").set_index("estrategia")
        self.e5r = {k: pd.read_csv(RES / f"exp5_resumo_{k}.csv") for k in ("A", "B")}
        self.e5d = {k: pd.read_csv(RES / f"exp5_disparidades_{k}.csv") for k in ("A", "B")}
        self.e5g = {k: pd.read_csv(RES / f"exp5_grupos_{k}.csv") for k in ("A", "B")}
        self.e5i = {k: pd.read_csv(RES / f"exp5_importancia_{k}.csv", index_col=0) for k in ("A", "B")}
        self.e5l = {k: pd.read_csv(RES / f"exp5_local_{k}.csv") for k in ("A", "B")}
        self.final = {k: json.loads((RES / "modelos" / f"modelo_final_{k}.json").read_text(encoding="utf-8"))
                      for k in ("A", "B")}

    # -------------------------------------------------- Experimento 1
    def e1_final(self, chave):
        d = self.e1[chave]
        conj = "sem perfil estático" if chave == "A" else "completo"
        return d[d["conjunto"] == conj].set_index("modelo")

    def tabela_e1(self):
        modelos = ["Regras", "Regressão Logística", "Random Forest", "XGBoost", "LightGBM", "MLP", "SVM (RBF)"]
        linhas = []
        for chave in ("B", "A"):
            d = self.e1_final(chave)
            for m in modelos:
                if m not in d.index:
                    continue
                r = d.loc[m]
                linhas.append([("B – Sparkov" if chave == "B" else "A – IBM"), m, br(r["precision"]), br(r["recall"]),
                               ic_txt(r["f1"], r["f1_ic_inf"], r["f1_ic_sup"]),
                               ic_txt(r["pr_auc"], r["pr_auc_ic_inf"], r["pr_auc_ic_sup"]), br(r["roc_auc"])])
        return linhas

    def tabela_perfil(self):
        d = self.e1["A"]
        c, s = d[d["conjunto"] == "completo"].set_index("modelo"), d[d["conjunto"] == "sem perfil estático"].set_index("modelo")
        linhas = []
        for m in ["Regressão Logística", "Random Forest", "XGBoost", "LightGBM", "MLP"]:
            linhas.append([m, f"{br(c.loc[m, 'val_pr_auc'])} → {br(s.loc[m, 'val_pr_auc'])}",
                           f"{br(c.loc[m, 'pr_auc'])} → {br(s.loc[m, 'pr_auc'])}",
                           f"{br(c.loc[m, 'f1'])} → {br(s.loc[m, 'f1'])}",
                           dif_txt(s.loc[m, "dif_f1_vs_completo"], s.loc[m, "dif_f1_ic_inf"], s.loc[m, "dif_f1_ic_sup"])])
        return linhas

    # -------------------------------------------------- Experimento 2
    def tabela_e2(self):
        estr = ["Sem tratamento", "Ponderação de classes", "Subamostragem 1:10", "SMOTE 10%", "SMOTE + subamostragem"]
        linhas = []
        for e in estr:
            linha = [e]
            for chave in ("B", "A"):
                for m in ("Random Forest", "XGBoost"):
                    d = self.e2[chave]
                    r = d[(d["estrategia"] == e) & (d["modelo"] == m)].iloc[0]
                    txt = br(r["f1"])
                    if e != "Ponderação de classes":
                        txt += signif(r["dif_f1_ic_inf"], r["dif_f1_ic_sup"])
                    linha.append(txt)
            linhas.append(linha)
        return linhas

    # -------------------------------------------------- Experimento 3
    def tabela_e3(self):
        linhas = []
        for chave in ("B", "A"):
            d = self.e3[chave]
            for _, r in d.iterrows():
                nome = r["modelo"] + (" (final)" if r["final"] else "")
                linhas.append([("B – Sparkov" if chave == "B" else "A – IBM"), nome, str(int(r["tentativas"])),
                               br(r["val_pr_auc"]), f"{br(r['f1_padrao'])} → {br(r['f1'])}",
                               dif_txt(r["dif_f1_vs_padrao"], r["dif_f1_ic_inf"], r["dif_f1_ic_sup"]),
                               f"{br(r['pr_auc_padrao'])} → {br(r['pr_auc'])}"])
        return linhas

    # -------------------------------------------------- Experimento 4
    def tabela_e4(self):
        linhas = []
        for e, r in self.e4t.iterrows():
            linhas.append([e, br(r["precision"]), br(r["recall"]), br(r["f1"]), br(r["pr_auc"]),
                           milhar(r["tp"]), milhar(r["fp"])])
        return linhas

    # -------------------------------------------------- Experimento 5
    def tabela_e5(self):
        linhas = []
        for chave in ("A", "B"):
            r = self.e5r[chave]
            disp = self.e5d[chave].set_index(["modelo", "atributo"])
            for _, x in r.iterrows():
                atrs = sorted({a for (_, a) in disp.index})
                razao = "; ".join(f"{a}: {br(disp.loc[(x['modelo'], a), 'razao_fpr_max_min'], 2)}" for a in atrs)
                difr = "; ".join(f"{a}: {br(disp.loc[(x['modelo'], a), 'dif_recall_max_min'], 2)}" for a in atrs)
                linhas.append([("A – IBM" if chave == "A" else "B – Sparkov"),
                               x["modelo"] + (" (final)" if x["final"] else ""), br(x["f1"]), br(x["pr_auc"]),
                               razao, difr])
        return linhas
