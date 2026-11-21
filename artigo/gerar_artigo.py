"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: artigo/gerar_artigo.py
Síntese:
    Gera o artigo final (artigo/Artigo_Final_IA_Fraudes.docx) no template SBC a partir
    dos textos (texto_fixo.py, metodologia.py, resultados.py) e dos resultados em
    resultados/. Com --pdf, exporta também o PDF usando o Microsoft Word (Windows).

Uso:
    python artigo/gerar_artigo.py [--pdf]

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import statistics
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "prototipo"))

import metodologia  # noqa: E402
import resultados  # noqa: E402
import texto_fixo as T  # noqa: E402
from dados_resultados import Resultados  # noqa: E402
from docx_sbc import Artigo, br  # noqa: E402

RES = RAIZ / "resultados"
SAIDA = AQUI / "Artigo_Final_IA_Fraudes.docx"


def numeros_prototipo(dia="2020-10-15"):
    """Roda o detector no período de teste da Base B (dia a dia) para os números da Seção 4.2.6."""
    from detector import DetectorFraude, periodo_de_teste
    det = DetectorFraude("B")
    fluxo = periodo_de_teste("B")
    fluxo["dia"] = fluxo["ts"].dt.date.astype(str)
    tot = {"transacoes": 0, "alertas": 0, "detectadas": 0, "falsos": 0, "perdidas": 0}
    tempos, p = [], {}
    for d, lote in fluxo.groupby("dia"):
        t0 = time.perf_counter()
        r = det.avaliar(lote)
        tempos.append(1000 * (time.perf_counter() - t0) / len(lote))
        y, a = lote["is_fraud"].to_numpy() == 1, r["alerta"].to_numpy()
        tot["transacoes"] += len(lote); tot["alertas"] += int(a.sum()); tot["detectadas"] += int((a & y).sum())
        tot["falsos"] += int((a & ~y).sum()); tot["perdidas"] += int((~a & y).sum())
        if d == dia:
            p = {"dia": f"{d[8:]}/{d[5:7]}/{d[:4]}", "dia_transacoes": len(lote), "dia_alertas": int(a.sum())}
    return {**tot, **p, "dias": fluxo["dia"].nunique(), "ms": statistics.median(tempos)}


def resumo_e_abstract(R):
    fA, fB = R.final["A"], R.final["B"]
    e1A = R.e1["A"].set_index(["conjunto", "modelo"])
    ret = R.e4t.loc["Retreino mensal", "f1"]
    # SBC: no máximo 10 linhas em cada resumo
    pt = (f"Este trabalho aplica aprendizado de máquina supervisionado à detecção de fraudes com cartão em duas bases "
          f"públicas sintéticas (IBM e Sparkov), com divisão temporal entre treino, validação e teste. Foram comparados "
          f"regras estáticas e seis classificadores, técnicas de balanceamento e otimização com Optuna. O modelo final "
          f"(LightGBM) obteve F1-score de {br(fB['teste']['f1'], 2)} na base Sparkov e de {br(fA['teste']['f1'], 2)} na "
          f"base IBM, contra {br(e1A.loc[('completo', 'Random Forest'), 'f1'], 2)} na etapa anterior. Esse ganho veio da "
          f"remoção de atributos de perfil que permitiam memorizar cartões fraudados. Com retreino mensal, o F1 chegou "
          f"a {br(ret, 2)}. A auditoria de vieses mostrou que o modelo sem "
          f"atributos sensíveis é mais preciso e mais equilibrado, e um protótipo explica cada alerta ao analista.")
    en = (f"This work applies supervised machine learning to credit card fraud detection on two public synthetic "
          f"datasets (IBM and Sparkov), using a temporal split into training, validation and test sets. Static rules "
          f"and six classifiers were compared, together with class-imbalance techniques and Optuna tuning. The final "
          f"model (LightGBM) reached an F1-score of {fB['teste']['f1']:.2f} on Sparkov and {fA['teste']['f1']:.2f} on "
          f"IBM, up from {e1A.loc[('completo', 'Random Forest'), 'f1']:.2f} in the previous stage. The gain came from "
          f"removing profile attributes that let the model memorize defrauded cards. Monthly retraining raised the "
          f"F1-score to {ret:.2f} under concept drift. A bias audit showed that the model without sensitive attributes "
          f"is both more accurate and fairer, and a prototype explains every alert to the analyst.")
    return pt, en


def gerar(pdf=False):
    R = Resultados(RES)
    art = Artigo(AQUI / "sbc_template.docx")
    art.titulo(T.TITULO)
    art.autores(T.AUTORES)
    art.endereco(T.ENDERECO)
    art.email(T.EMAILS)
    pt, en = resumo_e_abstract(R)
    art.resumo("Abstract", en)
    art.resumo("Resumo", pt)

    art.h1("1. Introdução")
    art.h2("1.1. Contextualização")
    for p in T.CONTEXTUALIZACAO:
        art.par(p)
    art.h2("1.2. Justificativa")
    for p in T.JUSTIFICATIVA:
        art.par(p)
    art.h2("1.3. Motivação")
    for p in T.MOTIVACAO:
        art.par(p)
    art.h2("1.4. Objetivo")
    for p in T.OBJETIVO:
        art.par(p)
    art.h2("1.5. Contribuições alcançadas")
    fA, fB = R.final["A"], R.final["B"]
    art.par("As contribuições alcançadas são quatro. A primeira é um *pipeline* reprodutível de detecção de "
            "fraudes, com avaliação temporal e intervalos de confiança. O modelo final obteve F1-score de "
            f"{br(fB['teste']['f1'], 2)} na base Sparkov e de {br(fA['teste']['f1'], 2)} na base IBM.")
    art.par("A segunda é a identificação de um vazamento por identidade na base IBM: atributos de perfil do titular "
            "levavam o modelo a memorizar cartões já fraudados. A terceira é a medição do efeito da mudança de "
            "conceito e do ganho do retreinamento mensal. A quarta é uma auditoria de vieses com medidas de "
            "mitigação, somada a um protótipo que explica cada alerta ao analista.")
    art.par(T.ORGANIZACAO)

    art.h1("2. Fundamentação Teórica")
    for titulo, pars in T.FUNDAMENTOS.items():
        art.h2(titulo)
        for p in pars:
            art.par(p)
    art.h2("2.6. Trabalhos relacionados")
    for p in T.RELACIONADOS_INTRO:
        art.par(p.replace("A Tabela 2", f"A Tabela {art.proxima_tabela()}"))
    art.tabela(["Trabalho", "Base de dados", "Solução", "Resultado relatado"], T.RELACIONADOS,
               "Trabalhos relacionados e resultados relatados pelos autores.",
               larguras_cm=[2.9, 3.6, 5.0, 3.5], fonte=8, alinhar=["l", "l", "l", "l"])
    for p in T.RELACIONADOS_DISCUSSAO:
        art.par(p)

    metodologia.secao(art, RES)
    resultados.secao(art, RES, numeros_prototipo())
    resultados.conclusao(art, RES)
    art.referencias(T.REFERENCIAS)
    art.salvar(SAIDA)
    print("artigo gravado em", SAIDA)
    if pdf:
        exportar_pdf(SAIDA, SAIDA.with_suffix(".pdf"))


def exportar_pdf(docx, pdf):
    import win32com.client  # pywin32
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(docx), ReadOnly=True)
        d.ExportAsFixedFormat(str(pdf), 17)  # 17 = PDF
        d.Close(False)
    finally:
        word.Quit()
    print("PDF gravado em", pdf)


if __name__ == "__main__":
    gerar(pdf="--pdf" in sys.argv)
