"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: prototipo/simular_fluxo.py
Síntese:
    Simulação, no terminal, do protótipo em operação: as transações do período de teste
    chegam dia a dia em ordem cronológica, o detector calcula o risco de cada uma e
    mostra os alertas com seus motivos, além de um resumo diário (alertas, fraudes
    detectadas, falsos alarmes e fraudes não detectadas). Grava os alertas em CSV.

Uso:
    python prototipo/simular_fluxo.py B --dias 7
    python prototipo/simular_fluxo.py A --dias 3 --mostrar 5

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import argparse
import sys
import time

from detector import DetectorFraude, periodo_de_teste, c


def main():
    ap = argparse.ArgumentParser(description="Simula o detector de fraudes no período de teste")
    ap.add_argument("base", choices=["A", "B"])
    ap.add_argument("--dias", type=int, default=7)
    ap.add_argument("--mostrar", type=int, default=3, help="alertas exibidos por dia")
    args = ap.parse_args()

    det = DetectorFraude(args.base)
    print(det)
    fluxo = periodo_de_teste(args.base)
    fluxo["dia"] = fluxo["ts"].dt.date
    alertas, tot = [], {"transacoes": 0, "alertas": 0, "detectadas": 0, "falsos": 0, "perdidas": 0}
    for dia in sorted(fluxo["dia"].unique())[: args.dias]:
        lote = fluxo[fluxo["dia"] == dia]
        t0 = time.perf_counter()
        r = det.avaliar(lote)
        ms = 1000 * (time.perf_counter() - t0) / len(lote)
        y = lote["is_fraud"].to_numpy() == 1
        a = r["alerta"].to_numpy()
        d = {"transacoes": len(lote), "alertas": int(a.sum()), "detectadas": int((a & y).sum()),
             "falsos": int((a & ~y).sum()), "perdidas": int((~a & y).sum())}
        for k in tot:
            tot[k] += d[k]
        print(f"\n{dia}: {d['transacoes']:,} transações | {d['alertas']} alertas "
              f"({d['detectadas']} fraudes, {d['falsos']} falsos alarmes) | {d['perdidas']} fraudes não detectadas "
              f"| {ms:.3f} ms por transação")
        top = r[r["alerta"]].sort_values("escore", ascending=False).head(args.mostrar)
        for i, linha in top.iterrows():
            real = "FRAUDE" if lote.loc[i, "is_fraud"] == 1 else "legítima"
            print(f"  [{lote.loc[i, 'ts']:%H:%M}] valor US$ {lote.loc[i, 'valor']:,.2f} | risco {linha['escore']:.3f} "
                  f"| real: {real}\n      motivos: {linha['motivos']}")
        alertas.append(lote[r["alerta"].to_numpy()].assign(escore=r.loc[r["alerta"], "escore"],
                                                             motivos=r.loc[r["alerta"], "motivos"]))
    print(f"\nTotal: {tot}")
    import pandas as pd
    saida = c.RESULTADOS / f"prototipo_alertas_{args.base}.csv"
    pd.concat(alertas)[["ts", "valor", "escore", "is_fraud", "motivos"]].to_csv(saida, index=False)
    print(f"Alertas gravados em {saida}")


if __name__ == "__main__":
    sys.exit(main())
