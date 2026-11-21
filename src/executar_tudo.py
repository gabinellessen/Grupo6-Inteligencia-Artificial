"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/executar_tudo.py
Síntese:
    Executa todos os experimentos na ordem correta e gera as figuras do artigo.
    Pressupõe que src/preparar_bases.py já gerou os parquets na pasta indicada pela
    variável FRAUDES_ATRIBUTOS (ou ./atributos). Etapas já concluídas são lidas do
    disco; use --refazer para recalcular tudo e --a-partir-de N para retomar.

Uso:
    python src/executar_tudo.py [--refazer] [--a-partir-de N]

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import sys
import time

import eda
import exp1_modelos
import exp2_balanceamento
import exp3_otimizacao
import exp4_drift
import exp5_etica_xai
import figura_arquitetura

ETAPAS = [
    ("Análise exploratória", lambda r: (eda.resumo(), eda.qualidade(), eda.concentracao_fraudes(), eda.figuras())),
    ("Exp. 1 - Base B", lambda r: exp1_modelos.executar("B", r)),
    ("Exp. 1 - Base A", lambda r: exp1_modelos.executar("A", r)),
    ("Exp. 2 - Base B", lambda r: exp2_balanceamento.executar("B", r)),
    ("Exp. 2 - Base A", lambda r: exp2_balanceamento.executar("A", r)),
    ("Exp. 3 - Base B", lambda r: exp3_otimizacao.executar("B", r)),
    ("Exp. 3 - Base A", lambda r: exp3_otimizacao.executar("A", r)),
    ("Exp. 4 - Base A", lambda r: exp4_drift.executar(r)),
    ("Exp. 5 - Base B", lambda r: exp5_etica_xai.executar("B", r)),
    ("Exp. 5 - Base A", lambda r: exp5_etica_xai.executar("A", r)),
    ("Figuras", lambda r: (figura_arquitetura.desenhar(), exp1_modelos.figuras(), exp2_balanceamento.figura(),
                           exp4_drift.figura(), exp5_etica_xai.figuras())),
]

if __name__ == "__main__":
    refazer = "--refazer" in sys.argv
    inicio = int(sys.argv[sys.argv.index("--a-partir-de") + 1]) if "--a-partir-de" in sys.argv else 0
    for i, (nome, f) in enumerate(ETAPAS):
        if i < inicio:
            continue
        t0 = time.time()
        print(f"\n===== [{i}] {nome} =====", flush=True)
        f(refazer)
        print(f"===== [{i}] {nome}: {time.time() - t0:.0f}s =====", flush=True)
