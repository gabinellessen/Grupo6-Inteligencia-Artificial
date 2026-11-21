"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: prototipo/detector.py
Síntese:
    Componente central do protótipo. Carrega o modelo final de uma base (treinado no
    Experimento 3), calcula o escore de risco de cada transação, aplica o limiar de
    decisão escolhido na validação e, para cada alerta, devolve os três motivos que
    mais aumentaram o risco (explicação local, ver src/explicacao.py). A decisão
    final sobre bloquear ou não a transação continua com um analista humano.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import comum as c  # noqa: E402
import explicacao as xai  # noqa: E402
from exp3_otimizacao import carregar_final  # noqa: E402


class DetectorFraude:
    def __init__(self, chave):
        self.chave = chave
        self.modelo, self.meta = carregar_final(chave)
        self.atributos = self.meta["atributos"]
        self.limiar = float(self.meta["limiar"])
        arq_ref = c.MODELOS_DIR / f"referencia_{chave}.json"
        self.referencia = pd.Series(c.ler_json(arq_ref)) if arq_ref.exists() else None

    def __repr__(self):
        return (f"DetectorFraude({self.meta['nome_base']}, modelo={self.meta['familia']}, "
                f"limiar={self.limiar:.3f}, {len(self.atributos)} atributos)")

    def avaliar(self, transacoes, explicar=True):
        """Recebe transações com os atributos já calculados e devolve escore, alerta e motivos."""
        X = transacoes[self.atributos]
        escore = c.pontuar(self.modelo, X)
        alerta = escore >= self.limiar
        saida = pd.DataFrame({"escore": escore, "alerta": alerta}, index=transacoes.index)
        saida["motivos"] = ""
        if explicar and alerta.any():
            Xa = X[alerta]
            contr = xai.contribuicoes(self.modelo, Xa, self.referencia)
            saida.loc[alerta, "motivos"] = [" | ".join(xai.motivos(contr.loc[i], Xa.loc[i])) for i in Xa.index]
        return saida


def periodo_de_teste(chave):
    """Transações do período de teste em ordem cronológica (simulam a chegada em tempo real).

    Os atributos comportamentais foram calculados apenas com transações anteriores a cada
    uma, então são os mesmos que um sistema em produção calcularia no momento da compra.
    """
    partes, _ = c.carregar(chave)
    return partes["teste"].sort_values("ts", kind="stable").reset_index(drop=True)
