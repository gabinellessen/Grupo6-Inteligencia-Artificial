"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/explicacao.py
Síntese:
    Inteligência Artificial Explicável (XAI). Calcula a contribuição de cada atributo
    para o escore de uma transação, na escala log-odds:
      - modelos de árvores (XGBoost, LightGBM): valores SHAP exatos (TreeSHAP),
        obtidos pelas próprias bibliotecas (pred_contribs / pred_contrib);
      - demais modelos (MLP, Regressão Logística): oclusão - cada atributo é trocado
        pelo valor de referência (mediana das transações legítimas do treino) e a
        contribuição é a variação do log-odds.
    Também gera o texto dos "motivos do alerta" usado pelo protótipo.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import numpy as np
import pandas as pd

import comum as c


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def metodo(modelo):
    return "TreeSHAP" if type(modelo).__name__ in ("XGBClassifier", "LGBMClassifier") else "oclusão"


def contribuicoes(modelo, X, referencia=None):
    """DataFrame (linhas de X x atributos) com a contribuição de cada atributo em log-odds."""
    nome = type(modelo).__name__
    if nome == "XGBClassifier":
        import xgboost as xgb
        contr = modelo.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        return pd.DataFrame(contr, columns=X.columns, index=X.index)
    if nome == "LGBMClassifier":
        contr = modelo.predict(X, pred_contrib=True)[:, :-1]
        return pd.DataFrame(contr, columns=X.columns, index=X.index)
    if referencia is None:
        raise ValueError("oclusão precisa dos valores de referência")
    base = _logit(c.pontuar(modelo, X))
    out = {}
    for col in X.columns:
        X2 = X.copy()
        X2[col] = referencia[col]
        out[col] = base - _logit(c.pontuar(modelo, X2))
    return pd.DataFrame(out, index=X.index)


def referencia_legitimas(treino, atributos):
    """Valores de referência para a oclusão: mediana das transações legítimas do treino."""
    return treino.loc[treino["is_fraud"] == 0, atributos].median()


def formatar_valor(nome, v):
    if nome.startswith(("novo_", "erro_")) or nome in ("online", "exterior", "estado_diferente", "cidade_diferente",
                                                       "cep_diferente", "estorno", "genero", "cartao_com_chip"):
        return "sim" if v >= 0.5 else "não"
    if nome == "seg_desde_ultima":
        return "primeira" if v < 0 else (f"{v / 3600:.1f} h".replace(".", ",") if v < 172800 else f"{v / 86400:.0f} dias")
    if nome in ("valor", "valor_24h", "limite_credito", "renda_anual"):
        return f"US$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if nome == "hora":
        return f"{int(v)}h"
    if nome == "idade":
        return f"{int(v)} anos"
    if nome == "categoria" and 0 <= int(v) < len(c.CATEGORIAS_SPARKOV):
        return c.CATEGORIAS_SPARKOV[int(v)]
    if float(v).is_integer():
        return f"{int(v)}"
    return c.br(v, 2)


def motivos(linha_contr, linha_x, k=3):
    """Os k atributos que mais AUMENTARAM o risco, em linguagem simples."""
    top = linha_contr.sort_values(ascending=False).head(k)
    return [f"{c.rotulo(n)} = {formatar_valor(n, linha_x[n])} (+{c.br(v, 2)})" for n, v in top.items() if v > 0]
