"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: prototipo/app.py
Síntese:
    Painel do protótipo (Streamlit) para o analista antifraude: escolhe a base e o dia
    do período de teste, vê os indicadores do dia, a fila de alertas ordenada por risco
    com os motivos de cada um, a explicação detalhada de um alerta e uma simulação
    "e se" (alterar valor e hora de uma transação e recalcular o risco).

Uso:
    streamlit run prototipo/app.py

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import pandas as pd
import streamlit as st

from detector import DetectorFraude, periodo_de_teste, c, xai

st.set_page_config(page_title="Detector de fraudes", layout="wide")


@st.cache_resource
def detector(chave):
    return DetectorFraude(chave)


@st.cache_data
def fluxo(chave):
    return periodo_de_teste(chave)


st.title("Detector de fraudes em transações com cartão")
st.caption("Protótipo do projeto de Inteligência Artificial - Universidade Presbiteriana Mackenzie. "
           "Dados sintéticos; a decisão final é sempre de um analista humano.")

with st.sidebar:
    # a base e o dia também podem vir da URL, ex.: ?base=B&dia=2020-10-15 (útil para demonstrações)
    qp = st.query_params
    bases = ["B", "A"]
    chave = st.radio("Base", bases, index=bases.index(qp.get("base", "B")) if qp.get("base") in bases else 0,
                     format_func=lambda k: c.BASES[k]["nome"])
    det = detector(chave)
    dados = fluxo(chave)
    dias = sorted(dados["ts"].dt.date.unique())
    inicial = next((d for d in dias if str(d) == qp.get("dia")), dias[0])
    dia = st.select_slider("Dia do período de teste", options=dias, value=inicial)
    st.markdown(f"**Modelo:** {det.meta['familia']}  \n**Limiar:** {det.limiar:.3f}  \n"
                f"**Atributos:** {len(det.atributos)}")

lote = dados[dados["ts"].dt.date == dia]
r = det.avaliar(lote)
y = lote["is_fraud"].to_numpy() == 1
a = r["alerta"].to_numpy()
col = st.columns(5)
col[0].metric("Transações", f"{len(lote):,}".replace(",", "."))
col[1].metric("Alertas", int(a.sum()))
col[2].metric("Fraudes detectadas", int((a & y).sum()))
col[3].metric("Falsos alarmes", int((a & ~y).sum()))
col[4].metric("Fraudes não detectadas", int((~a & y).sum()))

st.subheader("Fila de alertas (maior risco primeiro)")
fila = lote.loc[r["alerta"]].assign(risco=r.loc[r["alerta"], "escore"], motivos=r.loc[r["alerta"], "motivos"])
fila = fila.sort_values("risco", ascending=False)
tabela = pd.DataFrame({"hora": fila["ts"].dt.strftime("%H:%M"), "valor (US$)": fila["valor"].round(2),
                       "risco": fila["risco"].round(3), "rótulo real": fila["is_fraud"].map({1: "fraude", 0: "legítima"}),
                       "motivos do alerta": fila["motivos"]})
st.dataframe(tabela, use_container_width=True, hide_index=True)

if len(fila):
    st.subheader("Explicação de um alerta")
    escolha = st.selectbox("Alerta", fila.index,
                           format_func=lambda i: f"{fila.loc[i, 'ts']:%H:%M} - US$ {fila.loc[i, 'valor']:,.2f} - risco {fila.loc[i, 'risco']:.3f}")
    x = lote.loc[[escolha], det.atributos]
    contr = xai.contribuicoes(det.modelo, x, det.referencia).iloc[0]
    top = contr.reindex(contr.abs().sort_values(ascending=False).index).head(8)
    st.bar_chart(pd.DataFrame({"contribuição (log-odds)": top.values},
                              index=[f"{c.rotulo(n)} = {xai.formatar_valor(n, x.iloc[0][n])}" for n in top.index]),
                 horizontal=True)
    st.caption(f"Método: {xai.metodo(det.modelo)}. Barras positivas aumentam o risco; negativas diminuem.")

    st.subheader("Simulação \"e se\"")
    c1, c2 = st.columns(2)
    novo_valor = c1.number_input("Valor da compra (US$)", value=float(x.iloc[0]["valor"]), min_value=0.0, step=10.0)
    nova_hora = c2.slider("Hora", 0, 23, int(x.iloc[0]["hora"]))
    x2 = x.copy()
    x2["valor"], x2["hora"] = novo_valor, nova_hora
    if "valor_sobre_media_hist" in x2 and x.iloc[0]["valor"] > 0 and x.iloc[0]["valor_sobre_media_hist"] > 0:
        x2["valor_sobre_media_hist"] = x.iloc[0]["valor_sobre_media_hist"] * novo_valor / x.iloc[0]["valor"]
    if "valor_sobre_limite" in x2 and x.iloc[0]["valor"] > 0:
        x2["valor_sobre_limite"] = x.iloc[0]["valor_sobre_limite"] * novo_valor / x.iloc[0]["valor"]
    novo = float(c.pontuar(det.modelo, x2)[0])
    st.metric("Risco recalculado", f"{novo:.3f}", delta=f"{novo - fila.loc[escolha, 'risco']:+.3f}", delta_color="inverse")
    st.write("Alerta" if novo >= det.limiar else "Sem alerta")
