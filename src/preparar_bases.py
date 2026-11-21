"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/preparar_bases.py
Síntese:
    Preparação das duas bases públicas (IBM e Sparkov) e engenharia de atributos.
    Gera um parquet por base com os atributos usados pelos modelos, o rótulo
    (is_fraud) e as colunas de divisão temporal (split / split_hist).
    Todos os atributos comportamentais usam apenas transações ANTERIORES à
    transação corrente, evitando vazamento de informação do futuro.

Uso:
    python src/preparar_bases.py <pasta_ibm> <csv_sparkov> <pasta_saida>

    <pasta_ibm>   pasta com credit_card_transactions-ibm_v2.csv, sd254_users.csv e sd254_cards.csv
    <csv_sparkov> arquivo credit_card_frauds.csv (Base B)
    <pasta_saida> onde serão gravados base_a_ibm.parquet e base_b_sparkov.parquet

Histórico de alterações:
    2026-10-28 - Milton Almeida Leoncio - Versão inicial: limpeza, atributos e divisão temporal (N1).
    2026-11-20 - Milton Almeida Leoncio - Caminhos das bases por argumento; cabeçalho do projeto da disciplina;
                                          leitura robusta de sd254_cards.csv regravado pelo Excel.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc

DIA = 86_400  # segundos


def janela_24h(grupo, ts_seg, valor):
    """Para cada transação, conta e soma as transações do mesmo grupo nas 24h anteriores.

    Espera dados ordenados por (grupo, ts). Usa uma chave grupo*offset + ts para
    resolver a janela com uma única busca binária vetorizada.
    """
    chave = grupo.astype(np.int64) * 10**10 + ts_seg.astype(np.int64)
    ini = np.searchsorted(chave, chave - DIA, side="left")
    idx = np.arange(len(chave))
    acum = np.concatenate([[0.0], np.cumsum(valor, dtype=np.float64)])
    n = idx - ini  # transações anteriores na janela
    soma = acum[idx] - acum[ini]
    return n.astype(np.int32), soma.astype(np.float32)


def atributos_sequenciais(df, grupo_col, extras_novidade, fracoes_hist=()):
    """Atributos de histórico por cliente/cartão. `df` precisa estar ordenado por (grupo, ts).

    O número acumulado de transações só serve de denominador: como cresce com o
    tempo, usá-lo como atributo distorceria a divisão temporal.
    """
    ts_seg = df["ts"].astype("datetime64[s]").astype(np.int64)
    df["_ts_seg"] = ts_seg
    g = df.groupby(grupo_col, sort=False)
    df["seg_desde_ultima"] = g["_ts_seg"].diff().fillna(-1).astype(np.float32)
    df = df.drop(columns="_ts_seg")
    n_prev = g.cumcount()
    soma_prev = g["valor"].cumsum() - df["valor"]
    media_prev = (soma_prev / n_prev.replace(0, np.nan)).astype(np.float32)
    df["valor_sobre_media_hist"] = (df["valor"] / media_prev.abs().clip(lower=1)).fillna(-1).astype(np.float32)
    for col in fracoes_hist:  # fração das transações anteriores com a característica (ex.: online)
        df[f"frac_{col}_hist"] = ((g[col].cumsum() - df[col]) / n_prev.replace(0, np.nan)).fillna(-1).astype(np.float32)
    df["n_trans_24h"], df["valor_24h"] = janela_24h(df[grupo_col].to_numpy(), ts_seg.to_numpy(), df["valor"].to_numpy())
    for col in extras_novidade:
        df[f"novo_{col}"] = (df.groupby([grupo_col, col], sort=False).cumcount() == 0).astype(np.int8)
    return df


# ---------------------------------------------------------------- Base A (IBM)
def data_mes_ano(s):
    """Data de abertura da conta: "09/2002" (original) ou "Sep-02" (arquivo regravado pelo Excel)."""
    d = pd.to_datetime(s, format="%m/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%b-%y", errors="coerce"))


def preparar_ibm(pasta):
    cols = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name",
            "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]
    df = pc.read_csv(pasta / "credit_card_transactions-ibm_v2.csv",
                     convert_options=pc.ConvertOptions(include_columns=cols, column_types={"Time": pa.string()})).to_pandas()
    users = pd.read_csv(pasta / "sd254_users.csv")
    cards = pd.read_csv(pasta / "sd254_cards.csv")
    print(f"IBM: {len(df):,} transações, {len(users):,} usuários, {len(cards):,} cartões", flush=True)

    out = pd.DataFrame(index=df.index)
    hora = df["Time"].str.slice(0, 2).astype(np.int16)
    minuto = df["Time"].str.slice(3, 5).astype(np.int16)
    out["ts"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=hora, minute=minuto))
    out["cartao"] = (df["User"] * 10 + df["Card"]).astype(np.int32)
    out["usuario"] = df["User"].astype(np.int16)
    out["valor"] = df["Amount"].str.replace("$", "", regex=False).astype(np.float32)
    out["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(np.int8)
    out["ano"] = df["Year"].astype(np.int16)

    # atributos da transação
    out["hora"] = hora.astype(np.int8)
    out["dia_semana"] = out["ts"].dt.day_of_week.astype(np.int8)
    out["estorno"] = (out["valor"] < 0).astype(np.int8)
    out["tipo_uso"] = df["Use Chip"].map({"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}).astype(np.int8)
    out["mcc"] = df["MCC"].astype(np.int16)
    estado = df["Merchant State"]
    out["online"] = (df["Merchant City"] == "ONLINE").astype(np.int8)
    out["exterior"] = (estado.notna() & (estado.str.len() > 2)).astype(np.int8)
    erros = df["Errors?"].fillna("")
    for nome, chave in [("erro_saldo", "Insufficient Balance"), ("erro_pin", "Bad PIN"),
                        ("erro_tecnico", "Technical Glitch"), ("erro_num_cartao", "Bad Card Number"),
                        ("erro_cvv", "Bad CVV"), ("erro_validade", "Bad Expiration"), ("erro_cep", "Bad Zipcode")]:
        out[nome] = erros.str.contains(chave, regex=False).astype(np.int8)

    # perfil do usuário (sd254_users) - nome e endereço completo não são usados
    users = users.reset_index(names="usuario")
    # valores monetários: aceita "$24295" (original) e "$24,295 " (arquivo regravado pelo Excel)
    money = lambda s: s.astype(str).str.replace(r"[$,\s]", "", regex=True).astype(np.float32)
    u = pd.DataFrame({
        "usuario": users["usuario"].astype(np.int16),
        "ano_nasc": users["Birth Year"], "genero": (users["Gender"] == "Female").astype(np.int8),
        "renda_anual": money(users["Yearly Income - Person"]), "renda_per_capita_cep": money(users["Per Capita Income - Zipcode"]),
        "divida_total": money(users["Total Debt"]), "fico": users["FICO Score"].astype(np.int16),
        "n_cartoes_usuario": users["Num Credit Cards"].astype(np.int8),
        "u_estado": users["State"], "u_cidade": users["City"], "u_cep": users["Zipcode"],
    })
    out = out.merge(u, on="usuario", how="left")
    out["idade"] = (out["ano"] - out.pop("ano_nasc")).astype(np.int16)
    out["estado_diferente"] = ((estado.to_numpy() != out["u_estado"].to_numpy()) & (out["online"] == 0)).astype(np.int8)
    out["cidade_diferente"] = ((df["Merchant City"].to_numpy() != out["u_cidade"].to_numpy()) & (out["online"] == 0)).astype(np.int8)
    out["cep_diferente"] = ((df["Zip"].to_numpy() != out["u_cep"].to_numpy()) & (out["online"] == 0)).astype(np.int8)
    out = out.drop(columns=["u_estado", "u_cidade", "u_cep"])

    # perfil do cartão (sd254_cards) - número, CVV e validade não são usados
    c = pd.DataFrame({
        "cartao": (cards["User"] * 10 + cards["CARD INDEX"]).astype(np.int32),
        "bandeira": cards["Card Brand"].astype("category").cat.codes.astype(np.int8),
        "tipo_cartao": cards["Card Type"].astype("category").cat.codes.astype(np.int8),
        "cartao_com_chip": (cards["Has Chip"] == "YES").astype(np.int8),
        "cartoes_emitidos": cards["Cards Issued"].astype(np.int8),
        "limite_credito": money(cards["Credit Limit"]),
        "abertura": data_mes_ano(cards["Acct Open Date"]),
        "ano_troca_pin": cards["Year PIN last Changed"].astype(np.int16),
    })
    out = out.merge(c, on="cartao", how="left")
    out["idade_conta_anos"] = ((out["ts"] - out.pop("abertura")).dt.days / 365.25).astype(np.float32)
    out["anos_desde_troca_pin"] = (out["ano"] - out.pop("ano_troca_pin")).astype(np.int16)
    out["valor_sobre_limite"] = (out["valor"] / out["limite_credito"].clip(lower=1)).astype(np.float32)

    # histórico do cartão
    out["comerciante"] = df["Merchant Name"].to_numpy()
    out["mcc_hist"] = out["mcc"]
    out["estado_hist"] = estado.fillna("ONLINE").to_numpy()
    out["cidade_hist"] = df["Merchant City"].to_numpy()
    del df
    out = out.sort_values(["cartao", "ts"], kind="stable").reset_index(drop=True)
    out = atributos_sequenciais(out, "cartao", ["comerciante", "mcc_hist", "estado_hist", "cidade_hist"],
                                fracoes_hist=["online", "exterior"])
    out = out.drop(columns=["comerciante", "mcc_hist", "estado_hist", "cidade_hist"])

    # Divisão temporal. O teste é sempre 2019 (2020 só tem jan-fev e nenhuma fraude) e a validação out-dez/2018.
    # split: janela recente de treino (2017 a set/2018);
    # split_hist: todo o histórico até 2016, para medir o efeito da mudança no padrão das fraudes.
    teste = out["ano"] == 2019
    validacao = (out["ts"] >= "2018-10-01") & (out["ts"] < "2019-01-01")
    recente = (out["ts"] >= "2017-01-01") & (out["ts"] < "2018-10-01")
    out["split"] = np.select([teste, validacao, recente], ["teste", "validacao", "treino"], "fora")
    out["split_hist"] = np.select([teste, validacao, out["ano"] <= 2016], ["teste", "validacao", "treino"], "fora")
    out = out[(out["split"] != "fora") | (out["split_hist"] != "fora")]
    return out


# ------------------------------------------------------------ Base B (Sparkov)
def distancia_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return (6371 * 2 * np.arcsin(np.sqrt(a))).astype(np.float32)


def preparar_sparkov(arquivo):
    df = pd.read_csv(arquivo)
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    # a base não traz identificador do cartão; o cliente é identificado pela combinação abaixo
    df["cliente"] = df.groupby(["dob", "job", "city", "lat", "long"], sort=False).ngroup().astype(np.int32)
    print(f"Sparkov: {len(df):,} transações, {df['cliente'].nunique()} clientes, {df['merchant'].nunique()} comerciantes", flush=True)

    out = pd.DataFrame({
        "ts": df["ts"], "cliente": df["cliente"], "valor": df["amt"].astype(np.float32),
        "is_fraud": df["is_fraud"].astype(np.int8),
        "hora": df["ts"].dt.hour.astype(np.int8), "dia_semana": df["ts"].dt.day_of_week.astype(np.int8),
        "categoria": df["category"].astype("category").cat.codes.astype(np.int8),
        "estado": df["state"].astype("category").cat.codes.astype(np.int8),
        "idade": ((df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25).astype(np.float32),
        "log_pop_cidade": np.log1p(df["city_pop"]).astype(np.float32),
        "dist_comerciante_km": distancia_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]),
        "comerciante": df["merchant"], "categoria_hist": df["category"],
    })
    out = out.sort_values(["cliente", "ts"], kind="stable").reset_index(drop=True)
    out = atributos_sequenciais(out, "cliente", ["comerciante", "categoria_hist"])
    out = out.drop(columns=["comerciante", "categoria_hist"])

    # divisão temporal: treino jan/2019-jun/2020, validação jul-set/2020, teste out-dez/2020
    out["split"] = np.select([out["ts"] < "2020-07-01", out["ts"] < "2020-10-01"], ["treino", "validacao"], "teste")
    return out


if __name__ == "__main__":
    pasta_ibm, csv_sparkov, saida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    saida.mkdir(parents=True, exist_ok=True)
    preparar_sparkov(csv_sparkov).to_parquet(saida / "base_b_sparkov.parquet", index=False)
    print("Base B gravada", flush=True)
    preparar_ibm(pasta_ibm).to_parquet(saida / "base_a_ibm.parquet", index=False)
    print("ok", flush=True)
