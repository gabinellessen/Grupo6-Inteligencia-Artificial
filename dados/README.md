# Dados utilizados

O projeto usa duas bases **públicas e sintéticas** de transações com cartão de crédito, rotuladas como legítimas ou fraudulentas. Bases reais quase nunca são divulgadas, por causa do sigilo bancário e da LGPD. Por isso, usamos dados gerados por simuladores que reproduzem o comportamento de clientes e de fraudadores. Nenhuma das bases contém dados de pessoas reais.

| | Base A – IBM | Base B – Sparkov |
|---|---|---|
| Origem | IBM TabFormer (Altman, 2019) | simulador Sparkov (Harris, 2020) |
| Período | 1991 a fev./2020 | jan./2019 a dez./2020 |
| Transações | 24.386.900 | 339.607 |
| Fraudes | 29.757 (0,12%) | 1.782 (0,52%) |
| Titulares | 2.000 usuários, 6.146 cartões | 187 clientes, 13 estados do oeste dos EUA |
| Atributos gerados | 43 (29 no conjunto final) | 14 |

## Base A – IBM (não incluída no repositório: 2,3 GB)

- Download: <https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions> (requer conta no Kaggle).
- Arquivos usados: `credit_card_transactions-ibm_v2.csv` (transações), `sd254_users.csv` (perfil dos usuários) e `sd254_cards.csv` (cartões).
- Colunas das transações: usuário, cartão, ano, mês, dia, hora, valor, tipo de uso (tarja, chip ou online), comerciante, cidade, estado, CEP, MCC, erros de autorização e rótulo `Is Fraud?`.
- **Não usamos:** número do cartão, CVV, validade, nome e endereço do usuário. Eles são removidos na preparação.
- Observação: se o `sd254_cards.csv` tiver sido aberto e salvo no Excel, o limite aparece como `"$24,295 "` e a data de abertura como `Sep-02`. O script `src/preparar_bases.py` aceita os dois formatos.

## Base B – Sparkov (incluída: `credit_card_frauds.zip`)

- Arquivo: `credit_card_frauds.csv`, compactado em `dados/credit_card_frauds.zip` (61 MB descompactado).
- Gerado pelo simulador Sparkov (<https://github.com/namebrandon/Sparkov_Data_Generation>).
- Colunas: data e hora, comerciante, categoria (14), valor, cidade, estado, latitude e longitude do cliente, população da cidade, profissão, data de nascimento, identificador da transação, latitude e longitude do comerciante e `is_fraud`.
- Não há identificador de cartão. O cliente é identificado pela combinação de data de nascimento, profissão, cidade e coordenadas.

## Como preparar

Para usar os scripts sem alterar caminhos, coloque os arquivos assim:

```
dados/ibm/credit_card_transactions-ibm_v2.csv
dados/ibm/sd254_users.csv
dados/ibm/sd254_cards.csv
dados/credit_card_frauds.csv          (descompactado do .zip)
```

Depois, rode:

```bash
python src/preparar_bases.py dados/ibm dados/credit_card_frauds.csv atributos
```

O script gera `atributos/base_a_ibm.parquet` e `atributos/base_b_sparkov.parquet`. Na máquina do grupo (i7-14700F, 32 GB de RAM), levou cerca de 4 minutos. Por causa dos 24 milhões de linhas da Base A, recomendamos pelo menos 16 GB de RAM.

## Divisão temporal

| | Base A – IBM | Base B – Sparkov |
|---|---|---|
| Treino | jan./2017 a set./2018 (3.011.396; 2.036 fraudes) | jan./2019 a jun./2020 (243.168; 1.408 fraudes) |
| Validação | out. a dez./2018 (433.579; 710 fraudes) | jul. a set./2020 (44.882; 203 fraudes) |
| Teste | jan. a dez./2019 (1.723.938; 2.087 fraudes) | out. a dez./2020 (51.557; 171 fraudes) |

Na Base A, os dois meses de 2020 não têm fraudes rotuladas e foram descartados. O histórico até 2016 é usado no Experimento 4, para medir o efeito da mudança no padrão das fraudes.
