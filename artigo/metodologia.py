"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: artigo/metodologia.py
Síntese:
    Seção 3 do artigo (Metodologia e Resultados Esperados). Os números das bases
    vêm dos CSVs da análise exploratória, para o texto ficar sempre coerente com
    os resultados do repositório.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import pandas as pd

from docx_sbc import br, milhar


def secao(art, RES):
    resumo = pd.read_csv(RES / "eda_resumo_bases.csv", index_col=0)
    qual = pd.read_csv(RES / "eda_qualidade.csv").set_index("base")
    conc = pd.read_csv(RES / "eda_concentracao_fraudes.csv")
    canal = pd.read_csv(RES / "eda_canal_fraudes_ano_A.csv", index_col=0)
    A, B = "Base A - IBM", "Base B - Sparkov"
    ca = conc[conc["base"] == A].set_index("periodo")
    cb = conc[conc["base"] == B].set_index("periodo")
    f_arq = art.proxima_figura()

    art.h1("3. Metodologia e Resultados Esperados")
    art.h2("3.1. Visão geral da solução")
    art.par("A pesquisa é experimental e quantitativa. A Figura "
            f"{f_arq} resume a arquitetura da solução em oito etapas: obtenção dos dados, preparação, engenharia "
            "de atributos, divisão temporal, treinamento dos modelos, ajuste na validação, avaliação no teste e "
            "protótipo.")
    art.par("A implementação foi feita em Python. Os dados foram tratados com pandas e PyArrow. Os modelos usam "
            "scikit-learn [Pedregosa et al. 2011], XGBoost, LightGBM e imbalanced-learn. A otimização usa Optuna, e "
            "o painel do protótipo usa Streamlit. Cada etapa tem um script em *src/* e um notebook executado no "
            "repositório do projeto (Seção 4.3), todos com cabeçalho de identificação e histórico de alterações.")
    art.figura(RES / "fig_arquitetura.png", "Arquitetura da solução: do dado bruto ao protótipo usado pelo "
               "analista antifraude.", 15.0)

    art.h2("3.2. Bases de dados e análise exploratória")
    art.par("Bases reais de transações raramente são divulgadas, por causa do sigilo bancário e da LGPD. Por isso, "
            "foram usadas duas bases públicas e sintéticas, geradas por simuladores que imitam o comportamento de "
            f"clientes e fraudadores (Tabela {art.proxima_tabela()}).")
    art.par("A Base A [Altman 2019] tem 24,4 milhões de transações de 2.000 usuários e 6.146 cartões, entre 1991 e "
            "fevereiro de 2020, com 0,12% de fraudes. Ela vem acompanhada de tabelas com o perfil dos usuários "
            "(idade, gênero, renda, dívida e score de crédito) e dos cartões (bandeira, tipo, limite e data de "
            "abertura da conta).")
    art.par("A Base B, gerada pelo simulador Sparkov [Harris 2020], tem 339.607 transações de 187 clientes de 13 "
            "estados do oeste dos Estados Unidos, entre 2019 e 2020, com 0,52% de fraudes. Como ela não traz "
            "identificador de cartão, cada cliente foi identificado pela combinação de data de nascimento, "
            "profissão e endereço.")

    def periodo(base, s):
        return (f"{resumo.loc[f'{s}: período', base].replace(' a ', ' a ')}; "
                f"{milhar(resumo.loc[f'{s}: transações', base])} ({milhar(resumo.loc[f'{s}: fraudes', base])} fraudes)")
    t_bases = art.tabela(
        ["Característica", "Base A – IBM", "Base B – Sparkov"],
        [["Origem", "sintética (IBM TabFormer)", "sintética (simulador Sparkov)"],
         ["Período", "1991 a fev./2020", "jan./2019 a dez./2020"],
         ["Transações", "24.386.900", "339.607"],
         ["Fraudes", "29.757 (0,12%)", "1.782 (0,52%)"],
         ["Titulares", "2.000 usuários; 6.146 cartões", "187 clientes"],
         ["Atributos (completo / final)", "43 / 29", "14 / 14"],
         ["Treino", periodo(A, "treino"), periodo(B, "treino")],
         ["Validação", periodo(A, "validacao"), periodo(B, "validacao")],
         ["Teste", periodo(A, "teste"), periodo(B, "teste")]],
        "Características das bases e divisão temporal.", larguras_cm=[3.6, 5.6, 5.6], fonte=9)

    pct = lambda ano, col: br(100 * canal.loc[ano, col], 0)
    f_hora = art.proxima_figura()
    art.par(f"A análise exploratória não encontrou valores ausentes nem linhas duplicadas nas transações usadas. A "
            f"Base A tem {milhar(qual.loc[A, 'valores negativos (estornos)'])} estornos (valores negativos), "
            "mantidos com um indicador próprio.")
    art.par(f"As duas bases têm padrões de fraude distintos (Figura {f_hora}). Na Base B, as fraudes se concentram "
            "entre 22h e 3h e nas categorias de compras online e de supermercado presencial. Na Base A, a taxa é "
            "baixa e quase uniforme ao longo do dia.")
    art.par(f"Já o canal das fraudes da Base A muda com o tempo (Figura {f_hora + 1}). De 2012 a 2016, entre "
            f"{pct(2012, 'online')}% e {pct(2016, 'online')}% das fraudes foram compras online. Em 2017 e em 2019, "
            f"todas foram compras no exterior; em 2018, {pct(2018, 'exterior')}%. Essa mudança de conceito orientou "
            "a divisão temporal e o Experimento 4.")
    art.figura(RES / "fig_eda_fraude_por_hora.png", "Proporção de fraudes por hora do dia.", 14.0)
    art.figura(RES / "fig_eda_canal_fraudes_A.png", "Canal das fraudes por ano na Base A e cortes da divisão "
               "temporal: até 2016 predominam as compras online; a partir de 2017, as compras no exterior.", 14.0)
    art.par("As fraudes também se concentram em poucos titulares e em surtos curtos. No treino da Base A, as "
            f"{milhar(ca.loc['treino', 'fraudes'])} fraudes ocorreram em {milhar(ca.loc['treino', 'titulares_com_fraude'])} "
            f"dos {milhar(ca.loc['treino', 'titulares_ativos'])} cartões ativos, com mediana de "
            f"{br(ca.loc['treino', 'fraudes_por_titular_mediana'], 0)} fraudes por cartão em cerca de "
            f"{br(ca.loc['treino', 'duracao_surto_dias_mediana'], 0)} dias.")
    art.par(f"Dos {milhar(ca.loc['teste', 'titulares_com_fraude'])} cartões fraudados no teste, apenas "
            f"{milhar(ca.loc['teste', 'titulares_com_fraude_tambem_no_treino'])} já tinham sido fraudados no treino. "
            f"Na Base B, nenhum dos {milhar(cb.loc['teste', 'titulares_com_fraude'])} clientes fraudados no teste "
            "tinha sofrido fraude no treino. Isso significa que o modelo precisa reconhecer o comportamento da "
            "fraude, e não quem já foi vítima.")

    art.h2("3.3. Preparação e engenharia de atributos")
    art.par("A preparação converteu tipos (valores, datas e horas), codificou as variáveis categóricas e removeu "
            "campos sensíveis ou sem valor preditivo: número do cartão, CVV, validade, nome e endereço do titular e "
            "identificador da transação. Foram criados quatro grupos de atributos:")
    art.lista([
        "**temporais:** hora e dia da semana;",
        "**da transação e geográficos:** valor, tipo de uso (tarja, chip ou online), MCC ou categoria do comércio, "
        "compra online, compra no exterior, estado, cidade ou CEP diferentes do endereço do titular e erros de "
        "autorização (Base A); distância entre cliente e comerciante e população da cidade (Base B);",
        "**de perfil (estáticos):** idade, gênero, renda, dívida, score FICO, limite, bandeira, tipo e idade da "
        "conta (Base A); idade, estado e população da cidade do cliente (Base B);",
        "**comportamentais, por cartão (A) ou cliente (B):** tempo desde a transação anterior, razão entre o valor "
        "e a média histórica, número e soma das transações nas 24 horas anteriores, frações históricas de compras "
        "online e no exterior e indicadores de primeira compra em um comerciante, categoria, cidade ou estado.",
    ])
    art.par("Os atributos comportamentais usam apenas transações anteriores à avaliada. Assim, são os mesmos que um "
            "sistema em produção calcularia no momento da compra; o notebook 02 verifica essa propriedade. Ao todo, "
            "foram 43 atributos na Base A e 14 na Base B.")
    art.par("Como as fraudes do teste ocorrem em outros titulares, os atributos de perfil estático podem funcionar "
            "como identificadores: o modelo aprenderia quais cartões tiveram fraude no treino, informação que não "
            "se repete no futuro. Esse é um caso de vazamento por identidade [Kaufman et al. 2012]. Por isso, o "
            "Experimento 1 treina todos os modelos da Base A com e sem esses 14 atributos. O conjunto final de cada "
            "base é o que contém a combinação modelo–atributos de maior AUPRC na validação.")

    art.h2("3.4. Divisão temporal e protocolo de avaliação")
    art.par(f"As bases foram divididas de forma temporal (Tabela {t_bases}): o modelo é treinado com o passado e aplicado "
            "a transações futuras, como em um sistema antifraude real. Na Base A, o treino principal usa uma janela "
            "recente (janeiro de 2017 a setembro de 2018), por causa da mudança de conceito. Os dois meses de 2020 "
            "foram descartados por não terem fraudes rotuladas.")
    art.par("Todas as decisões usam só a validação: parada antecipada, escolha dos hiperparâmetros, do conjunto de "
            "atributos, do modelo final e do limiar de decisão (o que maximiza o F1-score). O teste é usado uma "
            "única vez, com a proporção real de fraudes.")
    art.par("Os intervalos de confiança de 95% (IC 95%) vêm de um bootstrap de Poisson sobre o teste [Efron and "
            "Tibshirani 1993], com 1.000 réplicas na Base B e 500 na Base A. As comparações entre dois modelos usam "
            "as mesmas réplicas (bootstrap pareado), e a diferença é considerada significativa quando o IC 95% da "
            "diferença não contém zero.")

    art.h2("3.5. Experimentos")
    art.par("Foram realizados cinco experimentos, descritos a seguir.")
    art.lista([
        "**E1 – Comparação de classificadores:** um sistema de regras estáticas e seis modelos supervisionados "
        "(Regressão Logística, Random Forest, XGBoost, LightGBM, MLP e SVM). O sistema de regras soma sinais de "
        "alerta usuais, dez na Base A e sete na Base B (valor acima do percentil 99, valor atípico para o cliente, "
        "muitas transações em 24h, madrugada, comerciante novo, compra no exterior etc.), com cortes calculados no "
        "treino; o limiar é o número de sinais que maximiza o F1 na validação. A Random Forest "
        "(200 árvores) e o XGBoost repetem a configuração da N1. O LightGBM usa a mesma regularização mínima do "
        "XGBoost. Os modelos de árvores usam ponderação de classes.",
        "**E1 (continuação):** a MLP tem duas camadas ocultas (64 e 32 neurônios) e foi treinada com subamostragem "
        "de uma fraude para 50 legítimas. A SVM com *kernel* RBF usou uma fraude para 20 legítimas e só a Base B, "
        "pois seu custo é proibitivo em milhões de transações.",
        "**E2 – Desbalanceamento:** Random Forest e XGBoost com cinco estratégias aplicadas só ao treino: sem "
        "tratamento, ponderação de classes, subamostragem aleatória (1:10), SMOTE (fraudes = 10% das legítimas) e "
        "SMOTE seguido de subamostragem (razão final 1:2).",
        "**E3 – Otimização:** Optuna (TPE) para XGBoost, LightGBM e MLP, maximizando a AUPRC na validação. Foram "
        "60, 60 e 40 tentativas na Base B e até 30, 40 e 30 na Base A, com limite de 20 minutos por família. O "
        "espaço de busca inclui profundidade ou número de folhas, taxa de aprendizado, regularização, "
        "subamostragem e o expoente do peso da classe positiva; na MLP, arquitetura, regularização e razão de "
        "subamostragem.",
        "**E4 – Mudança de conceito (Base A):** avaliação mês a mês de 2019 com o modelo final em três estratégias: "
        "treino só com o histórico até 2016 (10% das legítimas, por limitação de memória); treino único na janela "
        "recente; e retreino mensal com janela deslizante de 21 meses (18 de treino e 3 de validação). Supõe-se "
        "que os rótulos de um mês estejam disponíveis ao fim dele.",
        "**E5 – Ética e explicabilidade:** recall e taxa de falsos positivos (FPR) por gênero, faixa etária e "
        "quartil de renda (Base A) e por faixa etária (Base B), comparando o modelo final com uma variante de "
        "mesmos hiperparâmetros com ou sem os atributos sensíveis. Também foram calculadas a importância global e "
        "explicações locais: TreeSHAP para modelos de árvores; para os demais, oclusão (troca de cada atributo "
        "pela mediana das legítimas do treino).",
    ])

    art.h2("3.6. Protótipo")
    art.par("O protótipo empacota o modelo final de cada base em um componente *DetectorFraude*. Ele recebe as "
            "transações com os atributos já calculados e devolve o escore de risco e a decisão de alerta (pelo "
            "limiar da validação). Para cada alerta, devolve também os três atributos que mais aumentaram o risco, "
            "em linguagem simples.")
    art.par("Há duas interfaces. A primeira é uma simulação no terminal, em que as transações do período de teste "
            "chegam dia a dia. A segunda é um painel Streamlit para o analista, com indicadores do dia, fila de "
            "alertas, explicação de cada alerta e uma simulação \"e se\" (alterar valor e hora e recalcular o "
            "risco). A decisão de bloquear a transação fica sempre com o analista.")

    art.h2("3.7. Resultados esperados e evolução da proposta")
    art.par("Com base na literatura e nos resultados da N1, esperava-se que: (i) os modelos de *ensemble* de "
            "árvores superassem as regras estáticas e os modelos lineares; (ii) o SMOTE aumentasse o recall em "
            "relação à ponderação de classes; (iii) a otimização de hiperparâmetros melhorasse o F1-score dos "
            "modelos padrão; (iv) o retreinamento periódico reduzisse a perda de desempenho causada pela mudança de "
            "conceito; e (v) as explicações fossem coerentes com os padrões da análise exploratória.")
    art.par("Em relação à proposta e à N1, houve quatro mudanças. A comparação foi ampliada com regras, Regressão "
            "Logística, LightGBM e MLP. A SVM prevista ficou restrita à Base B, por custo computacional. O Apache "
            "Spark, citado na fundamentação da N1, não foi necessário: PyArrow e pandas processaram os 24 milhões "
            "de transações em uma única máquina. Por fim, a descoberta do vazamento por identidade levou à seleção "
            "do conjunto de atributos da Base A pela validação.")
