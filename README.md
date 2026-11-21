# Detecção de fraudes em transações financeiras com aprendizado supervisionado

Projeto da disciplina **Inteligência Artificial** – 7º semestre de Ciência da Computação
Universidade Presbiteriana Mackenzie – Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

| Integrante | RA | E-mail |
|---|---|---|
| Cheuk Ki Yu | 10419664 | cheuk.yu@mackenzista.com.br |
| Gabriela Nellessen de Sousa | 10441930 | gabrielanellessen@gmail.com |
| Milton Almeida Leoncio | 10416764 | milton.almeida@live.com |

- **Repositório:** <https://github.com/gabinellessen/Grupo6-Inteligencia-Artificial>
- **Artigo final (template SBC):** [`artigo/Artigo_Final_IA_Fraudes.pdf`](artigo/Artigo_Final_IA_Fraudes.pdf)
- **Vídeo de apresentação:** [https://youtu.be/Ve9pj3yQw5M]
- **ODS:** 16 (meta 16.4 – fluxos financeiros ilícitos), 8 (meta 8.10 – instituições financeiras) e 9 (inovação)

## O problema

Fraudes com cartão de crédito são raras (menos de 1% das transações), mudam de padrão ao longo do tempo e têm custos assimétricos: uma fraude não detectada gera prejuízo, e um alarme falso bloqueia um cliente legítimo. O projeto treina e avalia modelos supervisionados que estimam o risco de cada transação. A avaliação respeita a ordem do tempo (o modelo aprende com o passado e é testado no futuro) e inclui uma análise de vieses, explicações para cada alerta e um protótipo de painel para o analista antifraude.

## Resultados principais

Desempenho do modelo final (LightGBM otimizado, escolhido pela AUPRC na validação), medido no período de teste, que mantém a proporção real de fraudes:

| Base | Teste | Precisão | Recall | F1 | AUPRC |
|---|---|---|---|---|---|
| B – Sparkov | out.–dez./2020 (51.557 transações, 171 fraudes) | 0,871 | 0,830 | **0,850** | 0,912 |
| A – IBM | 2019 (1.723.938 transações, 2.087 fraudes) | 0,712 | 0,671 | **0,691** | 0,759 |
| A – IBM, com retreino mensal | 2019 | 0,725 | 0,730 | **0,728** | 0,784 |

Para comparação, os resultados preliminares (N1) na Base A tinham F1 de 0,548 (Random Forest), e um sistema de regras estáticas fica em 0,11 (A) e 0,39 (B).

Principais conclusões:

- **Vazamento por identidade:** na Base A, atributos fixos do titular (renda, dívida, FICO, limite etc.) permitiam ao modelo "decorar" quais cartões tiveram fraude no treino. Só 11 dos 244 cartões fraudados no teste tinham sido fraudados antes. Removê-los elevou o F1 do XGBoost de 0,51 para 0,64.
- **Desbalanceamento:** sob mudança de conceito, pesos de classe muito altos (cerca de 1.500:1) atrapalham a generalização. O SMOTE melhorou a precisão, mas não o recall.
- **Mudança de conceito:** treinado só com o padrão antigo (fraudes online até 2016), o modelo cai para F1 de 0,19 em 2019. O retreino mensal supera o modelo estático na maioria dos meses.
- **Ética:** sem atributos sensíveis, o modelo da Base A é mais preciso (F1 0,69 contra 0,61) e reduz as diferenças de recall entre grupos de renda (0,38 → 0,17) e de idade (0,14 → 0,06).
- **Explicabilidade:** cada alerta traz os atributos que mais pesaram (TreeSHAP), e a decisão final é do analista.

## Estrutura do repositório

```
artigo/        artigo final (.pdf e .docx), template SBC e scripts que geram o artigo a partir dos resultados
dados/         descrição das bases e a Base B (Sparkov) compactada; a Base A (IBM) é baixada do Kaggle
notebooks/     notebooks executados, um por etapa (análise exploratória → protótipo)
src/           código-fonte dos experimentos (cada arquivo com cabeçalho e histórico de alterações)
prototipo/     detector de fraudes, simulação em linha de comando e painel Streamlit
resultados/    tabelas (.csv), figuras (.png) e modelos finais (resultados/modelos)
```

| Arquivo | Conteúdo |
|---|---|
| `src/preparar_bases.py` | limpeza, remoção de dados sensíveis, engenharia de atributos e divisão temporal |
| `src/eda.py` | análise exploratória (resumo, qualidade, fraude por hora, canal das fraudes por ano) |
| `src/exp1_modelos.py` | Exp. 1: regras, Reg. Logística, Random Forest, XGBoost, LightGBM, MLP e SVM; efeito do perfil estático |
| `src/exp2_balanceamento.py` | Exp. 2: ponderação de classes, subamostragem e SMOTE |
| `src/exp3_otimizacao.py` | Exp. 3: Optuna (TPE) para XGBoost, LightGBM e MLP; escolha e gravação do modelo final |
| `src/exp4_drift.py` | Exp. 4: mudança de conceito e retreinamento mensal (Base A) |
| `src/exp5_etica_xai.py` | Exp. 5: auditoria de vieses por grupo, modelo sem atributos sensíveis e explicabilidade |
| `src/explicacao.py` | contribuições por atributo (TreeSHAP ou oclusão) e motivos de cada alerta |
| `src/comum.py` | carga das bases, métricas, limiar, bootstrap e estilo das figuras |
| `src/executar_tudo.py` | roda todas as etapas na ordem |

## Como reproduzir

Requisitos: Python 3.14 (testado no Windows 11) e pelo menos 16 GB de RAM por causa da Base A.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

1. Obtenha as bases conforme [`dados/README.md`](dados/README.md).
2. Gere os atributos (cerca de 4 minutos):
   ```bash
   python src/preparar_bases.py dados/ibm dados/credit_card_frauds.csv atributos
   ```
3. Rode todos os experimentos. Na máquina do grupo (i7-14700F, 32 GB), levou cerca de 2h45 no total, a maior parte na otimização da Base A:
   ```bash
   python src/executar_tudo.py
   ```
   As etapas já concluídas são lidas de `resultados/`. Para recalcular tudo, use `--refazer`; para retomar de uma etapa, use `--a-partir-de N`.
4. Os notebooks em `notebooks/` reproduzem cada etapa e mostram tabelas e figuras. Se os atributos estiverem em outra pasta, defina `FRAUDES_ATRIBUTOS`.
5. O artigo é gerado a partir dos resultados com `python artigo/gerar_artigo.py --pdf`. A exportação para PDF usa o Microsoft Word e o pacote `pywin32`, só no Windows.

Todas as sementes aleatórias são fixas (`SEMENTE = 42`). O limiar de decisão e a escolha do modelo usam só a validação; o teste é usado uma única vez.

## Protótipo

O protótipo usa o modelo final de cada base (`resultados/modelos/`). Ele calcula o risco das transações, gera alertas e explica cada alerta com os três atributos que mais aumentaram o risco. A decisão final continua com um analista humano.

```bash
python prototipo/simular_fluxo.py B --dias 7   # chegada das transações dia a dia, no terminal
streamlit run prototipo/app.py                 # painel do analista (fila de alertas, explicação e simulação "e se")
```

## Histórico de alterações do repositório

| Data | Autor | Descrição |
|---|---|---|
| 2026-10-28 | Milton Almeida Leoncio | Preparação das bases e resultados preliminares (N1) |
| 2026-11-20 | Milton Almeida Leoncio | Projeto final (N2): novos experimentos, notebooks, protótipo e artigo |
