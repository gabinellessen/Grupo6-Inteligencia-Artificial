"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: artigo/resultados.py
Síntese:
    Seções 4 (Resultados: aspectos éticos, resultados dos experimentos, GitHub e
    vídeo) e 5 (Conclusão) do artigo. Todos os números vêm dos CSVs de resultados.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import pandas as pd

import texto_fixo as T
from dados_resultados import Resultados, dif_txt, ic_txt
from docx_sbc import br, milhar


def _d(v, casas=3):
    """Diferença com sinal."""
    return f"{'+' if v >= 0 else '−'}{br(abs(v), casas)}"


def _ic_dif(r, pref):
    return f"IC 95%: {br(r[pref + '_ic_inf'])} a {br(r[pref + '_ic_sup'])}"


def secao(art, RES, prototipo):
    R = Resultados(RES)
    e1A, e1B = R.e1_final("A"), R.e1_final("B")
    e1Ac = R.e1["A"][R.e1["A"]["conjunto"] == "completo"].set_index("modelo")
    e2 = {k: R.e2[k].set_index(["modelo", "estrategia"]) for k in ("A", "B")}
    e3 = {k: R.e3[k].set_index("modelo") for k in ("A", "B")}
    e5r = {k: R.e5r[k].set_index("modelo") for k in ("A", "B")}
    e5d = {k: R.e5d[k].set_index(["modelo", "atributo"]) for k in ("A", "B")}
    gA = R.e5g["A"].set_index(["modelo", "atributo", "grupo"])
    gB = R.e5g["B"].set_index(["modelo", "atributo", "grupo"])
    fA, fB = R.final["A"], R.final["B"]
    t4 = R.e4t

    art.h1("4. Resultados")

    # ------------------------------------------------------------------ 4.1 ética
    art.h2("4.1. Aspectos éticos e responsabilidade no uso da IA")
    art.par("Um sistema antifraude decide sobre o dinheiro e a rotina das pessoas, e erra nos dois sentidos. A fraude "
            "não detectada causa prejuízo à vítima e à instituição. O falso alarme bloqueia a compra de um cliente "
            "legítimo, às vezes em viagem, e gera constrangimento. Por isso, a responsabilidade no uso da IA foi "
            "tratada como requisito do projeto, e não como apêndice. Os riscos identificados e as medidas adotadas "
            "estão a seguir.")
    art.par("**Privacidade e proteção de dados.** As duas bases são sintéticas e não contêm dados de pessoas reais. "
            "Mesmo assim, a preparação descartou número do cartão, CVV, validade, nome e endereço, seguindo os "
            "princípios de finalidade e necessidade da LGPD (art. 6º). Em um uso real, o tratamento para prevenção "
            "à fraude tem amparo legal, inclusive para dados sensíveis (art. 11, II, *g*), mas exige minimização, "
            "controle de acesso e registro das decisões [Brasil 2018].")

    sA_c, sA_s = e5r["A"].loc["Com atributos sensíveis"], e5r["A"].loc["Sem atributos sensíveis"]
    dA = lambda m, a: e5d["A"].loc[(m, a)]
    rec = lambda m, g: gA.loc[(m, "Quartil de renda", g), "recall"]
    art.par("**Vieses e discriminação.** A Base A traz gênero, idade, renda, dívida e score de crédito. Para verificar "
            "se o modelo trata grupos de forma desigual, comparamos o recall (chance de uma fraude ser detectada) e "
            "a taxa de falsos alarmes entre grupos. Isso segue a ideia de igualdade de oportunidade [Hardt et al. "
            f"2016]. A Tabela {art.proxima_tabela()} e a Figura {art.proxima_figura()} mostram o resultado.")
    art.par(f"O modelo final da Base A não usa atributos sensíveis. Ele é ao mesmo tempo mais preciso e mais "
            f"equilibrado do que a variante que os inclui. O F1-score cai de {br(sA_s['f1'])} para "
            f"{br(sA_c['f1'])} com os atributos sensíveis ({_ic_dif(sA_c, 'dif_f1')}).")
    art.par(f"A maior diferença de recall entre faixas de renda cai de "
            f"{br(dA('Com atributos sensíveis', 'Quartil de renda')['dif_recall_max_min'], 2)} para "
            f"{br(dA('Sem atributos sensíveis', 'Quartil de renda')['dif_recall_max_min'], 2)}. Entre faixas "
            f"etárias, cai de {br(dA('Com atributos sensíveis', 'Faixa etária')['dif_recall_max_min'], 2)} para "
            f"{br(dA('Sem atributos sensíveis', 'Faixa etária')['dif_recall_max_min'], 2)}. Entre gêneros, de "
            f"{br(dA('Com atributos sensíveis', 'Gênero')['dif_recall_max_min'], 2)} para "
            f"{br(dA('Sem atributos sensíveis', 'Gênero')['dif_recall_max_min'], 2)}. As taxas de falsos alarmes "
            f"ficam entre {br(dA('Sem atributos sensíveis', 'Quartil de renda')['fpr_10k_min'], 1)} e "
            f"{br(dA('Sem atributos sensíveis', 'Quartil de renda')['fpr_10k_max'], 1)} por 10 mil transações "
            "legítimas em todos os grupos.")
    art.par("Ainda resta um viés relevante. Os clientes do quartil de maior renda têm recall de "
            f"{br(rec('Sem atributos sensíveis', 'Q4 (maior)'), 2)}, contra "
            f"{br(rec('Sem atributos sensíveis', 'Q1 (menor)'), 2)} no de menor renda. Ou seja, as fraudes contra "
            "eles são detectadas com menos frequência. Como a renda não é entrada do modelo, a diferença vem de "
            "padrões de consumo correlacionados a ela, um efeito que remover atributos não elimina.")

    sB_c, sB_s = e5r["B"].loc["Com atributos sensíveis"], e5r["B"].loc["Sem atributos sensíveis"]
    rB = lambda m, g: gB.loc[(m, "Faixa etária", g)]
    art.par(f"Na Base B, retirar a idade reduz o F1-score de {br(sB_c['f1'])} para {br(sB_s['f1'])} "
            f"({_ic_dif(sB_s, 'dif_f1')}) e não reduz a diferença de recall entre faixas etárias. Ela passa de "
            f"{br(e5d['B'].loc[('Com atributos sensíveis', 'Faixa etária'), 'dif_recall_max_min'], 2)} para "
            f"{br(e5d['B'].loc[('Sem atributos sensíveis', 'Faixa etária'), 'dif_recall_max_min'], 2)}, com "
            f"recall de {br(rB('Com atributos sensíveis', '30-44')['recall'], 2)} entre 30 e 44 anos e de "
            f"{br(rB('Com atributos sensíveis', '60 ou mais')['recall'], 2)} acima de 60. Como o teste tem só "
            f"{milhar(rB('Com atributos sensíveis', '30-44')['fraudes'])} a "
            f"{milhar(rB('Com atributos sensíveis', '60 ou mais')['fraudes'])} fraudes por faixa, essas diferenças "
            "são instáveis.")
    art.par("O modelo final da Base B mantém a idade porque foi escolhido pela validação. Para uso real, porém, "
            "recomendamos a variante sem idade. A perda de desempenho é pequena, e o princípio da não "
            "discriminação da LGPD (art. 6º, IX) pede que se evite basear a decisão na idade do cliente.")

    linhas = []
    for chave, rot in (("A", "A – IBM"), ("B", "B – Sparkov")):
        for modelo in ("Sem atributos sensíveis", "Com atributos sensíveis"):
            x = e5r[chave].loc[modelo]
            atrs = ["Gênero", "Faixa etária", "Quartil de renda"] if chave == "A" else ["Faixa etária"]
            disp = e5d[chave]
            linhas.append([rot, modelo + (" (final)" if x["final"] else ""), str(int(x["n_atributos"])),
                           br(x["f1"]), br(x["pr_auc"]),
                           "; ".join(f"{a.split()[0].lower()}: {br(disp.loc[(modelo, a), 'dif_recall_max_min'], 2)}" for a in atrs),
                           "; ".join(f"{br(disp.loc[(modelo, a), 'fpr_10k_min'], 1)}–{br(disp.loc[(modelo, a), 'fpr_10k_max'], 1)}" for a in atrs)])
    art.tabela(["Base", "Modelo", "Atrib.", "F1", "AUPRC", "Maior diferença de recall entre grupos",
                "Falsos alarmes por 10 mil (mín.–máx.)"], linhas,
               "Desempenho e disparidades entre grupos com e sem atributos sensíveis (mesmos hiperparâmetros). "
               "Grupos: gênero, faixa etária e quartil de renda (A); faixa etária (B).",
               larguras_cm=[1.7, 3.0, 1.1, 1.0, 1.1, 3.6, 3.4], fonte=8)
    art.figura(RES / "fig_exp5_grupos_A.png", "Recall e taxa de falsos positivos por grupo na Base A, com e sem "
               "atributos sensíveis. O modelo final (sem atributos sensíveis) reduz as diferenças de recall, "
               "mas o quartil de maior renda continua menos protegido.", 15.0)

    art.par("**Transparência e revisão humana.** Cada alerta do protótipo vem com os três atributos que mais "
            "aumentaram o risco (Seção 4.2.5). Isso permite que o analista confirme ou descarte o alerta e "
            "explique a decisão ao cliente, como exige o direito de revisão de decisões automatizadas (LGPD, "
            "art. 20). O sistema é de apoio à decisão: ele ordena a fila de análise, mas não bloqueia sozinho.")
    art.par("**Falsos positivos e segurança.** As explicações mostram que, na Base A, o modelo depende muito de "
            "*compra no exterior*. Clientes legítimos em viagem são, portanto, os mais expostos a falsos alarmes. "
            "Além disso, um fraudador que conheça esse padrão pode contorná-lo, como mostrou a mudança de "
            "conceito. Por isso, os detalhes do modelo devem ficar restritos à equipe antifraude, e o desempenho "
            "precisa ser monitorado continuamente.")
    art.par("**Medidas concretas propostas e adotadas.** Seis medidas resumem o que foi feito e o que se recomenda "
            "para um uso real.")
    art.lista([
        "minimização dos dados, com descarte de identificadores e atributos de perfil sem necessidade (adotada);",
        "auditoria mensal de recall e falsos alarmes por grupo, com revisão do modelo quando a diferença de "
        "recall passar de 0,10 (proposta; a auditoria está automatizada no Experimento 5);",
        "explicação de cada alerta, decisão final humana e canal de contestação para o cliente (adotada no "
        "protótipo);",
        "em vez de bloquear a compra, pedir uma verificação adicional (por exemplo, confirmação no aplicativo), o "
        "que reduz o dano do falso positivo (proposta);",
        "monitoramento do desempenho e retreinamento mensal (avaliado no Experimento 4);",
        "controle de acesso ao modelo, versionamento e registro das decisões para prestação de contas (proposta).",
    ])
    art.par("**Limitação ética dos dados.** Por serem sintéticos, os dados reproduzem as escolhas dos simuladores, "
            "inclusive a forma como associam perfis demográficos a padrões de gasto. As disparidades medidas aqui "
            "ilustram o método de auditoria, mas não descrevem a população real. Elas precisam ser refeitas com "
            "dados reais antes de qualquer implantação.")

    # ------------------------------------------------------------------ 4.2 resultados
    art.h2("4.2. Resultados obtidos")
    art.h2("4.2.1. Comparação de classificadores (E1)")
    t_e1 = art.proxima_tabela()
    f_e1 = art.proxima_figura()
    art.par(f"A Tabela {t_e1} e a Figura {f_e1} mostram o desempenho no teste, no conjunto de atributos final de "
            "cada base.")
    art.par(f"Na Base B, o XGBoost (F1 = {br(e1B.loc['XGBoost', 'f1'])}) e o LightGBM "
            f"({br(e1B.loc['LightGBM', 'f1'])}) ficaram à frente, com intervalos de confiança sobrepostos. Em "
            f"seguida vieram Random Forest ({br(e1B.loc['Random Forest', 'f1'])}), MLP "
            f"({br(e1B.loc['MLP', 'f1'])}) e SVM ({br(e1B.loc['SVM (RBF)', 'f1'])}).")
    art.par(f"Na Base A, os melhores foram a MLP (F1 = {br(e1A.loc['MLP', 'f1'])}) e o XGBoost "
            f"({br(e1A.loc['XGBoost', 'f1'])}). O LightGBM com ponderação integral foi instável "
            f"({br(e1A.loc['LightGBM', 'f1'])}): sua AUPRC na validação caiu já a partir da segunda árvore, efeito "
            "do peso de cerca de 1.500 para 1 da classe positiva. O Experimento 3 corrigiu esse problema.")
    art.par(f"O sistema de regras estáticas foi o pior nas duas bases. Na Base A, ele detectou "
            f"{br(100 * e1A.loc['Regras', 'recall'], 0)}% das fraudes, mas com {milhar(e1A.loc['Regras', 'fp'])} "
            f"falsos alarmes no ano (precisão de {br(e1A.loc['Regras', 'precision'], 2)}). Isso confirma a "
            "limitação das regras fixas que motivou o projeto.")
    art.par(f"A Regressão Logística foi a segunda pior na Base B (F1 = {br(e1B.loc['Regressão Logística', 'f1'])}). "
            "Lá as fraudes dependem de interações, como valor alto à noite, que um modelo linear não captura. Os "
            f"valores de ROC-AUC, acima de {br(min(e1A.loc[m, 'roc_auc'] for m in ['Random Forest', 'XGBoost', 'MLP']), 3)} "
            "na Base A para modelos com F1 entre 0,6 e 0,7, mostram como essa métrica é otimista em dados "
            "desbalanceados [Saito and Rehmsmeier 2015].")
    art.tabela(["Base", "Modelo", "Precisão", "Recall", "F1 [IC 95%]", "AUPRC [IC 95%]", "ROC-AUC"],
               R.tabela_e1(), "Desempenho no teste (E1), com limiar escolhido na validação. Base A no conjunto sem "
               "perfil estático; modelos de árvores com ponderação de classes.",
               larguras_cm=[1.9, 3.1, 1.4, 1.3, 2.9, 2.9, 1.5], fonte=8)
    art.figura(RES / "fig_exp1_f1_ic.png", "F1-score no teste com IC 95% (bootstrap) por modelo e base.", 15.0)

    xg, rf, ml, lr = (e1A.loc[m] for m in ("XGBoost", "Random Forest", "MLP", "Regressão Logística"))
    art.par(f"**Perfil estático e vazamento por identidade.** A Tabela {art.proxima_tabela()} compara os dois "
            "conjuntos de atributos da Base A. Retirar os 14 atributos de perfil estático aumentou a AUPRC de "
            f"validação do XGBoost de {br(e1Ac.loc['XGBoost', 'val_pr_auc'])} para {br(xg['val_pr_auc'])}, a "
            "maior da Base A no E1. Por isso, esse passou a ser o conjunto final.")
    art.par(f"No teste, o ganho de F1 foi de {_d(xg['dif_f1_vs_completo'])} no XGBoost, "
            f"{_d(rf['dif_f1_vs_completo'])} na Random Forest e {_d(ml['dif_f1_vs_completo'])} na MLP, todos "
            "significativos. Já a Regressão Logística perdeu "
            f"{br(abs(lr['dif_f1_vs_completo']))}. Ela não consegue isolar cartões específicos a partir de "
            "combinações de atributos e, por isso, não sofria com a memorização.")
    art.par("O efeito é maior justamente nos modelos mais capazes de memorizar, os *ensembles* de árvores. Isso, "
            "somado ao fato de os cartões fraudados no teste serem outros (Seção 3.2), confirma o vazamento por "
            "identidade. Em relação à N1, a correção elevou o F1 do XGBoost na Base A de "
            f"{br(e1Ac.loc['XGBoost', 'f1'])} para {br(xg['f1'])}.")
    art.tabela(["Modelo", "AUPRC validação", "AUPRC teste", "F1 teste", "ΔF1 teste"], R.tabela_perfil(),
               "Base A: atributos completos → sem perfil estático. ΔF1 com * quando o IC 95% pareado não contém zero.",
               larguras_cm=[3.6, 3.0, 3.0, 3.0, 2.0], fonte=8.5)

    # E2
    art.h2("4.2.2. Tratamento do desbalanceamento (E2)")
    xA = lambda e: e2["A"].loc[("XGBoost", e)]
    rA = lambda e: e2["A"].loc[("Random Forest", e)]
    xB = lambda e: e2["B"].loc[("XGBoost", e)]
    art.par(f"A Tabela {art.proxima_tabela()} mostra que o melhor tratamento depende da base. Na Base B, ponderar as "
            "classes ou não tratar o desbalanceamento dá resultados equivalentes (diferenças não significativas). "
            "Subamostragem e SMOTE pioraram o F1 dos dois modelos: o SMOTE reduziu o F1 do XGBoost em "
            f"{br(abs(xB('SMOTE 10%')['dif_f1_vs_ponderacao']))} e também o recall "
            f"({br(xB('SMOTE 10%')['recall'])} contra {br(xB('Ponderação de classes')['recall'])}).")
    art.par("Na Base A, a ponderação integral foi pior do que não tratar o desbalanceamento: o F1 subiu "
            f"{_d(rA('Sem tratamento')['dif_f1_vs_ponderacao'])} na Random Forest e "
            f"{_d(xA('Sem tratamento')['dif_f1_vs_ponderacao'])} no XGBoost sem pesos. O melhor resultado veio do "
            f"XGBoost com SMOTE (F1 = {br(xA('SMOTE 10%')['f1'])}, AUPRC = {br(xA('SMOTE 10%')['pr_auc'])}). O "
            f"ganho, porém, foi de precisão ({br(xA('SMOTE 10%')['precision'])} contra "
            f"{br(xA('Ponderação de classes')['precision'])}), e não de recall.")
    art.par("Assim, a hipótese (ii) não se confirmou, ao contrário do que relatam Btoush et al. (2026) com divisão "
            "aleatória. Uma interpretação é que, com poucos cartões fraudados e mudança de conceito, pesos muito "
            "altos fazem o modelo se ajustar demais às fraudes do treino. Já as fraudes sintéticas do SMOTE "
            "suavizam a fronteira de decisão. A subamostragem descarta mais de 90% das transações legítimas e "
            "piorou o desempenho nas duas bases.")
    art.tabela(["Estratégia (só no treino)", "B: RF", "B: XGBoost", "A: RF", "A: XGBoost"], R.tabela_e2(),
               "F1 no teste por estratégia de balanceamento (E2). * indica diferença significativa em relação à "
               "ponderação de classes (IC 95% pareado sem o zero).",
               larguras_cm=[4.6, 2.3, 2.3, 2.3, 2.3], fonte=8.5)

    # E3
    art.h2("4.2.3. Otimização de hiperparâmetros e modelo final (E3)")
    lA, mA, xA3 = e3["A"].loc["LightGBM"], e3["A"].loc["MLP"], e3["A"].loc["XGBoost"]
    lB, mB, xB3 = e3["B"].loc["LightGBM"], e3["B"].loc["MLP"], e3["B"].loc["XGBoost"]
    art.par(f"A Tabela {art.proxima_tabela()} compara os modelos otimizados com a configuração padrão do E1. Na "
            f"Base B, a otimização não mudou o F1 do XGBoost ({_d(xB3['dif_f1_vs_padrao'])}) nem do LightGBM "
            f"({_d(lB['dif_f1_vs_padrao'])}); a base já estava perto do teto. A otimização aumentou a AUPRC do "
            f"LightGBM ({_d(lB['dif_pr_auc_vs_padrao'])}, significativo) e o F1 da MLP "
            f"({_d(mB['dif_f1_vs_padrao'])}).")
    art.par(f"Na Base A, a otimização trouxe ganhos significativos para todos. O XGBoost foi de "
            f"{br(xA3['f1_padrao'])} para {br(xA3['f1'])}, e a MLP de {br(mA['f1_padrao'])} para {br(mA['f1'])}. "
            f"O LightGBM se estabilizou: de {br(lA['f1_padrao'])} para {br(lA['f1'])}.")
    art.par("As configurações escolhidas na Base A têm forte regularização: no XGBoost, profundidade 5 e "
            "regularização L2 de 86; no LightGBM, ao menos 823 transações por folha e 37% dos atributos por árvore. "
            "Também usam um peso da classe positiva muito menor que o integral (cerca de 4 e de 37, em vez de "
            "1.478). Isso é coerente com o E2: na Base A, generalizar para outro período e para outros cartões "
            "exige um modelo mais conservador.")
    art.par(f"O modelo final de cada base foi o LightGBM otimizado, que teve a maior AUPRC na validação "
            f"({br(fB['val_pr_auc'])} na Base B e {br(fA['val_pr_auc'])} na Base A). No teste, ele chegou a "
            f"F1 = {br(fB['teste']['f1'])} e AUPRC = {br(fB['teste']['pr_auc'])} na Base B, e a "
            f"F1 = {br(fA['teste']['f1'])} e AUPRC = {br(fA['teste']['pr_auc'])} na Base A.")
    art.par(f"Na Base A, a MLP otimizada foi um pouco melhor no teste (F1 = {br(mA['f1'])}), apesar da validação "
            f"menor ({br(mA['val_pr_auc'])}). Mantivemos a escolha feita pela validação, porque escolher pelo teste "
            "tornaria a avaliação otimista. A divergência entre validação e teste é mais um sinal de que o "
            "comportamento das fraudes muda entre os períodos.")
    art.par(f"Com limite de 20 minutos por família, a busca da Base A completou {int(xA3['tentativas'])}, "
            f"{int(lA['tentativas'])} e {int(mA['tentativas'])} tentativas, menos do que o planejado. Uma busca "
            "maior poderia melhorar esses números.")
    art.tabela(["Base", "Modelo", "Tent.", "AUPRC val.", "F1 padrão → otim.", "ΔF1", "AUPRC padrão → otim."],
               R.tabela_e3(), "Otimização com Optuna (E3): AUPRC na validação e métricas no teste. "
               "* = diferença pareada significativa.", larguras_cm=[1.9, 2.6, 1.1, 1.7, 3.0, 1.6, 3.2], fonte=8)

    # E4
    art.h2("4.2.4. Mudança de conceito e retreinamento (E4)")
    est, ret, hist = (t4.loc[n] for n in ("Estático (janela recente)", "Retreino mensal", "Estático (histórico até 2016)"))
    mensal = R.e4m[R.e4m["fraudes"] > 0].pivot(index="mes", columns="estrategia", values="f1")
    meses_pos = mensal.index[1:]
    melhor = int((mensal.loc[meses_pos, "Retreino mensal"] > mensal.loc[meses_pos, "Estático (janela recente)"]).sum())
    ganho_2s = (mensal.loc["2019-06":"2019-10", "Retreino mensal"] - mensal.loc["2019-06":"2019-10", "Estático (janela recente)"])
    lim = pd.read_json(RES / "exp4_limiares.json", typ="series")["limiares_retreino"]
    art.par(f"A Figura {art.proxima_figura()} mostra o F1 mês a mês em 2019. Novembro e dezembro não aparecem porque "
            "não têm fraudes rotuladas na base. O modelo treinado só com o histórico até 2016, quando as fraudes "
            f"eram online, ficou com F1 de {br(hist['f1'])} no ano, contra {br(est['f1'])} do mesmo algoritmo "
            "treinado na janela recente. Esse resultado reproduz, com o modelo final, a conclusão da N1: a "
            "atualidade dos dados importa mais do que o volume.")
    art.par(f"O retreino mensal elevou o F1 anual para {br(ret['f1'])} e a AUPRC para {br(ret['pr_auc'])}. Foram "
            f"{milhar(ret['tp'] - est['tp'])} fraudes detectadas a mais, com {milhar(ret['fp'] - est['fp'])} falsos "
            f"alarmes a mais. Em janeiro os dois modelos coincidem. Nos nove meses seguintes, o retreino foi melhor "
            f"em {melhor}, com ganhos de {br(ganho_2s.min(), 2)} a {br(ganho_2s.max(), 2)} entre junho e outubro, "
            "quando o modelo estático se distancia do período de treino.")
    art.par(f"O limiar escolhido na validação variou de {br(min(lim.values()), 2)} a {br(max(lim.values()), 2)} "
            "entre os meses. Isso mostra que o limiar também precisa ser recalibrado.")
    art.figura(RES / "fig_exp4_drift_mensal.png", "F1-score mensal na Base A em 2019 para as três estratégias de "
               "treinamento (E4).", 14.5)

    # XAI
    art.h2("4.2.5. Explicabilidade")
    impA = R.e5i["A"]["media_abs_contribuicao"]
    impB = R.e5i["B"]["media_abs_contribuicao"]
    from comum import rotulo
    topA = ", ".join(rotulo(n) for n in impA.index[:5])
    topB = ", ".join(rotulo(n) for n in impB.index[:5])
    locB = R.e5l["B"]
    locA = R.e5l["A"]
    art.par(f"A Figura {art.proxima_figura()} mostra a importância global dos atributos (média do valor absoluto "
            "das contribuições TreeSHAP) no modelo final. Na Base B, os atributos mais importantes são "
            f"{topB}. Isso é coerente com a análise exploratória (fraudes de valor alto, à noite e em sequência) e "
            "confirma o papel dos atributos comportamentais.")
    art.par(f"Na Base A, os mais importantes são {topA}. Como todas as fraudes do período recente foram compras no "
            "exterior, o modelo aprendeu esse padrão.")
    art.figura(RES / "fig_exp5_importancia_global.png", "Importância global dos atributos no modelo final de cada "
               "base (TreeSHAP).", 15.0)
    tpB = locB[locB["tipo"] == "Fraude detectada"].iloc[0]
    fpB = locB[locB["tipo"] == "Falso alarme"]
    art.par(f"A Figura {art.proxima_figura()} explica dois alertas da Base B. A fraude de maior escore foi uma "
            f"compra de {tpB['valor']} às 22h, que elevou para "
            f"{locB[(locB.tipo == 'Fraude detectada') & (locB.atributo == 'valor_24h')]['valor'].iloc[0]} o total "
            "gasto pelo cliente em 24 horas.")
    art.par(f"O falso alarme de maior escore tem o mesmo perfil: {fpB[fpB.atributo == 'valor']['valor'].iloc[0]} "
            "às 22h. É o tipo de caso que o analista precisa revisar, e a explicação mostra por quê. Na Base A, os "
            "dois exemplos equivalentes são dominados por *compra no exterior* e *estado diferente do titular*. "
            "O falso alarme é uma compra legítima de "
            f"{locA[(locA.tipo == 'Falso alarme') & (locA.atributo == 'valor')]['valor'].iloc[0]} fora do país.")
    art.figura(RES / "fig_exp5_local_B.png", "Explicação local de uma fraude detectada e de um falso alarme na "
               "Base B: contribuição de cada atributo para o risco (vermelho aumenta, azul diminui).", 15.0)

    # Protótipo
    art.h2("4.2.6. Protótipo")
    p = prototipo
    art.par(f"A Figura {art.proxima_figura()} mostra o painel do analista no dia {p['dia']} da Base B. Das "
            f"{milhar(p['dia_transacoes'])} transações, {p['dia_alertas']} geraram alerta, e todas eram fraudes. "
            "Cada linha traz os motivos do alerta em linguagem simples.")
    art.par(f"Na simulação dos {p['dias']} dias do teste da Base B, o protótipo emitiu {milhar(p['alertas'])} "
            f"alertas para {milhar(p['transacoes'])} transações (menos de dois por dia). Detectou "
            f"{milhar(p['detectadas'])} das {milhar(p['detectadas'] + p['perdidas'])} fraudes, com "
            f"{milhar(p['falsos'])} falsos alarmes. Processando um dia por vez, o custo mediano foi de "
            f"{br(p['ms'], 3)} ms por transação, incluindo as explicações. Isso é compatível com o uso em tempo "
            "real, embora o cálculo dos atributos em produção exija acesso ao histórico do cliente.")
    art.figura(RES / "fig_prototipo_painel.png", "Painel do protótipo (Streamlit): indicadores do dia, fila de "
               "alertas ordenada por risco e motivos de cada alerta.", 15.0)

    # Discussão
    art.h2("4.2.7. Discussão e limitações")
    art.par("Em relação aos resultados esperados (Seção 3.7), três hipóteses se confirmaram. Os *ensembles* de "
            "árvores superaram regras e modelos lineares (i), embora a MLP tenha sido competitiva na Base A. A "
            "otimização melhorou os modelos na base mais difícil (iii). O retreinamento reduziu a perda causada "
            "pela mudança de conceito (iv).")
    art.par("Duas hipóteses não se confirmaram como esperado. O SMOTE não aumentou o recall (ii); seu benefício na "
            "Base A foi de precisão. E as explicações, embora coerentes com a análise exploratória (v), revelaram "
            "a dependência do modelo da Base A de um único padrão. O achado mais importante não estava previsto: "
            "o vazamento por identidade causado pelos atributos de perfil estático, que só aparece com divisão "
            "temporal e titulares diferentes no teste.")
    art.par("As principais limitações são quatro. Primeiro, as bases são sintéticas, e os números não se transferem "
            "diretamente para dados reais. Segundo, o teste da Base B tem apenas 171 fraudes de 16 clientes, o que "
            "explica os intervalos de confiança largos. Terceiro, a busca de hiperparâmetros da Base A foi limitada "
            "pelo tempo. Quarto, o retreino supõe rótulos disponíveis ao fim de cada mês; na prática, a "
            "confirmação de uma fraude pode levar semanas [Dal Pozzolo et al. 2018]. Além disso, o protótipo usa "
            "atributos já calculados e não implementa o cálculo em fluxo contínuo.")

    # ------------------------------------------------------------------ 4.3
    art.h2("4.3. Repositório no GitHub e vídeo")
    art.par("Todo o material do projeto está no repositório público indicado abaixo: este artigo, a descrição das bases "
            "(com a Base B incluída), os notebooks executados de cada etapa, o código-fonte com cabeçalhos e "
            "histórico de alterações, as tabelas e figuras de resultados, os modelos finais e o protótipo. O "
            "arquivo README documenta as dependências e os passos para reproduzir os experimentos.")
    for rot, url in (("Repositório", T.GITHUB), ("Vídeo", T.VIDEO)):
        p = art.par(f"**{rot}:** {url}", recuo=False)
        p.alignment = 0  # à esquerda


def conclusao(art, RES):
    R = Resultados(RES)
    fA, fB = R.final["A"], R.final["B"]
    e1A = R.e1["A"].set_index(["conjunto", "modelo"])
    t4 = R.e4t
    art.h1("5. Conclusão")
    art.par("Este trabalho aplicou aprendizado de máquina supervisionado à detecção de fraudes em transações com "
            "cartão, em duas bases públicas e sintéticas avaliadas com divisão temporal. Foram comparados seis "
            "classificadores e um sistema de regras, cinco estratégias de balanceamento e a otimização com Optuna. "
            "Também foram medidos o efeito da mudança de conceito, os vieses entre grupos e as explicações de cada "
            "alerta, e o modelo final foi empacotado em um protótipo para o analista antifraude.")
    art.par("O objetivo geral foi alcançado. Na Base B, o modelo final detectou "
            f"{br(100 * fB['teste']['recall'], 0)}% das fraudes com precisão de {br(fB['teste']['precision'], 2)} "
            f"(F1 = {br(fB['teste']['f1'])}). Na Base A, mais difícil por causa da mudança de conceito, o F1 subiu "
            f"de {br(e1A.loc[('completo', 'Random Forest'), 'f1'])} na N1 para {br(fA['teste']['f1'])}. Com "
            f"retreino mensal, chegou a {br(t4.loc['Retreino mensal', 'f1'])}. Em ambas as bases, os modelos "
            "superaram com folga o sistema de regras estáticas.")
    art.par("Os objetivos específicos foram cumpridos, com uma ressalva: a SVM foi avaliada só na Base B, por "
            "custo computacional.")
    art.par("As contribuições efetivas são cinco:")
    art.lista([
        "um *pipeline* reprodutível, com atributos comportamentais sem vazamento temporal, divisão temporal e "
        "intervalos de confiança por bootstrap pareado;",
        "a identificação e a correção de um vazamento por identidade na base IBM, em que atributos de perfil do "
        "titular permitiam memorizar cartões fraudados; a correção melhorou o desempenho e a equidade ao mesmo "
        "tempo;",
        "evidência de que, sob mudança de conceito, pesos de classe muito altos prejudicam a generalização e de "
        "que o retreinamento mensal recupera parte do desempenho;",
        "uma auditoria de vieses por grupo com medidas concretas de mitigação e explicações TreeSHAP para cada "
        "alerta;",
        "um protótipo de apoio à decisão, com fila de alertas explicados e simulação \"e se\".",
    ])
    art.par("Como trabalhos futuros, propõem-se:")
    art.lista([
        "validar a abordagem com dados reais e rótulos atrasados;",
        "calcular os atributos em fluxo contínuo;",
        "combinar o retreinamento com a detecção automática de mudança de conceito;",
        "investigar modelos sequenciais, como os de Padhi et al. (2021);",
        "incorporar à auditoria de vieses o efeito de atributos correlacionados à renda.",
    ])
