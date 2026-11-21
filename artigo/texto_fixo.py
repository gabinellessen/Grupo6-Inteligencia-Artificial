"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: artigo/texto_fixo.py
Síntese:
    Texto das seções do artigo que não dependem dos números dos experimentos
    (identificação, introdução, fundamentação teórica e boa parte da metodologia)
    e a lista de referências bibliográficas no formato SBC.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""

TITULO = "Agente Identificador de Fraudes Bancárias"
AUTORES = ("Cheuk Ki Yu (RA 10419664), Gabriela Nellessen de Sousa (RA 10441930), "
           "Milton Almeida Leoncio (RA 10416764)")
ENDERECO = ["Faculdade de Computação e Informática – Universidade Presbiteriana Mackenzie",
            "Rua da Consolação, 930 – 01302-907 – São Paulo – SP – Brasil",
            "Inteligência Artificial – Prof. Dr. Ivan Carlos Alcântara de Oliveira"]
EMAILS = "cheuk.yu@mackenzista.com.br, gabrielanellessen@gmail.com, milton.almeida@live.com"

GITHUB = "https://github.com/gabinellessen/Grupo6-Inteligencia-Artificial"
VIDEO = "https://youtu.be/Ve9pj3yQw5M"

CONTEXTUALIZACAO = [
    "A digitalização dos serviços bancários e a popularização dos pagamentos eletrônicos aumentaram o volume "
    "de transações com cartão e, com ele, as oportunidades de fraude. As fraudes geram prejuízo para emissores, "
    "comerciantes e consumidores, e combatê-las exige decidir, em poucos milissegundos, se cada transação foge do "
    "comportamento esperado do cliente [Dal Pozzolo et al. 2014]. Sistemas tradicionais, baseados em regras fixas "
    "(por exemplo, bloquear compras de valor alto ou feitas no exterior), são fáceis de auditar, mas não se adaptam "
    "às novas estratégias dos fraudadores e costumam gerar muitos alarmes falsos.",
    "O aprendizado de máquina supervisionado trata o problema como classificação binária: a partir de transações "
    "rotuladas como legítimas ou fraudulentas, o modelo aprende a estimar o risco das novas transações. Três "
    "características tornam a tarefa difícil: o forte desbalanceamento (as fraudes costumam ser bem menos de 1% das "
    "transações), a mudança do padrão das fraudes ao longo do tempo (*concept drift*) e o custo assimétrico dos erros. "
    "Uma fraude não detectada gera perda financeira; um alarme falso bloqueia a compra de um cliente legítimo "
    "[Dal Pozzolo et al. 2018].",
]

JUSTIFICATIVA = [
    "Detectar melhor as fraudes reduz perdas e aumenta a confiança nos meios de pagamento digitais, que são um "
    "vetor de inclusão financeira. O projeto se alinha a três Objetivos de Desenvolvimento Sustentável (ODS). Ao "
    "ODS 16 (Paz, Justiça e Instituições Eficazes), especialmente à meta 16.4, de reduzir os fluxos financeiros "
    "ilícitos e combater o crime organizado. Ao ODS 8 (Trabalho Decente e Crescimento Econômico), meta 8.10, de "
    "fortalecer as instituições financeiras e ampliar o acesso a serviços bancários. E ao ODS 9 (Indústria, "
    "Inovação e Infraestrutura), pela aplicação de tecnologia à infraestrutura financeira.",
    "Como decisões automatizadas sobre transações afetam pessoas diretamente, o problema também exige atenção a "
    "privacidade, vieses e transparência, aspectos previstos na Lei Geral de Proteção de Dados (LGPD) [Brasil 2018].",
]

MOTIVACAO = [
    "Este projeto dá continuidade ao Trabalho de Conclusão de Curso de parte do grupo, aproveitado na disciplina "
    "como o enunciado permite. Na etapa anterior (N1), com Random Forest e XGBoost, o F1-score chegou a 0,86 na "
    "base Sparkov e a 0,55 na base IBM. Também se observou que modelos treinados só com o histórico antigo "
    "praticamente não detectavam as fraudes recentes. Esses resultados motivaram as perguntas desta etapa. "
    "Outros classificadores e técnicas de balanceamento superam os modelos iniciais? Quanto a otimização de "
    "hiperparâmetros acrescenta? Como manter o desempenho quando o padrão das fraudes muda? E como explicar as "
    "decisões do modelo e verificar se ele trata grupos de clientes de forma desigual?",
]

OBJETIVO = [
    "O objetivo geral é desenvolver e avaliar modelos de aprendizado de máquina supervisionado para detectar "
    "fraudes em transações com cartão de crédito, com ênfase na redução de falsos positivos, na adaptação a novas "
    "estratégias fraudulentas e no uso responsável da IA. Os objetivos específicos são: (i) comparar "
    "classificadores supervisionados entre si e com um sistema de regras estáticas; (ii) avaliar técnicas de "
    "tratamento do desbalanceamento; (iii) otimizar hiperparâmetros com validação temporal; (iv) medir o efeito "
    "da mudança de conceito e de uma estratégia de retreinamento periódico; (v) auditar vieses e explicar as "
    "decisões do modelo; e (vi) implementar um protótipo de sistema de detecção de fraudes.",
]

ORGANIZACAO = ("O restante do artigo está organizado assim. A Seção 2 resume os conceitos e os trabalhos "
               "relacionados. A Seção 3 descreve a metodologia, as bases e os resultados esperados. A Seção 4 "
               "discute os aspectos éticos e apresenta os resultados. A Seção 5 conclui.")

FUNDAMENTOS = {
    "2.1. Aprendizado supervisionado em dados desbalanceados": [
        "No aprendizado supervisionado, um modelo aprende, a partir de exemplos rotulados, a estimar a "
        "probabilidade de a transação ser fraudulenta. A decisão compara esse escore com um limiar: acima dele, "
        "a transação gera um alerta. Em bases desbalanceadas, o limiar de 0,5 raramente é adequado, e algoritmos "
        "que minimizam o erro global tendem a favorecer a classe majoritária [He and Garcia 2009].",
        "Há três estratégias usuais para lidar com isso. A ponderação de classes aumenta o peso dos erros na "
        "classe minoritária. A subamostragem aleatória descarta parte das transações legítimas. E a sobreamostragem "
        "sintética, como o SMOTE, cria fraudes artificiais por interpolação entre uma fraude e suas vizinhas mais "
        "próximas [Chawla et al. 2002].",
    ],
    "2.2. Modelos de classificação": [
        "A Regressão Logística modela o log-odds como função linear dos atributos. É fácil de interpretar, mas "
        "não captura interações entre atributos sem engenharia explícita. A Random Forest combina centenas de "
        "árvores de decisão treinadas em amostras *bootstrap* e em subconjuntos aleatórios de atributos, o que "
        "reduz a variância [Breiman 2001].",
        "O *gradient boosting* constrói árvores em sequência, cada uma corrigindo os erros das anteriores. O "
        "XGBoost [Chen and Guestrin 2016] e o LightGBM [Ke et al. 2017] são implementações eficientes, com "
        "regularização e suporte a grandes volumes de dados.",
        "O perceptron multicamadas (MLP) é uma rede neural que aprende funções não lineares por retropropagação "
        "[Goodfellow et al. 2016]. A máquina de vetores de suporte (SVM) com *kernel* RBF busca a margem máxima "
        "entre as classes em um espaço transformado [Cortes and Vapnik 1995]; seu custo de treino cresce "
        "aproximadamente com o quadrado do número de exemplos.",
        "Os hiperparâmetros foram ajustados com o Optuna [Akiba et al. 2019]. Ele usa o *Tree-structured Parzen "
        "Estimator* (TPE) [Bergstra et al. 2011] para propor novas configurações a partir das tentativas "
        "anteriores.",
    ],
    "2.3. Métricas de avaliação": [
        "Como as fraudes são raras, a acurácia é pouco informativa: classificar tudo como legítimo já resulta em "
        "acurácia acima de 99%. Por isso se usam a precisão (fração dos alertas que são fraudes), o *recall* "
        "(fração das fraudes detectadas) e o F1-score, média harmônica entre os dois. Para avaliar o escore "
        "independentemente do limiar, a área sob a curva precisão-*recall* (AUPRC, calculada como *average "
        "precision*) é mais informativa do que a área sob a curva ROC em bases desbalanceadas "
        "[Saito and Rehmsmeier 2015]. Neste trabalho, F1-score e AUPRC são as métricas principais.",
    ],
    "2.4. Mudança de conceito e vazamento de informação": [
        "Mudança de conceito (*concept drift*) é a alteração, ao longo do tempo, da relação entre os atributos e "
        "o rótulo [Gama et al. 2014]. Em fraude, ela vem da adaptação dos fraudadores e da mudança de hábitos dos "
        "clientes. Costuma ser tratada com janelas de treino recentes e retreinamento periódico "
        "[Dal Pozzolo et al. 2018].",
        "Um risco relacionado é o vazamento de informação (*leakage*). Ele ocorre quando o modelo usa, no treino, "
        "uma informação que não estará disponível no uso real ou que não terá o mesmo significado, como dados do "
        "futuro ou atributos que identificam entidades. O resultado é uma avaliação otimista [Kaufman et al. 2012].",
    ],
    "2.5. Inteligência Artificial Explicável": [
        "Os valores SHAP atribuem a cada atributo uma contribuição para a predição de uma instância, com base nos "
        "valores de Shapley da teoria dos jogos. A soma das contribuições com um valor-base reproduz o escore do "
        "modelo [Lundberg and Lee 2017]. Para modelos de árvores, o algoritmo TreeSHAP calcula esses valores de "
        "forma exata e eficiente [Lundberg et al. 2020].",
        "A explicabilidade importa em fraude por dois motivos. O analista precisa entender por que uma transação "
        "foi barrada. E a LGPD garante ao titular o direito de pedir a revisão de decisões tomadas unicamente com "
        "base em tratamento automatizado (art. 20) [Brasil 2018].",
    ],
}

RELACIONADOS_INTRO = [
    "A Tabela 2 resume trabalhos próximos a este. Os dados bibliográficos foram conferidos no Crossref e nas "
    "páginas das editoras; quando o resumo não traz números, os valores citados vêm do corpo do artigo. Para "
    "uma introdução em português aos algoritmos e às técnicas de balanceamento, ver de Souza e Bordin Jr. (2023).",
]

# trabalho | base | solução | resultado relatado
RELACIONADOS = [
    ["Dal Pozzolo et al. (2018)", "fluxo real com mais de 75 milhões de transações (3 anos)",
     "formalização realista do problema, métricas de alerta e estratégia de aprendizado para desbalanceamento, "
     "*concept drift* e atraso na verificação", "não informa métrica numérica no resumo"],
    ["Padhi et al. (2021)", "Base A deste trabalho (IBM, 24 milhões de transações)",
     "TabBERT (*transformer* tabular) gerando representações de janelas de 10 transações, com cabeça LSTM",
     "F1 = 0,86 (0,83 com atributos brutos e LSTM)"],
    ["Khalid et al. (2024)", "cartões europeus (Kaggle): 284.807 transações, 492 fraudes",
     "*ensemble* de SVM, KNN, RF, *bagging* e *boosting* com subamostragem e SMOTE",
     "acurácia de 0,947 com subamostragem"],
    ["Feng e Kim (2024)", "cartões europeus, subamostrada (550 legítimas e 492 fraudes)",
     "*compact data learning* (redução de dados e atributos) com RF+AdaBoost, GBDT, SVM, CNN e KNN",
     "acurácia entre 93% e 94%"],
    ["Theodorakopoulos et al. (2025)", "cartões europeus (Kaggle)",
     "LR, árvore de decisão, RF, XGBoost e CatBoost distribuídos em PySpark", "acurácia de 99,97% (XGBoost)"],
    ["Almusallam e Qayyum (2025)", "sintética: 1 milhão de transações, 5% de fraudes",
     "seleção de atributos (RFE, PCA, informação mútua), agrupamento e votação (SVM, RF, GBC), com SHAP",
     "F1 = 0,99; falsos positivos < 3,5%"],
    ["Btoush et al. (2026)", "cartões europeus (Kaggle), divisão aleatória 80/20",
     "sete classificadores combinados com SMOTE, Tomek Links, ENN e combinações",
     "CatBoost: F1 = 0,888 e AUPRC = 0,889 sem reamostragem; AUPRC = 0,905 com Tomek"],
]

RELACIONADOS_DISCUSSAO = [
    "Quatro dos sete trabalhos usam a base europeia do Kaggle. Ela cobre só dois dias e tem atributos "
    "anonimizados por PCA, o que impede avaliar a mudança de conceito e a engenharia de atributos "
    "comportamentais. Nesses trabalhos, a acurácia, próxima de 100% em uma base com 0,17% de fraudes, é a "
    "métrica mais destacada. Btoush et al. (2026), que reportam F1 e AUPRC, observam que o SMOTE favorece o "
    "recall e que Tomek Links e ENN favorecem a precisão; mas a divisão aleatória dos dados não separa passado "
    "e futuro.",
    "Padhi et al. (2021) usam a mesma Base A, com um modelo sequencial profundo. Seu protocolo, porém, é "
    "diferente (janelas de transações e sobreamostragem da classe fraude), o que impede comparar os números "
    "diretamente.",
    "Este trabalho se diferencia em quatro pontos: avalia os modelos em divisão temporal, com intervalos de "
    "confiança; mede o efeito da mudança de conceito e do retreinamento; identifica e corrige um vazamento por "
    "identidade; e audita vieses entre grupos de clientes, explicando cada alerta.",
]

# Referências (formato SBC). Os dados dos trabalhos relacionados foram conferidos no Crossref/editoras (N2).
REFERENCIAS = [
    "Almusallam, N. and Qayyum, J. (2025). A hybrid feature selection and clustering-based ensemble learning "
    "approach for real-time fraud detection in financial transactions. *Computers, Materials & Continua*, "
    "85(2):3653–3687.",
    "Btoush, E., Kobbaey, T., Tamimi, H. and Zhou, X. (2026). Machine learning-based cyber fraud detection: a "
    "comparative study of resampling methods for imbalanced credit card data. *Applied Sciences*, 16(2):850.",
    "Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C. and Bontempi, G. (2018). Credit card fraud detection: "
    "a realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning "
    "Systems*, 29(8):3784–3797.",
    "de Souza, D. H. M. and Bordin Jr., C. J. (2023). Detecção de fraude de cartão de crédito por meio de "
    "algoritmos de aprendizado de máquina. *Revista Brasileira de Computação Aplicada*, 15(1):1–11.",
    "Feng, X. and Kim, S.-K. (2024). Novel machine learning based credit card fraud detection systems. "
    "*Mathematics*, 12(12):1869.",
    "Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J. and Adejoh, J. (2024). Enhancing credit card "
    "fraud detection: an ensemble machine learning approach. *Big Data and Cognitive Computing*, 8(1):6.",
    "Padhi, I., Schiff, Y., Melnyk, I., Rigotti, M., Mroueh, Y., Dognin, P., Ross, J., Nair, R. and Altman, E. "
    "(2021). Tabular transformers for modeling multivariate time series. In *ICASSP 2021 – IEEE International "
    "Conference on Acoustics, Speech and Signal Processing*, pages 3565–3569. IEEE.",
    "Theodorakopoulos, L., Theodoropoulou, A., Tsimakis, A. and Halkiopoulos, C. (2025). Big data-driven "
    "distributed machine learning for scalable credit card fraud detection using PySpark, XGBoost, and CatBoost. "
    "*Electronics*, 14(9):1754.",
    "Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019). Optuna: a next-generation hyperparameter "
    "optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge "
    "Discovery & Data Mining*, pages 2623–2631. ACM.",
    "Altman, E. R. (2019). Synthesizing credit card transactions. *arXiv preprint* arXiv:1910.03033.",
    "Bergstra, J., Bardenet, R., Bengio, Y. and Kégl, B. (2011). Algorithms for hyper-parameter optimization. "
    "In *Advances in Neural Information Processing Systems 24*, pages 2546–2554.",
    "Brasil (2018). Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD). "
    "*Diário Oficial da União*, Brasília, DF.",
    "Breiman, L. (2001). Random forests. *Machine Learning*, 45(1):5–32.",
    "Chawla, N. V., Bowyer, K. W., Hall, L. O. and Kegelmeyer, W. P. (2002). SMOTE: synthetic minority "
    "over-sampling technique. *Journal of Artificial Intelligence Research*, 16:321–357.",
    "Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. In *Proceedings of the 22nd ACM "
    "SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 785–794. ACM.",
    "Cortes, C. and Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3):273–297.",
    "Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S. and Bontempi, G. (2014). Learned lessons in "
    "credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, "
    "41(10):4915–4928.",
    "Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall, New York.",
    "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. and Bouchachia, A. (2014). A survey on concept drift "
    "adaptation. *ACM Computing Surveys*, 46(4):44:1–44:37.",
    "Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*. MIT Press, Cambridge.",
    "Hardt, M., Price, E. and Srebro, N. (2016). Equality of opportunity in supervised learning. In *Advances in "
    "Neural Information Processing Systems 29*, pages 3315–3323.",
    "Harris, B. (2020). Sparkov Data Generation. GitHub. "
    "https://github.com/namebrandon/Sparkov_Data_Generation. Acesso em: 28 out. 2026.",
    "He, H. and Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data "
    "Engineering*, 21(9):1263–1284.",
    "Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in data mining: formulation, "
    "detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4):15:1–15:21.",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: a highly "
    "efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*, "
    "pages 3146–3154.",
    "Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in "
    "Neural Information Processing Systems 30*, pages 4765–4774.",
    "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., "
    "Bansal, N. and Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for "
    "trees. *Nature Machine Intelligence*, 2(1):56–67.",
    "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning "
    "Research*, 12:2825–2830.",
    "Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when "
    "evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3):e0118432.",
]
