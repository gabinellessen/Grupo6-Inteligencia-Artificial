"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/exp3_otimizacao.py
Síntese:
    Experimento 3 - otimização de hiperparâmetros com Optuna (amostrador TPE) para
    XGBoost, LightGBM e MLP. O objetivo é a AUPRC na validação temporal; o teste
    só é usado depois, uma única vez. O modelo final de cada base é o de maior
    AUPRC na validação; ele é salvo (com limiar e lista de atributos) para os
    experimentos seguintes e para o protótipo.

Uso:
    python src/exp3_otimizacao.py A|B [--refazer]

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

import comum as c
from exp1_modelos import finais as exp1_finais, subamostrar

FAMILIAS = ["XGBoost", "LightGBM", "MLP"]
TENTATIVAS = {"A": {"XGBoost": 30, "LightGBM": 40, "MLP": 30}, "B": {"XGBoost": 60, "LightGBM": 60, "MLP": 40}}
LIMITE_S = {"A": 1200, "B": 600}  # tempo máximo por família


def sugerir(trial, familia):
    if familia == "XGBoost":
        return {"max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "expoente_peso": trial.suggest_float("expoente_peso", 0, 1)}
    if familia == "LightGBM":
        return {"num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "expoente_peso": trial.suggest_float("expoente_peso", 0, 1)}
    camadas = trial.suggest_int("camadas", 1, 3)
    largura = trial.suggest_categorical("largura", [16, 32, 64, 128])
    return {"camadas": camadas, "largura": largura,
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "max_iter": trial.suggest_int("max_iter", 20, 80),
            "razao_subamostragem": trial.suggest_categorical("razao_subamostragem", [10, 20, 50, 100, 200])}


def construir(familia, params, chave, atributos, y_tr):
    """Estimador ainda não treinado a partir dos hiperparâmetros (também usado pelo protótipo)."""
    p = dict(params)
    if familia in ("XGBoost", "LightGBM"):
        spw = c.peso_positivo(y_tr) ** p.pop("expoente_peso")
        return (c.xgb_padrao if familia == "XGBoost" else c.lgbm_padrao)(y_tr, scale_pos_weight=spw, **p)
    camadas = (p["largura"],) * p["camadas"] if p["camadas"] == 1 else \
        tuple(max(8, p["largura"] // (2 ** i)) for i in range(p["camadas"]))
    return make_pipeline(c.preprocessador(atributos, c.BASES[chave]["categoricas"]),
                         MLPClassifier(hidden_layer_sizes=camadas, alpha=p["alpha"],
                                       learning_rate_init=p["learning_rate_init"], batch_size=p["batch_size"],
                                       max_iter=p["max_iter"], random_state=c.SEMENTE))


def dados_treino(familia, params, treino):
    if familia == "MLP":
        return subamostrar(treino, params["razao_subamostragem"])
    return treino


def treinar(familia, params, chave, atributos, treino, val):
    dtr = dados_treino(familia, params, treino)
    m = construir(familia, params, chave, atributos, dtr["is_fraud"].to_numpy())
    c.ajustar_com_parada(m, dtr[atributos], dtr["is_fraud"].to_numpy(), val[atributos], val["is_fraud"].to_numpy())
    return m


def otimizar(chave, familia, atributos, tr, val):
    def objetivo(trial):
        params = sugerir(trial, familia)
        m = treinar(familia, params, chave, atributos, tr, val)
        trial.set_user_attr("n_arvores", c.n_arvores(m))
        return average_precision_score(val["is_fraud"], c.pontuar(m, val[atributos]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=c.SEMENTE),
                                 study_name=f"{chave}-{familia}")
    estudo.optimize(objetivo, n_trials=TENTATIVAS[chave][familia], timeout=LIMITE_S[chave], gc_after_trial=True)
    return estudo


def executar(chave, refazer=False, n_boot=None):
    n_boot = n_boot or (1000 if chave == "B" else 500)  # Base A: 1,7 milhão de linhas no teste
    saida = c.RESULTADOS / f"exp3_metricas_{chave}.csv"
    if saida.exists() and not refazer:
        return pd.read_csv(saida)
    partes, todos = c.carregar(chave)
    atributos = c.atributos_finais(chave, todos)  # conjunto escolhido no Experimento 1
    tr, val, te = partes["treino"], partes["validacao"], partes["teste"]
    y_val, y_te = val["is_fraud"].to_numpy(), te["is_fraud"].to_numpy()
    linhas, preds, finais = [], {}, {}
    for familia in FAMILIAS:
        with c.Cronometro() as t:
            estudo = otimizar(chave, familia, atributos, tr, val)
        estudo.trials_dataframe().to_csv(c.RESULTADOS / f"exp3_tentativas_{chave}_{familia}.csv", index=False)
        params = estudo.best_params
        m = treinar(familia, params, chave, atributos, tr, val)
        p_val, p_te = c.pontuar(m, val[atributos]), c.pontuar(m, te[atributos])
        limiar = c.melhor_limiar(y_val, p_val)
        met = c.metricas(y_te, p_te, limiar)
        val_auprc = average_precision_score(y_val, p_val)
        preds[f"{familia} otimizado"] = (p_te, limiar)
        finais[familia] = (m, params, limiar, val_auprc, met)
        linhas.append({"base": c.BASES[chave]["nome"], "modelo": familia, "tentativas": len(estudo.trials),
                       "tempo_busca_s": round(t.s, 1), "val_pr_auc": val_auprc, "n_arvores": c.n_arvores(m),
                       "params": str(params), **met})
        print(f"[{chave}] {familia:9s} {len(estudo.trials)} tentativas ({t.s:.0f}s) val AUPRC={val_auprc:.3f} | "
              f"teste F1={met['f1']:.3f} AUPRC={met['pr_auc']:.3f}", flush=True)

    # comparação pareada com a configuração padrão do Experimento 1
    met1, pred1 = exp1_finais(chave)
    met1 = met1.set_index("modelo")
    for familia in FAMILIAS:
        preds[f"{familia} padrão"] = (pred1[met1.loc[familia, "id"]].to_numpy(), met1.loc[familia, "limiar"])
    boot = c.bootstrap(y_te, preds, n_boot=n_boot)
    df = pd.DataFrame(linhas)
    for i, r in df.iterrows():
        f = r["modelo"]
        for m in ("f1", "pr_auc"):
            df.loc[i, f"{m}_ic_inf"], df.loc[i, f"{m}_ic_sup"] = c.ic(boot[f"{f} otimizado|{m}"])
            df.loc[i, f"{m}_padrao"] = met1.loc[f, m]
            dif = boot[f"{f} otimizado|{m}"] - boot[f"{f} padrão|{m}"]
            df.loc[i, f"dif_{m}_vs_padrao"] = dif.mean()
            df.loc[i, f"dif_{m}_ic_inf"], df.loc[i, f"dif_{m}_ic_sup"] = c.ic(dif)
    escolhido = max(finais, key=lambda f: finais[f][3])  # maior AUPRC na VALIDAÇÃO
    df["final"] = df["modelo"] == escolhido
    df.to_csv(saida, index=False, float_format="%.6f")

    m, params, limiar, val_auprc, met = finais[escolhido]
    c.MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, c.MODELOS_DIR / f"modelo_final_{chave}.joblib", compress=3)
    c.salvar_json({"base": chave, "nome_base": c.BASES[chave]["nome"], "familia": escolhido, "params": params,
                   "atributos": atributos, "limiar": limiar, "val_pr_auc": val_auprc, "teste": met,
                   "n_arvores": c.n_arvores(m)}, c.MODELOS_DIR / f"modelo_final_{chave}.json")
    pd.DataFrame({"y": y_te.astype(np.int8), "p": c.pontuar(m, te[atributos]).astype(np.float64)}).to_parquet(
        c.pasta_atributos() / f"pred_final_{chave}.parquet", index=False)
    print(f"[{chave}] modelo final: {escolhido}", flush=True)
    return df


def carregar_final(chave):
    meta = c.ler_json(c.MODELOS_DIR / f"modelo_final_{chave}.json")
    return joblib.load(c.MODELOS_DIR / f"modelo_final_{chave}.joblib"), meta


if __name__ == "__main__":
    executar(sys.argv[1], refazer="--refazer" in sys.argv)
