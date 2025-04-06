from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np


def avaliar_superfeature(df, cols, nome, colunas_numericas):
    """
    Avalia o impacto de uma superfeature combinada (feature composta por múltiplas colunas) 
    na predição de uma variável binária utilizando regressão logística.

    A função cria uma nova feature categórica a partir da combinação de colunas especificadas, 
    calcula a frequência dos perfis, treina um modelo de regressão logística com OneHotEncoder 
    e exibe a métrica AUC. Também retorna os coeficientes do modelo para análise de impacto.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contendo os dados dos clientes, incluindo as colunas a serem combinadas e variáveis numéricas.

    cols : list of str
        Lista com os nomes das colunas que serão combinadas para formar a superfeature categórica.

    nome : str
        Nome identificador para a superfeature (usado para nomear a nova coluna).

    colunas_numericas : list of str
        Lista com os nomes das colunas numéricas que serão mantidas no modelo como variáveis contínuas.

    Returns
    -------
    pandas.DataFrame
        DataFrame com os coeficientes da regressão logística contendo:
        - Perfil : str, nome da categoria (codificada ou variável numérica)
        - Coef : float, coeficiente associado à variável no modelo

    Notes
    -----
    - A função assume que a coluna de destino (target) se chama "Review Binary" 
      e que os valores positivos são representados pela string "Alta".
    - Retorna um DataFrame vazio em caso de incompatibilidade entre o número de features e coeficientes.
    """
    print(f"\n🔎 Avaliando: {nome}")

    # Criar super coluna
    col_nome = f"Super_{nome.replace(' ', '_')}"
    df[col_nome] = df[cols].astype(str).agg("_".join, axis=1)

    # Frequência dos perfis
    freq = df[col_nome].value_counts()
    print(f"➡️ Perfis únicos: {freq.shape[0]}")
    print(f"📊 Frequência:\n{freq.value_counts().head()}")

    # Treinar modelo simples
    X = df[[col_nome] + colunas_numericas]
    y = df["Review Binary"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), [col_nome])
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipeline.fit(X, y)
    y_prob = pipeline.predict_proba(X)[:, 1]
    auc = roc_auc_score((y == "Alta").astype(int), y_prob)
    print(f"🎯 AUC: {auc:.3f}")

    # Coeficientes
    modelo = pipeline.named_steps["modelo"]
    ohe = pipeline.named_steps["pre"].named_transformers_["cat"]
    features_cat = ohe.get_feature_names_out([col_nome])  # ← essa linha corrige o problema
    features_total = features_cat.tolist() + colunas_numericas

    coef = modelo.coef_[0]

    if len(coef) != len(features_total):
        print(f"🚨 Tamanho incompatível! Features: {len(features_total)}, Coef: {len(coef)}")
        return pd.DataFrame()  # retorna vazio para evitar erro

    df_coef = pd.DataFrame({
        "Perfil": features_total,
        "Coef": coef
    }).sort_values("Coef", ascending=False)

    return df_coef