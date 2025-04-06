from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score

def avaliar_grupo(nome_grupo, colunas, df, y):
    """
    Treina e avalia um modelo de regressão logística para um grupo de variáveis categóricas.

    Parameters
    ----------
    nome_grupo : str
        Nome do grupo de variáveis (ex: "Cliente", "Produto").
    colunas : list of str
        Lista com os nomes das colunas a serem utilizadas como preditoras.
    df : pd.DataFrame
        DataFrame com os dados originais.
    y : pd.Series
        Variável alvo binária ("Review Binary").

    Returns
    -------
    dict
        Dicionário contendo métricas de desempenho: AUC, F1-score da classe 'Alta' e Acurácia.
    """
    
    X = df[colunas]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), colunas)
    ])

    pipe = Pipeline([
        ("preprocessamento", preprocessor),
        ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score((y_test == "Alta").astype(int), y_prob)
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    f1 = relatorio["Alta"]["f1-score"]
    acc = relatorio["accuracy"]

    return {
        "Grupo": nome_grupo,
        "Variáveis": colunas,
        "AUC": auc,
        "F1_Alta": f1,
        "Accuracy": acc
    }
