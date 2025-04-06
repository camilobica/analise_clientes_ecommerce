import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

def diagnostico_superfeature(df, nome_feature, y):
    """
    Analisa uma superfeature categórica por meio de regressão logística e exibe os coeficientes e suas frequências.

    A função aplica um pipeline de pré-processamento com OneHotEncoder seguido de regressão logística para avaliar a importância
    de cada categoria da superfeature na predição da variável alvo. Exibe os 15 maiores impactos positivos e negativos separadamente.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contendo a superfeature e os dados utilizados para treinamento.
    
    nome_feature : str
        Nome da coluna no DataFrame representando a superfeature (coluna categórica combinada).
    
    y : array-like or pandas.Series
        Variável alvo binária a ser prevista pelo modelo.

    Returns
    -------
    pandas.DataFrame
        Um DataFrame contendo os perfis categóricos com os seguintes campos:
        - Perfil : str, nome da categoria (sem prefixo)
        - Coef : float, coeficiente do modelo logístico
        - Freq : int, frequência da categoria no DataFrame
        - AbsCoef : float, valor absoluto do coeficiente
    """
    print(f"\n🔎 Analisando: {nome_feature}")
    
    X = df[[nome_feature]]
    
    # Pipeline simples: OneHot + Regressão
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), [nome_feature])
    ])

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipe.fit(X, y)
    
    # Extração dos coeficientes
    modelo = pipe.named_steps["modelo"]
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    
    features = ohe.get_feature_names_out([nome_feature])
    coefs = modelo.coef_[0]
    
    df_coef = pd.DataFrame({
        "Perfil": [f.replace(f"{nome_feature}_", "") for f in features],
        "Coef": coefs
    })

    # Frequência de cada perfil
    freq = df[nome_feature].value_counts().reset_index()
    freq.columns = ["Perfil", "Freq"]
    
    df_final = df_coef.merge(freq, on="Perfil", how="left")
    df_final["AbsCoef"] = df_final["Coef"].abs()

    # 🔝 Top 15 positivos
    df_positivos = df_final[df_final["Coef"] > 0].sort_values("Coef", ascending=False).head(15)
    print("\n✅ Top 15 - Perfis com maior impacto positivo:")
    display(df_positivos)

    # 🔻 Top 15 negativos
    df_negativos = df_final[df_final["Coef"] < 0].sort_values("Coef").head(15)
    print("\n🔻 Top 15 - Perfis com maior impacto negativo:")
    display(df_negativos)

    return df_final

