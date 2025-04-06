import joblib
import pandas as pd
import numpy as np

# Caminho do modelo salvo
CAMINHO_MODELO = r"C:\Users\Camilo_Bica\data_science\portifolio\customer_shopping\modelos\modelo_logistico_pipeline.pkl"

# Carregar pipeline
pipeline = joblib.load(CAMINHO_MODELO)

def pontuar_cliente(dados_cliente):
    """
    Recebe um dicionário com os dados de um cliente e retorna:
    - Classe prevista (Alta / Não-Alta)
    - Probabilidade de ser 'Alta'
    """

    # Converter para DataFrame de 1 linha
    df_novo = pd.DataFrame([dados_cliente])

    # Criar superfeatures necessárias
    df_novo["Category_Item_Color"] = df_novo[["Category", "Item Purchased", "Color"]].astype(str).agg("_".join, axis=1)
    df_novo["Category_Item_Size"] = df_novo[["Category", "Item Purchased", "Size"]].astype(str).agg("_".join, axis=1)
    df_novo["Category_Item_Location"] = df_novo[["Category", "Item Purchased", "Location"]].astype(str).agg("_".join, axis=1)

    # Prever classe e probabilidade da classe 'Alta'
    probs = pipeline.predict_proba(df_novo)
    idx_alta = np.where(pipeline.classes_ == "Alta")[0][0]
    prob = probs[0][idx_alta]
    classe = pipeline.predict(df_novo)[0]

    print("📌 Classe Prevista:", classe)
    print(f"📈 Probabilidade de ser 'Alta': {prob:.2%}")

    return classe, prob
