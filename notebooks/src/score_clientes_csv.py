import numpy as np

def pontuar_em_lote(df_clientes, pipeline):
    """
    Recebe um DataFrame com os dados dos clientes e retorna as colunas:
    - Classe Prevista
    - Prob Alta (%) (probabilidade da classe 'Alta')

    Adiciona as superfeatures necessárias antes de prever.
    """
    # Criar superfeatures
    df_clientes["Category_Item_Color"] = df_clientes[["Category", "Item Purchased", "Color"]].astype(str).agg("_".join, axis=1)
    df_clientes["Category_Item_Size"] = df_clientes[["Category", "Item Purchased", "Size"]].astype(str).agg("_".join, axis=1)
    df_clientes["Category_Item_Location"] = df_clientes[["Category", "Item Purchased", "Location"]].astype(str).agg("_".join, axis=1)

    # Prever classe e probabilidade correta da classe 'Alta'
    probs = pipeline.predict_proba(df_clientes)
    idx_alta = np.where(pipeline.classes_ == "Alta")[0][0]

    df_clientes["Classe Prevista"] = pipeline.predict(df_clientes)
    df_clientes["Prob Alta (%)"] = (probs[:, idx_alta] * 100).round(2)

    return df_clientes
