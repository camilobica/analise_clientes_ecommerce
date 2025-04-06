import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clusterizar_clientes(   
    df,
    colunas_numericas,
    colunas_categoricas_ordenadas,
    colunas_categoricas_nao_ordenadas,
    n_clusters=4,
    pca_components=2,
    random_state=42,
    plot=True,
    show_silhouette=True
):
    """
    Realiza a clusterização de clientes utilizando KMeans após redução de dimensionalidade com PCA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados dos clientes.
    colunas_numericas : list
        Lista com os nomes das colunas numéricas.
    colunas_categoricas_ordenadas : list
        Lista com os nomes das colunas categóricas ordenadas.
    colunas_categoricas_nao_ordenadas : list
        Lista com os nomes das colunas categóricas nominais.
    n_clusters : int, optional
        Número de clusters desejado para o KMeans (default=4).
    pca_components : int, optional
        Número de componentes principais a serem mantidos no PCA (default=2).
    random_state : int, optional
        Semente aleatória para reprodução de resultados (default=42).
    plot : bool, optional
        Define se o gráfico de dispersão dos clusters será exibido (default=True).
    show_silhouette : bool, optional
        Define se o Silhouette Score será exibido (default=True).

    Returns
    -------
    df_resultado : pd.DataFrame
        DataFrame original com as colunas dos componentes principais e os rótulos de cluster atribuídos.
    pipeline : sklearn.pipeline.Pipeline
        Pipeline treinado contendo pré-processamento, PCA e modelo de clusterização.

    Notas
    -----
    - A clusterização é feita com KMeans e os dados são reduzidos com PCA para visualização.
    - Um gráfico dos clusters e o Silhouette Score são exibidos, se habilitados.
    - Um resumo com a média da nota de review e quantidade de clientes por cluster é mostrado.
    """
    
    # Features (sem target)
    X = df[colunas_numericas + colunas_categoricas_nao_ordenadas + colunas_categoricas_ordenadas]

    # Pré-processamento
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas_nao_ordenadas + colunas_categoricas_ordenadas),
        ("num", StandardScaler(), colunas_numericas)
    ])

    # Pipeline
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("pca", PCA(n_components=pca_components)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state))
    ])

    # Fit pipeline
    pipeline.fit(X)

    # Extrair passos para visualização
    X_pca = pipeline.named_steps["pca"].transform(
        pipeline.named_steps["pre"].transform(X)
    )
    clusters = pipeline.named_steps["kmeans"].labels_

    # Montar DataFrame final
    df_resultado = df.copy()
    df_resultado["Cluster"] = clusters
    for i in range(pca_components):
        df_resultado[f"PCA{i+1}"] = X_pca[:, i]

    # Plot
    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_resultado, x="PCA1", y="PCA2", hue="Cluster", palette="Accent", s=60)
        plt.title(f"Clusters de Clientes via KMeans + PCA (k={n_clusters})", fontsize=13, weight="bold")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.grid(False)
        plt.legend(title="Cluster")
        plt.tight_layout()

        plt.savefig(r"C:\Users\Camilo_Bica\data_science\portifolio\customer_shopping\imagens\cluster_clientes.png", dpi=200, bbox_inches='tight')
        plt.show()

    # Silhouette
    if show_silhouette:
        score = silhouette_score(X_pca, clusters)
        print(f"Silhouette Score: {score:.3f}")

    # Estatísticas por cluster
    resumo = df_resultado.groupby("Cluster").agg(
        Media_Review_Rating=("Review Rating", "mean"),
        Quantidade=("Review Rating", "count")
    )
    display(resumo)

    return df_resultado, pipeline