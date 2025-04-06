import pandas as pd
from pandas.api.types import is_numeric_dtype

def gerar_perfis_clusters(
    df_clusterizado,
    colunas_numericas,
    colunas_categoricas_ordenadas,
    colunas_categoricas_nao_ordenadas,
    coluna_alvo="Review Rating",
    salvar_csv=True,
    caminho_csv=r"C:\Users\Camilo_Bica\data_science\portifolio\customer_shopping\dados\perfil_clusters_shopping.csv",
    ordenar_por="Purchase Amount (USD)"
):

    """
    Gera o perfil estratégico de clusters a partir de um DataFrame clusterizado,
    reordenando os clusters com base em uma variável numérica de interesse.

    Para cada cluster, exibe características médias (numéricas) e mais frequentes (categóricas),
    além de sugestões estratégicas com base em padrões de comportamento.
    """

    df_temp = df_clusterizado.copy()
    perfil_numerico = df_temp.groupby("Cluster")[colunas_numericas].mean().round(2)
    ordenados = perfil_numerico[ordenar_por].sort_values(ascending=False)
    mapa_clusters = {antigo: novo for novo, antigo in enumerate(ordenados.index)}
    df_temp["Cluster"] = df_temp["Cluster"].map(mapa_clusters)

    perfil_numerico = df_temp.groupby("Cluster")[colunas_numericas].mean().round(2)
    perfil_categorico = df_temp.groupby("Cluster")[
        colunas_categoricas_nao_ordenadas + colunas_categoricas_ordenadas
    ].agg(lambda x: x.mode().iloc[0])

    perfil_clusters = pd.concat([perfil_numerico, perfil_categorico], axis=1)
    tamanho_cluster = df_temp["Cluster"].value_counts().sort_index()

    print("\n===== PERFIL ESTRATÉGICO DOS CLUSTERS (ordenados) =====\n")

    for cluster_id, row in perfil_clusters.iterrows():
        print(f"\n🔹 Cluster {cluster_id} ({tamanho_cluster[cluster_id]} clientes)")
        print("-" * 40)

        # Exibir variáveis principais
        principais = ["Age", "Purchase Amount (USD)", "Previous Purchases", "Gender"]
        for atributo in principais:
            valor = row.get(atributo, "-")
            print(f"{atributo:>25}: {valor}")

        # Exibir top categorias frequentes
        print("\n📊 Top categorias mais frequentes:")
        df_cluster = df_temp[df_temp["Cluster"] == cluster_id]
        for col in colunas_categoricas_nao_ordenadas + colunas_categoricas_ordenadas:
            if col not in principais:
                top_valores = df_cluster[col].value_counts(normalize=True).head(3)
                top_formatado = ", ".join([f"{idx} ({p*100:.1f}%)" for idx, p in top_valores.items()])
                print(f"- {col}: {top_formatado}")

        print("\n📦 Estratégia Sugerida:")

        idade = row.get("Age")
        gasto = row.get("Purchase Amount (USD)")
        desconto = row.get("Discount Applied")
        frequencia = str(row.get("Frequency of Purchases", "")).lower()

        if idade and idade > 50:
            print("- Cliente maduro: destaque produtos premium ou coleções clássicas.")
        elif idade and idade < 40:
            print("- Cliente jovem: invista em campanhas visuais, tendências e mídias sociais.")

        if gasto and gasto > 70:
            print("- Alto ticket médio: explore upsell e kits exclusivos.")
        elif gasto and gasto < 50:
            print("- Sensível ao preço: campanhas com desconto ou fidelidade.")

        if desconto == 1:
            print("- Responde bem a promoções: destacar ofertas relâmpago ou cupons.")
        else:
            print("- Não usa descontos: foco em valor percebido, frete ou exclusividade.")

        if "annually" in frequencia or "year" in frequencia:
            print("- Pouca frequência: campanhas de reativação e lembretes sazonais.")
        elif "quarter" in frequencia or "every 3" in frequencia:
            print("- Compra regular: mantenha contato com novidades trimestrais.")
        else:
            print("- Frequência indefinida: coletar mais dados e testar cadência.")

        print("=" * 50)

    if salvar_csv:
        perfil_clusters.insert(0, "Tamanho do Cluster", tamanho_cluster)
        perfil_clusters.to_csv(caminho_csv, index_label="Cluster")
        print(f"\n📁 Arquivo exportado com sucesso para:\n{caminho_csv}")

    return df_temp
