
"""
App Streamlit – Previsão de Review Rating

Este aplicativo carrega automaticamente o arquivo 'clientes_com_score.csv' e aplica
um modelo de classificação para prever a probabilidade de avaliações altas (> 4.1),
classificando o risco de review por cliente. Exibe data de atualização do CSV e 
visualizações interativas.

Rodar com: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from cycler import cycler

# Configuração global de cores
cores = plt.get_cmap('Accent').colors
plt.rc('axes', prop_cycle=cycler('color', cores))

st.title("📈 Previsão de Reviews - Modelo de Classificação")
st.markdown("Análise automática da probabilidade de review com nota **alta** (> 4.1).")

# Caminhos dos arquivos
CAMINHO_CSV = Path(__file__).resolve().parents[2] / "resultados" / "clientes_com_score.csv"
CAMINHO_MODELO = Path(__file__).resolve().parents[2] / "modelos" / "modelo_logistico_pipeline.pkl"

# Verificação dos arquivos
if not CAMINHO_CSV.exists():
    st.error(f"Arquivo CSV não encontrado em: {CAMINHO_CSV}")
    st.stop()

if not CAMINHO_MODELO.exists():
    st.error(f"Modelo não encontrado em: {CAMINHO_MODELO}")
    st.stop()

# Carregar CSV
df = pd.read_csv(CAMINHO_CSV)
mod_time = time.ctime(CAMINHO_CSV.stat().st_mtime)

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)

pipeline = carregar_modelo()

# Superfeatures
def adicionar_superfeatures(df):
    df["Category_Item_Color"] = df[["Category", "Item Purchased", "Color"]].astype(str).agg("_".join, axis=1)
    df["Category_Item_Size"] = df[["Category", "Item Purchased", "Size"]].astype(str).agg("_".join, axis=1)
    df["Category_Item_Location"] = df[["Category", "Item Purchased", "Location"]].astype(str).agg("_".join, axis=1)
    return df

def classificar_risco(prob):
    if prob >= 70:
        return "Alta probabilidade"
    elif prob >= 50:
        return "Moderada"
    else:
        return "Baixa"

# Exibir info do CSV carregado
st.subheader("1. Dados carregados automaticamente")
st.markdown(f"**Caminho:** `{CAMINHO_CSV}`")
st.markdown(f"**Última atualização:** `{mod_time}`")

# Aplicar modelo
df = adicionar_superfeatures(df)
probs = pipeline.predict_proba(df)
idx_alta = np.where(pipeline.classes_ == "Alta")[0][0]
df["Prob Alta (%)"] = (probs[:, idx_alta] * 100).round(2)
df["Classe Prevista"] = pipeline.predict(df)
df["Risco de Review"] = df["Prob Alta (%)"].apply(classificar_risco)

st.success("✅ Previsões geradas com sucesso!")

# Resultado principal
st.subheader("2. Resultado geral")
st.dataframe(df[["Classe Prevista", "Prob Alta (%)", "Risco de Review"] + df.columns.tolist()[:5]])

# Cliente específico
st.subheader("3. Análise de cliente específico")
idx = st.number_input("Escolha a linha do cliente", min_value=0, max_value=len(df)-1, step=1)
cliente = df.iloc[[idx]]
st.write("### Dados do cliente")
st.dataframe(cliente.T)
st.metric("Probabilidade Alta", f"{cliente['Prob Alta (%)'].values[0]:.2f}%")
st.metric("Classe Prevista", cliente['Classe Prevista'].values[0])
st.metric("Risco", cliente['Risco de Review'].values[0])

# Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📁 Baixar resultado em CSV", csv, "clientes_com_score.csv", "text/csv")

# Visualizações
st.subheader("4. Análises visuais")
riscos = st.multiselect("Filtrar por risco:", options=df["Risco de Review"].unique(), default=df["Risco de Review"].unique())
df_filtro = df[df["Risco de Review"].isin(riscos)]

st.write("#### Distribuição da Probabilidade de Alta")
fig1, ax1 = plt.subplots()
sns.histplot(df_filtro["Prob Alta (%)"], kde=True, bins=20, ax=ax1)
st.pyplot(fig1)

st.write("#### Clientes por Risco")
fig2, ax2 = plt.subplots()
plot = sns.countplot(x="Risco de Review", data=df_filtro, order=["Alta probabilidade", "Moderada", "Baixa"], ax=ax2)
for container in plot.containers:
    plot.bar_label(container, fmt="%d", label_type="edge")
st.pyplot(fig2)

# Top clientes
st.write("#### Top 10 com maior probabilidade de nota alta")
st.dataframe(df_filtro.sort_values("Prob Alta (%)", ascending=False).head(10))

st.write("#### Top 10 com menor probabilidade de nota alta")
st.dataframe(df_filtro.sort_values("Prob Alta (%)", ascending=True).head(10))
