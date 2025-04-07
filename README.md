# Análise de Clientes – E-commerce

Este projeto analisa o comportamento de clientes de um e-commerce com foco na **avaliação dos produtos (Review Rating)**. O objetivo central é identificar padrões de compra, segmentar perfis e prever a probabilidade de um cliente avaliar com **nota alta (> 4.1)**.

A análise combina estatística descritiva, criação de superfeatures, machine learning e clusterização para gerar insights acionáveis para estratégias de marketing e fidelização.

![imagem ilustrativa](imagens/ecommerce_imagem.jpg)

## Etapas do Projeto

1. **Importação e Pré-processamento**: Carregamento do dataset bruto e remoção de colunas irrelevantes.
2. **Tratamento de Dados**: Criação de variáveis derivadas e conversão de tipos para análise.
3. **Análise Exploratória (EDA)**: Visualizações, boxplots, histogramas e distribuição por categorias.
4. **Estatística Descritiva e Testes**: ANOVA, Qui-quadrado e análise de correlação.
5. **Criação de Superfeatures**: Combinação de variáveis para aumentar a explicabilidade.
6. **Clusterização (KMeans)**: Agrupamento de clientes por comportamento de compra.
7. **Modelagem Preditiva**: Pipeline com pré-processamento, SMOTE e regressão logística.
8. **Aplicativo em Streamlit**: App interativo para previsão automática da nota de avaliação.

## Ferramentas Utilizadas

- **Pandas e Numpy** – Manipulação e tratamento dos dados.
- **Matplotlib e Seaborn** – Visualização de dados.
- **Scipy** – Testes estatísticos (ANOVA, Qui-quadrado).
- **Scikit-learn** – Modelos de machine learning e pipeline.
- **Imbalanced-learn (SMOTE)** – Balanceamento da variável alvo.
- **Joblib** – Serialização do modelo final.
- **Streamlit** – Aplicação web para previsão em tempo real.
- **Cycler** – Personalização da paleta de cores nos gráficos.

## Organização do Projeto

```
├── LICENSE
├── README.md                <- Este arquivo.
│
├── dados/                   <- Dados originais ou bases intermediárias.
├── resultados/              <- Dados tratados e com previsões geradas.
├── modelos/                 <- Modelos treinados salvos (.pkl).
├── imagens/                 <- Ilustrações e gráficos para o projeto.
├── notebooks/               <- Jupyter Notebooks do projeto.
│   └── src/                 <- Scripts Python para apoio analítico e operacional.
│       ├── app.py                        <- Aplicativo Streamlit para pontuação automática.
│       ├── avaliacao_grupo.py           <- Avaliação de grupos de variáveis usando regressão logística.
│       ├── clusters.py                  <- Clusterização de clientes com PCA + KMeans.
│       ├── clusters_perfis.py           <- Geração de perfis estratégicos para clusters identificados.
│       ├── estatistica.py               <- Funções estatísticas: tabelas de frequência, boxplots, histogramas.
│       ├── score_clientes.py            <- Função para pontuação individual de clientes com modelo salvo.
│       ├── score_clientes_csv.py        <- Função para pontuação em lote de clientes via DataFrame.
│       ├── superfeature.py              <- Criação e avaliação de superfeatures com análise de coeficientes.
│       └── superfeature_diagnostico.py  <- Diagnóstico detalhado dos impactos das superfeatures criadas.
├── referenciais/            <- Dicionário de dados e documentos auxiliares.
```

## Configuração do Ambiente

1. Clone o repositório:

```bash
git clone git@github.com:camilobica/analise-clientes-ecommerce.git
```

2. Crie o ambiente:

```bash
conda env create -f ambiente_ecommerce.yml --name ecommerce
conda activate ecommerce
```

## 📑 Dicionário de Dados

[Clique aqui](referenciais/dicionario_de_dados.md) para acessar o dicionário completo da base utilizada.

## Principais Descobertas

### 🎯 Variáveis Categóricas com maior impacto

- Tipo de envio, estação e categoria do item estão entre os fatores com maior influência na nota de avaliação.
- Variáveis numéricas, como idade ou valor da compra, possuem baixo poder explicativo isoladamente.

### 🔧 Superfeatures e Diagnósticos

- A combinação de variáveis categóricas (ex: `Category_Item_Location`) melhora a performance do modelo.
- Mesmo com superfeatures, a previsão da nota depende de **fatores subjetivos não presentes no dataset**, como atendimento e expectativa do cliente.

### 🔍 Clusterização

- Clientes foram agrupados por comportamento, possibilitando estratégias segmentadas:
  - **Cluster 0**: Jovens compradores frequentes → recomendação personalizada.
  - **Cluster 1**: Alta compra, baixa recorrência → ofertas de fidelização.
  - **Cluster 2**: Exigentes nas avaliações → foco em experiência e suporte.

### 🤖 Modelo de Classificação

- Pipeline com balanceamento (SMOTE) e regressão logística.
- Classificação em três níveis de risco: **Alta, Moderada, Baixa probabilidade** de nota alta.
- Aplicável diretamente via app `Streamlit`.

---

## Conclusão

O projeto oferece um sistema completo de análise e previsão de avaliações em e-commerce, com aplicabilidade real em marketing personalizado, CRM e estratégias de retenção.

✅ Clusterização com perfis claros de comportamento  
✅ App funcional para uso direto por times de negócios  
✅ Base estruturada para evolução com novos dados qualitativos
