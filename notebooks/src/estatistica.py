import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tabela_distribuicao_frequencias(dataframe, coluna, coluna_frequencia=False):
    """Cria uma tabela de distribuição de frequências para uma coluna de um dataframe.
    Espera uma coluna categórica.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe com os dados.
    coluna : str
        Nome da coluna categórica.
    coluna_frequencia : bool
        Informa se a coluna passada já é com os valores de frequência ou não. Padrão: False

    Returns
    -------
    pd.DataFrame
        Dataframe com a tabela de distribuição de frequências.
    """ 
    
    df_estatistica = pd.DataFrame()

    if coluna_frequencia:
        df_estatistica["frequencia"] = dataframe[coluna]  # Corrigido 'dataframse' para 'dataframe'
        df_estatistica["frequencia_relativa"] = df_estatistica["frequencia"] / df_estatistica["frequencia"].sum()
    else:
        df_estatistica["frequencia"] = dataframe[coluna].value_counts().sort_index()
        df_estatistica["frequencia_relativa"] = dataframe[coluna].value_counts(normalize=True).sort_index()
    
    df_estatistica["frequencia_acumulada"] = df_estatistica["frequencia"].cumsum()
    df_estatistica["frequencia_relativa_acumulada"] = df_estatistica["frequencia_relativa"].cumsum()
    
    return df_estatistica


def composicao_histograma_boxplot(dataframe, coluna, intervalos="auto"):
    """
    Cria uma composição gráfica com boxplot e histograma da distribuição de uma variável numérica.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str
        Nome da coluna numérica a ser visualizada.
    intervalos : int, str, or sequence, optional
        Número ou definição dos intervalos do histograma. Padrão é "auto".

    Returns
    -------
    None
        A função exibe os gráficos, mas não retorna nenhum valor.
    """
    
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={
            "height_ratios": (0.15, 0.85),
            "hspace": 0.02
        }
    )
    
    sns.boxplot(
        data=dataframe,
        x=coluna,
        showmeans=True,
        meanline=True,
        meanprops={"color": "C1", "linewidth": 1.5, "linestyle": "--"},
        medianprops={"color": "C2", "linewidth": 1.5, "linestyle": "--"},
        ax=ax1
    )
    
    sns.histplot(data=dataframe, x=coluna, kde=True, bins="sturges", ax=ax2)
    
    ax1.grid(False)
    ax1.tick_params(left=False, bottom=False)
    
    ax2.axvline(dataframe[coluna].mean(), color="C1", linestyle="--", label="Média")
    ax2.axvline(dataframe[coluna].median(), color="C2", linestyle="--", label="Mediana")
    ax2.axvline(dataframe[coluna].mode()[0], color="C4", linestyle="--", label="Moda")
    ax2.grid(False)
    ax2.set_ylabel("")
    ax2.legend()

    plt.savefig(r"C:\Users\Camilo_Bica\data_science\portifolio\customer_shopping\imagens\boxplot_histograma.png", dpi=200, bbox_inches='tight')
    
    plt.show()
