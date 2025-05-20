import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

dados = [
    [0.465137919, 0.9743215312, 0.2441329545],
    [0.8626107218, 0.4040862985, 0.3085435947],
    [0.8450308147, 0.7830696563, 0.2767885267],
    [0.9430230004, 0.3478771849, 0.5576936393]
]

# Criação do DataFrame de exemplo
exemplo = pd.DataFrame(dados, columns=["Coluna 1", "Coluna 2", "Coluna 3"])

# Define o plot do gráfico como wide(ocupando toda a disposição da tela) 
st.set_page_config(layout="wide")

# Títulos
st.title('Ajuste Polinomial')
st.sidebar.header('Data')

# Função principal que recebe as entradas
def parameteres():
    
    # Trecho responsável por orientar e tratar possíveis erros de importação
    data = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if data is None:
        st.warning("⚠️ Por favor, envie um arquivo CSV na barra ao lado para continuar.")
        st.write('Se atente a formatação do arquivo:')
        st.write('  - Só as colunas numéricas ficarão disponíveis')
        st.write('  - Todas as colunas devem possuir label(título)')
        st.write('Exemplo:')
        st.write(exemplo)
        return None, None
    
    # Trecho que lê o arquivo e faz a disposição das colunas para futura escolha
    features = pd.read_csv(data)

    opcoes = features.select_dtypes('number').columns.tolist()

    if len(opcoes) == 0:
        st.warning("⚠️ O arquivo não possui colunas numéricas")
        st.stop()

    opcoes.insert(0, "Selecione...")

    y = st.sidebar.selectbox("Alvo:", opcoes)
    x = st.sidebar.selectbox("Variável preditora:", opcoes)

    # Foi necessário incluir uma opção neutra devido ao bug do streamlit com o Plotly, que tava o cursos do mouse no zoom.
    if y == "Selecione...":
        st.warning("⚠️ Por favor, selecione uma variável alvo.")
        st.stop()

    if x == "Selecione...":
        st.warning("⚠️ Por favor, selecione uma variável preditora.")
        st.stop()

    split = st.sidebar.slider('Proporção para treinamento', 0.1, 0.9)

    degree = st.sidebar.slider('Grau de Ajuste do Polinômio', 1, 20)

    report_data = {
        'x':x,
        'y':y,
        'degree':degree,
        'split':split
    }

    return report_data, features

# Traz os dados inseridos
user_report, user_data = parameteres()

# Tratativa feita, também no intuito de contornar o bug entre o streamlit e o plotly
if user_report is not None:
    import matplotlib.pyplot as plt

    x = user_data[user_report['x']].values.ravel()
    y = user_data[user_report['y']].values.ravel()

    # Ajuste do polinômio, garantindo que não haja data leakage da curva na base de teste
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=user_report['split'], random_state=42, shuffle=False)

    ajuste1 = np.polyfit(x_train, y_train, user_report['degree'])
    polinomio1 = np.poly1d(ajuste1)

    y_pred = polinomio1(x)

    ref1 = np.sort(x)

    import plotly.express as px
    import plotly.graph_objects as go

    indice_corte = int(len(ref1) * (user_report['split']))
    corte_x = ref1[indice_corte]


    # Plotagem do gráfico interativo
    fig = px.scatter(x=x, y=y)
    fig.add_trace(go.Scatter(x=ref1, y=polinomio1(ref1), mode='lines',line=dict(color='white', width=3), name='Curva Polinomial'))
  

    fig.update_layout(
        title='Gráfico de Dispersão com Ajuste Polinomial',
        xaxis_title='Variável',
        yaxis_title='Alvo',
        font=dict(size=18),
        width=900,
        height=650,
        showlegend=True,
        shapes=[dict(
            type='line',
            x0=corte_x,
            x1=corte_x,
            y0=0,
            y1=1,
            xref='x',
            yref='paper',
            line_color='red',
            line_width=2,
            line_dash='dash'
        )]
    )

    st.plotly_chart(fig)