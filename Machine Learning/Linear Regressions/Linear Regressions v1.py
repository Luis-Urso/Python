# MBA DATA SCIENCE & ANALYTICS USP/Esalq
# SUPERVISED MACHINE LEARNING: ANÁLISE DE REGRESSÃO SIMPLES E MÚLTIPLA
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários
    
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from statsmodels.iolib.summary2 import summary_col
from skimage import io
import plotly.graph_objs as go
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder


# In[ ]:
#############################################################################
#                          REGRESSÃO LINEAR SIMPLES                         #
#                  EXEMPLO 01 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################
    
df = pd.read_csv("tempodist.csv", delimiter=',')
df

#Características das variáveis do dataset
df.info()

#Estatísticas univariadas
df.describe()


# In[ ]: Gráfico de dispersão

#Regressão linear que melhor se adequa às obeservações: função 'sns.regplot'

plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=False, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Estimação do modelo de regressão linear simples

#Estimação do modelo
modelo = sm.OLS.from_formula("tempo ~ distancia", df).fit()

#Observação dos parâmetros resultantes da estimação
modelo.summary()


# In[ ]: Salvando fitted values (variável yhat) 
# e residuals (variável erro) no dataset

df['yhat'] = modelo.fittedvalues
df['erro'] = modelo.resid
df


# In[ ]: Gráfico didático para visualizar o conceito de R²

y = df['tempo']
yhat = df['yhat']
x = df['distancia']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot([x[i],x[i]], [yhat[i],y[i]],'--', color='#2ecc71')
    plt.plot([x[i],x[i]], [yhat[i],mean[i]], ':', color='#9b59b6')
    plt.plot(x, y, 'o', color='#2c3e50')
    plt.axhline(y = y.mean(), color = '#bdc3c7', linestyle = '-')
    plt.plot(x,yhat, color='#34495e')
    plt.title('R2: ' + str(round(modelo.rsquared,4)))
    plt.xlabel("Distância")
    plt.ylabel("Tempo")
plt.show()


# In[ ]: Cálculo manual do R²

R2 = ((df['yhat']-
       df['tempo'].mean())**2).sum()/(((df['yhat']-
                                        df['tempo'].mean())**2).sum()+
                                        (df['erro']**2).sum())

round(R2,4)


# In[ ]: Coeficiente de ajuste (R²) é a correlação ao quadrado

#Correlação de Pearson
df[['tempo','distancia']].corr()

#R²
(df[['tempo','distancia']].corr())**2


# In[ ]: Modelo auxiliar para mostrar R² igual a 100% (para fins didáticos)

#Estimação do modelo com yhat como variável dependente,
#resultará em uma modelo com R² igual a 100%
modelo_auxiliar = sm.OLS.from_formula("yhat ~ distancia", df).fit()

#Parâmetros resultantes da estimação
modelo_auxiliar.summary()


# In[ ]:Gráfico mostrando o perfect fit

plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='yhat', ci=False, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Voltando ao nosso modelo original

#Plotando o intervalo de confiança de 90%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=90, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#Plotando o intervalo de confiança de 95%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=95, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#Plotando o intervalo de confiança de 99%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=99, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#Plotando o intervalo de confiança de 99,999%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=99.999, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Calculando os intervalos de confiança

#Nível de significância de 10% / Nível de confiança de 90%
modelo.conf_int(alpha=0.1)

#Nível de significância de 5% / Nível de confiança de 95%
modelo.conf_int(alpha=0.05)

#Nível de significância de 1% / Nível de confiança de 99%
modelo.conf_int(alpha=0.01)

#Nível de significância de 0,001% / Nível de confiança de 99,999%
modelo.conf_int(alpha=0.00001)


# In[ ]: Fazendo predições em modelos OLS
#Ex.: Qual seria o tempo gasto, em média, para percorrer a distância de 25km?

modelo.predict(pd.DataFrame({'distancia':[25]}))

#Cálculo manual - mesmo valor encontrado
5.8784 + 1.4189*(25)


# In[ ]: Nova modelagem para o mesmo exemplo, com novo dataset que
#contém replicações

#Quantas replicações de cada linha você quer? -> função 'np.repeat'
df_replicado = pd.DataFrame(np.repeat(df.values, 3, axis=0))
df_replicado.columns = df.columns
df_replicado


# In[ ]: Estimação do modelo com valores replicados

modelo_replicado = sm.OLS.from_formula("tempo ~ distancia",
                                       df_replicado).fit()

#Parâmetros do modelo
modelo_replicado.summary()


# In[ ]: Calculando os novos intervalos de confiança

#Nível de significância de 5% / Nível de confiança de 95%
modelo_replicado.conf_int(alpha=0.05)


# In[ ]: Plotando o novo gráfico com intervalo de confiança de 95%
#Note o estreitamento da amplitude dos intervalos de confiança!

plt.figure(figsize=(20,10))
sns.regplot(data=df_replicado, x='distancia', y='tempo', ci=95, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: PROCEDIMENTO ERRADO: ELIMINAR O INTERCEPTO QUANDO ESTE NÃO SE MOSTRAR
#ESTATISTICAMENTE SIGNIFICANTE

modelo_errado = sm.OLS.from_formula("tempo ~ 0 + distancia", df).fit()

#Parâmetros do modelo
modelo_errado.summary()


# In[ ]: Comparando os parâmetros do modelo inicial (objeto 'modelo')
#com o 'modelo_errado'

summary_col([modelo, modelo_errado])

#Outro modo mais completo também pela função 'summary_col'
summary_col([modelo, modelo_errado],
            model_names=["MODELO INICIAL","MODELO ERRADO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })


# In[ ]: Gráfico didático para visualizar o viés decorrente de se eliminar
# erroneamente o intercepto em modelos regressivos

x = df['distancia']
y = df['tempo']

yhat = df['yhat']
yhat_errado = modelo_errado.fittedvalues

plt.plot(x, y, 'o', color='dimgray')
plt.plot(x, yhat, color='limegreen')
plt.plot(x, yhat_errado, color='red')
plt.xlabel("Distância")
plt.ylabel("Tempo")
plt.legend(['Observados','Fitted Values OLS','Sem Intercepto'])
plt.show()


# In[ ]: Inserção de figuras no gráfico -> comandar Shift + Enter

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

f = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRw3cY0sCZKUu7KGTNpf5NMURoLdy1UHpxS-w&usqp=CAU"
bom = io.imread(f)

f = "https://icon-library.com/images/error-image-icon/error-image-icon-23.jpg"
ruim = io.imread(f)

fig, ax = plt.subplots()

x = df['distancia']
y = df['tempo']

yhat = df['yhat']
yhat_errado = modelo_errado.fittedvalues

img_box = OffsetImage(ruim, zoom=0.05)
xy=[25,20]

img_bom = AnnotationBbox(img_box, xy, xybox=(10,-10), boxcoords='offset points')

img_box = OffsetImage(bom, zoom=0.15)
xy=[10,40]

img_ruim = AnnotationBbox(img_box, xy, xybox=(10,-10), boxcoords='offset points')

ax.set_xlabel("Distância")
ax.set_ylabel("Tempo")

ax.plot(x, y, 'o', color='dimgray')
ax.plot(x, yhat, color='limegreen')
ax.plot(x, yhat_errado, color='red')

ax.legend(['Observado','Fitted values OLS', 'Sem Intercepto'])
ax.add_artist(img_bom)
ax.add_artist(img_ruim)
plt.show()


# In[ ]:
#############################################################################
#                         REGRESSÃO LINEAR MÚLTIPLA                         #
#                EXEMPLO 02 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_paises = pd.read_csv('paises.csv', delimiter=',', encoding="utf-8")
df_paises

#Características das variáveis do dataset
df_paises.info()

#Estatísticas univariadas
df_paises.describe()


# In[ ]: Gráfico 3D com scatter

import plotly.io as pio
pio.renderers.default = 'browser'

trace = go.Scatter3d(
    x=df_paises['horas'], 
    y=df_paises['idade'], 
    z=df_paises['cpi'], 
    mode='markers',
    marker={
        'size': 5,
        'opacity': 0.8,
    },
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)
plot_figure.update_layout(scene = dict(
                        xaxis_title='horas',
                        yaxis_title='idade',
                        zaxis_title='cpi'))
plot_figure.show()


# In[ ]: Matriz de correlações

corr = df_paises.corr()
corr

plt.figure(figsize=(15,10))
sns.heatmap(df_paises.corr(), annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':22})
plt.show()

#Palettes de cores
#sns.color_palette("viridis", as_cmap=True)
#sns.color_palette("magma", as_cmap=True)
#sns.color_palette("inferno", as_cmap=True)
#sns.color_palette("Blues", as_cmap=True)
#sns.color_palette("Greens", as_cmap=True)
#sns.color_palette("Reds", as_cmap=True)


# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_paises, diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Estimando um modelo múltiplo com as variáveis do dataset 'paises'

#Estimando a regressão múltipla
modelo_paises = sm.OLS.from_formula("cpi ~ idade + horas", df_paises).fit()

#Parâmetros do modelo
modelo_paises.summary()

#Parâmetros do modelo com intervalos de confiança
#Nível de significância de 5% / Nível de confiança de 95%
modelo_paises.conf_int(alpha=0.05)


# In[ ]: Salvando os fitted values na base de dados

df_paises['cpifit'] = modelo_paises.fittedvalues
df_paises


# In[ ]: Gráfico 3D com scatter e fitted values resultantes do modelo

trace = go.Scatter3d(
    x=df_paises['horas'], 
    y=df_paises['idade'], 
    z=df_paises['cpi'], 
    mode='markers',
    marker={
        'size': 5,
        'opacity': 0.8,
    },
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
    xaxis_title='X AXIS TITLE',
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)
plot_figure.add_trace(go.Mesh3d(
                    x=df_paises['horas'], 
                    y=df_paises['idade'], 
                    z=df_paises['cpifit'], 
                    opacity=0.5,
                    color='pink'
                  ))
plot_figure.update_layout(scene = dict(
                        xaxis_title='horas',
                        yaxis_title='idade',
                        zaxis_title='cpi'))
plot_figure.show()


# In[ ]:
#############################################################################
#         REGRESSÃO COM UMA VARIÁVEL EXPLICATIVA (X) QUALITATIVA            #
#             EXEMPLO 03 - CARREGAMENTO DA BASE DE DADOS                    #
#############################################################################

df_corrupcao = pd.read_csv('corrupcao.csv',delimiter=',',encoding='utf-8')
df_corrupcao

#Características das variáveis do dataset
df_corrupcao.info()

#Estatísticas univariadas
df_corrupcao.describe()

# Estatísticas univariadas por região
df_corrupcao.groupby('regiao').describe()

#Tabela de frequências da variável 'regiao'
#Função 'value_counts' do pacote 'pandas' sem e com o argumento 'normalize'
#para gerar, respectivamente, as contagens e os percentuais
contagem = df_corrupcao['regiao'].value_counts(dropna=False)
percent = df_corrupcao['regiao'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=False)


# In[ ]: Conversão dos dados de 'regiao' para dados numéricos, a fim de
#se mostrar a estimação de modelo com o problema da ponderação arbitrária

label_encoder = LabelEncoder()
df_corrupcao['regiao_numerico'] = label_encoder.fit_transform(df_corrupcao['regiao'])
df_corrupcao['regiao_numerico'] = df_corrupcao['regiao_numerico'] + 1
df_corrupcao.head(10)

#A nova variável 'regiao_numerico' é quantitativa (ERRO!), fato que
#caracteriza a ponderação arbitrária!
df_corrupcao['regiao_numerico'].info()
df_corrupcao.describe()


# In[ ]: Modelando com a variável preditora numérica, resultando na
#estimação ERRADA dos parâmetros
#PONDERAÇÃO ARBITRÁRIA!
modelo_corrupcao_errado = sm.OLS.from_formula("cpi ~ regiao_numerico",
                                              df_corrupcao).fit()

#Parâmetros do modelo
modelo_corrupcao_errado.summary()

#Calculando os intervalos de confiança com nível de significância de 5%
modelo_corrupcao_errado.conf_int(alpha=0.05)


# In[ ]: Plotando os fitted values do modelo_corrupcao_errado considerando,
#PROPOSITALMENTE, a ponderação arbitrária, ou seja, assumindo que as regiões
#representam valores numéricos (América do Sul = 1; Ásia = 2; EUA e Canadá = 3;
#Europa = 4; Oceania = 5).

ax =sns.lmplot(
    data=df_corrupcao,
    x="regiao_numerico", y="cpi",
    height=10
)
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']) + " " +
                str(point['y']))
plt.title('Resultado da ponderação arbitrária')
plt.xlabel('Região')
plt.ylabel('Corruption Perception Index')
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca()) 


# In[ ]: Dummizando a variável 'regiao'. O código abaixo automaticamente fará: 
# a)o estabelecimento de dummies que representarão cada uma das regiões do dataset; 
# b)removerá a variável original a partir da qual houve a dummização; 
# c)estabelecerá como categoria de referência a primeira categoria, ou seja,
# a categoria 'America_do_sul' por meio do argumento 'drop_first=True'.

df_corrupcao_dummies = pd.get_dummies(df_corrupcao, columns=['regiao'],
                                      drop_first=True)

df_corrupcao_dummies.head(10)

#A variável 'regiao' está inicialmente definida como 'object' no dataset
df_corrupcao.info()
#O procedimento atual também poderia ter sido realizado em uma variável
#dos tipos 'category' ou 'string'. Para fins de exemplo, podemos transformar a
#variável 'regiao' para 'category' ou 'string' e comandar o código anterior:
df_corrupcao['regiao'] = df_corrupcao['regiao'].astype("category")
df_corrupcao.info()
df_corrupcao['regiao'] = df_corrupcao['regiao'].astype("string")
df_corrupcao.info()


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

modelo_corrupcao_dummies = sm.OLS.from_formula("cpi ~ regiao_Asia + \
                                              regiao_EUA_e_Canada + \
                                              regiao_Europa + \
                                              regiao_Oceania",
                                              df_corrupcao_dummies).fit()

#Parâmetros do modelo
modelo_corrupcao_dummies.summary()

#Outro método (sugestão de uso para muitas dummies no dataset)
# Definição da fórmula utilizada no modelo
lista_colunas = list(df_corrupcao_dummies.drop(columns=['cpi','pais','regiao_numerico']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "cpi ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_corrupcao_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_corrupcao_dummies).fit()

#Parâmetros do modelo
modelo_corrupcao_dummies.summary()


# In[ ]: Plotando o modelo_corrupcao_dummies de forma interpolada

#Fitted values do 'modelo_corrupcao_dummies' no dataset 'df_corrupcao_dummies'
df_corrupcao_dummies['fitted'] = modelo_corrupcao_dummies.fittedvalues
df_corrupcao_dummies.head()

#Gráfico
from scipy import interpolate

plt.figure(figsize=(10,10))

df2 = df_corrupcao_dummies[['regiao_numerico','fitted']].groupby(['regiao_numerico']).median().reset_index()
x = df2['regiao_numerico']
y = df2['fitted']

tck = interpolate.splrep(x, y, k=2)
xnew = np.arange(1,5,0.1) 
ynew = interpolate.splev(xnew, tck, der=0)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']) + " " + str(point['y']))

plt.scatter(df_corrupcao_dummies['regiao_numerico'], df_corrupcao_dummies['cpi'])
plt.scatter(df_corrupcao_dummies['regiao_numerico'], df_corrupcao_dummies['fitted'])
plt.plot(xnew, ynew)
plt.title('Resultado da ponderação arbitrária')
plt.xlabel('Região')
plt.ylabel('Corruption Perception Index')
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca())


# In[ ]:
#############################################################################
#            REGRESSÃO NÃO LINEAR E TRANSFORMAÇÃO DE BOX-COX                #
#              EXEMPLO 04 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_bebes = pd.read_csv('bebes.csv', delimiter=',')
df_bebes

#Características das variáveis do dataset
df_bebes.info()

#Estatísticas univariadas
df_bebes.describe()


# In[ ]: Gráfico de dispersão

plt.figure(figsize=(10,10))
plt.scatter(df_bebes['idade'],df_bebes['comprimento'])
plt.title('Dispersão dos dados', fontsize=17)
plt.xlabel('Idade em semanas', fontsize=16)
plt.ylabel('Comprimento em cm', fontsize=16)
plt.show()


# In[ ]: Estimação de um modelo OLS linear
modelo_linear = sm.OLS.from_formula('comprimento ~ idade', df_bebes).fit()

#Observar os parâmetros resultantes da estimação
modelo_linear.summary()


# In[ ]: Gráfico de dispersão com ajustes (fits) linear e não linear

plt.figure(figsize=(10,10))
sns.regplot(x="idade", y="comprimento", data=df_bebes,
            x_estimator=np.mean, logx=True, color='#ed7d31')
plt.plot(df_bebes['idade'],modelo_linear.fittedvalues, color='#2e1547')
plt.title('Dispersão dos dados', fontsize=17)
plt.xlabel('Idade em semanas', fontsize=16)
plt.ylabel('Comprimento em cm', fontsize=16)
plt.show()


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro
#shapiro(modelo_linear.resid)

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiroFrancia' do pacote
#'sfrancia'
# Autores: Luiz Paulo Fávero e Helder Prado Santos
#pip install sfrancia==1.0.8
from sfrancia import shapiroFrancia
shapiroFrancia(modelo_linear.resid)


# In[ ]: Histograma dos resíduos do modelo OLS linear

plt.figure(figsize=(10,10))
sns.histplot(data=modelo_linear.resid, kde=True, bins=30)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#x é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
x, lmbda = boxcox(df_bebes['comprimento'])

#Inserindo a variável transformada ('bc_comprimento') no dataset
#para a estimação de um novo modelo
df_bebes['bc_comprimento'] = x

df_bebes

#Apenas para fins de comparação e comprovação do cálculo de x
df_bebes['bc_comprimento2'] = ((df_bebes['comprimento']**lmbda)-1)/lmbda

df_bebes

del(df_bebes['bc_comprimento2'])


# In[ ]: Estimando um novo modelo OLS com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_comprimento ~ idade', df_bebes).fit()

#Parâmetros do modelo
modelo_bc.summary()


# In[ ]: Comparando os parâmetros do 'modelo_linear' com os do 'modelo_bc'
#CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!

summary_col([modelo_linear, modelo_bc])

#Outro modo mais completo também pela função 'summary_col'
summary_col([modelo_linear, modelo_bc],
            model_names=["MODELO LINEAR","MODELO BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

#Repare que há um salto na qualidade do ajuste para o modelo não linear (R²)

pd.DataFrame({'R² OLS':[round(modelo_linear.rsquared,4)],
              'R² Box-Cox':[round(modelo_bc.rsquared,4)]})


# In[ ]: Verificando a normalidade dos resíduos do 'modelo_bc'

# Teste de Shapiro-Francia
shapiroFrancia(modelo_bc.resid)


# In[ ]: Histograma dos resíduos do modelo_bc

plt.figure(figsize=(10,10))
sns.histplot(data=modelo_bc.resid, kde=True, bins=30)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Fazendo predições com os modelos OLS linear e Box-Cox
#Qual é o comprimento esperado de um bebê com 52 semanas de vida?

#Modelo OLS Linear:
modelo_linear.predict(pd.DataFrame({'idade':[52]}))

#Modelo Não Linear (Box-Cox):
modelo_bc.predict(pd.DataFrame({'idade':[52]}))

#Não podemos nos esquecer de fazer o cálculo para a obtenção do fitted
#value de Y (variável 'comprimento')
(54251.109775 * lmbda + 1) ** (1 / lmbda)


# In[ ]: Salvando os fitted values dos dois modelos (modelo_linear e modelo_bc)
#no dataset 'bebes'

df_bebes['yhat_linear'] = modelo_linear.fittedvalues
df_bebes['yhat_modelo_bc'] = (modelo_bc.fittedvalues * lmbda + 1) ** (1 / lmbda)
df_bebes


# In[ ]: Ajustes dos modelos
#valores previstos (fitted values) X valores reais

from scipy.optimize import curve_fit

def objective(x, a, b, c, d, e):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + e

xdata = df_bebes['comprimento']
ydata_linear = df_bebes['yhat_linear']
ydata_bc = df_bebes['yhat_modelo_bc']

plt.figure(figsize=(10,10))

popt, _ = curve_fit(objective, xdata, ydata_linear)
a, b, c, d, e = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e)
plt.plot(x_line, y_line, '--', color='#2e1547', linewidth=3)

popt, _ = curve_fit(objective, xdata, ydata_bc)
a, b, c, d, e = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e)
plt.plot(x_line, y_line, '--', color='#ed7d31', linewidth=3)

plt.plot(xdata,xdata, color='gray', linestyle='-')
plt.scatter(xdata,ydata_linear, alpha=0.5, s=100, color='#2e1547')
plt.scatter(xdata,ydata_bc, alpha=0.5, s=100, color='#ed7d31')
plt.xlabel('Comprimento', fontsize=16)
plt.ylabel('Fitted Values', fontsize=16)
plt.legend(['OLS Linear','Box-Cox','45º graus'], fontsize=17)
plt.title('Dispersão e Fitted Values', fontsize=16)
plt.show()


# In[ ]:
#############################################################################
#                        REGRESSÃO NÃO LINEAR MÚLTIPLA                      #
#                  EXEMPLO 05 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################

df_empresas = pd.read_csv('empresas.csv', delimiter=',')
df_empresas

#Características das variáveis do dataset
df_empresas.info()

#Estatísticas univariadas
df_empresas.describe()


# In[ ]: Matriz de correlações

corr = df_empresas.corr()
corr

plt.figure(figsize=(15,10))
sns.heatmap(df_empresas.corr(), annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':15})
plt.show()


# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_empresas, diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Estimando a Regressão Múltipla
modelo_empresas = sm.OLS.from_formula('retorno ~ disclosure +\
                                      endividamento + ativos +\
                                          liquidez', df_empresas).fit()

# Parâmetros do modelo
modelo_empresas.summary()

#Note que o parâmetro da variável 'endividamento' não é estatisticamente
#significante ao nível de significância de 5% (nível de confiança de 95%).


# In[ ]: Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'stepwise_process.statsmodels'
#pip install "stepwise-process==2.5"
# Autores: Helder Prado Santos e Luiz Paulo Fávero
from stepwise_process.statsmodels import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_empresas = stepwise(modelo_empresas, pvalue_limit=0.05)


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiroFrancia' do pacote
#'sfrancia'
# Autores: Luiz Paulo Fávero e Helder Prado Santos
#pip install sfrancia==1.0.8
from sfrancia import shapiroFrancia
shapiroFrancia(modelo_step_empresas.resid)


# In[ ]: Plotando os resíduos do modelo step_empresas e acrescentando
#uma curva normal teórica para comparação entre as distribuições

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_empresas.resid, fit=norm, kde=True, bins=20)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#xt é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
xt, lmbda = boxcox(df_empresas['retorno'])

print("Primeiros valores: ",xt[:5])
print("Lambda: ",lmbda)


# In[ ]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_empresas['bc_retorno'] = xt
df_empresas


# In[ ]: Estimando um novo modelo múltiplo com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_retorno ~ disclosure +\
                                endividamento + ativos +\
                                    liquidez', df_empresas).fit()

# Parâmetros do modelo
modelo_bc.summary()


# In[ ]: Aplicando o procedimento Stepwise no 'modelo_bc"

modelo_step_empresas_bc = stepwise(modelo_bc, pvalue_limit=0.05)

#Note que a variável 'disclosure' acaba voltando ao modelo
#na forma funcional não linear!


# In[ ]: Verificando a normalidade dos resíduos do 'modelo_step_empresas_bc'

# Teste de Shapiro-Francia
shapiroFrancia(modelo_step_empresas_bc.resid)


# In[ ]: Plotando os novos resíduos do 'modelo_step_empresas_bc'

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_empresas_bc.resid, fit=norm, kde=True, bins=20)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()


# In[ ]: Resumo dos dois modelos obtidos pelo procedimento Stepwise
#(linear e com Box-Cox)

summary_col([modelo_step_empresas, modelo_step_empresas_bc],
            model_names=["STEPWISE","STEPWISE BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })


# In[ ]: Fazendo predições com o modelo_step_empresas_bc
# Qual é o valor do retorno, em média, para disclosure igual a 50,
#liquidez igual a 14 e ativo igual a 4000, ceteris paribus?

modelo_step_empresas_bc.predict(pd.DataFrame({'const':[1],
                                              'disclosure':[50],
                                              'ativos':[4000],
                                              'liquidez':[14]}))


# In[ ]: Não podemos nos esquecer de fazer o cálculo para a obtenção do fitted
#value de Y (retorno)

#Não podemos nos esquecer de fazer o cálculo para a obtenção do fitted value de Y (retorno)
(3.702016 * lmbda + 1) ** (1 / lmbda)


# In[ ]: Salvando os fitted values de 'modelo_step_empresas' e
#'modelo_step_empresas_bc'

df_empresas['yhat_step_empresas'] = modelo_step_empresas.fittedvalues
df_empresas['yhat_step_empresas_bc'] = (modelo_step_empresas_bc.fittedvalues
                                        * lmbda + 1) ** (1 / lmbda)

#Visualizando os dois fitted values no dataset
#modelos 'modelo_step_empresas e modelo_step_empresas_bc
df_empresas[['empresa','retorno','yhat_step_empresas','yhat_step_empresas_bc']]


# In[ ]: Ajustes dos modelos: valores previstos (fitted values) X valores reais

from scipy.optimize import curve_fit

def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

xdata = df_empresas['retorno']
ydata_linear = df_empresas['yhat_step_empresas']
ydata_bc = df_empresas['yhat_step_empresas_bc']

plt.figure(figsize=(10,10))

popt, _ = curve_fit(objective, xdata, ydata_linear)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='#2e1547', linewidth=3)

popt, _ = curve_fit(objective, xdata, ydata_bc)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='#ed7d31', linewidth=3)

plt.plot(xdata,xdata, color='gray', linestyle='-')
plt.scatter(xdata,ydata_linear, alpha=0.5, s=100, color='#2e1547')
plt.scatter(xdata,ydata_bc, alpha=0.5, s=100, color='#ed7d31')
plt.xlabel('Retorno', fontsize=16)
plt.ylabel('Fitted Values', fontsize=16)
plt.legend(['Stepwise','Stepwise Box-Cox','45º graus'], fontsize=14)
plt.title('Dispersão e Fitted Values', fontsize=16)
plt.show()


# In[ ]:
#############################################################################
#       DIAGNÓSTICO DE MULTICOLINEARIDADE EM MODELOS DE REGRESSÃO           #
#              EXEMPLO 06 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_salarios = pd.read_csv("salarios.csv", delimiter=',')
df_salarios

#Características das variáveis do dataset
df_salarios.info()

#Estatísticas univariadas
df_salarios.describe()


# In[ ]: CORRELAÇÃO PERFEITA:

corr1 = df_salarios[['rh1','econometria1']].corr()
corr1

plt.figure(figsize=(15,10))
sns.heatmap(corr1, annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':27})

# Estimando um modelo com variáveis preditoras com correlação perfeita
modelo_1 = sm.OLS.from_formula('salario ~ rh1 + econometria1',
                               df_salarios).fit()

# Parâmetros do modelo
modelo_1.summary()


# In[ ]: CORRELAÇÃO BAIXA:

corr3 = df_salarios[['rh3','econometria3']].corr()
corr3

plt.figure(figsize=(15,10))
sns.heatmap(corr3, annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':27})

# Estimando um modelo com variáveis preditoras com correlação baixa
modelo_3 = sm.OLS.from_formula('salario ~ rh3 + econometria3',
                               df_salarios).fit()

# Parâmetros do modelo
modelo_3.summary()

# Diagnóstico de multicolinearidade (Variance Inflation Factor e Tolerance)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df_salarios[['rh3','econometria3']]
X = sm.add_constant(X)

vif = pd.Series([variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])],index=X.columns)
vif

tolerance = 1/vif
tolerance

pd.concat([vif,tolerance], axis=1, keys=['VIF', 'Tolerance'])


# In[ ]: CORRELAÇÃO MUITO ALTA, PORÉM NÃO PERFEITA:

corr2 = df_salarios[['rh2','econometria2']].corr()
corr2

plt.figure(figsize=(15,10))
sns.heatmap(corr2, annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':27})

# Estimando um modelo com variáveis preditoras com correlação quase perfeita
modelo_2 = sm.OLS.from_formula('salario ~ rh2 + econometria2',
                               df_salarios).fit()

# Parâmetros do modelo
modelo_2.summary()

# Diagnóstico de multicolinearidade (Variance Inflation Factor e Tolerance)

X = df_salarios[['rh2','econometria2']]
X = sm.add_constant(X)

vif = pd.Series([variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])],index=X.columns)
vif

tolerance = 1/vif
tolerance

pd.concat([vif,tolerance], axis=1, keys=['VIF', 'Tolerance'])


# In[ ]:
#############################################################################
#      DIAGNÓSTICO DE HETEROCEDASTICIDADE EM MODELOS DE REGRESSÃO           #
#              EXEMPLO 07 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################
    
df_saeb_rend = pd.read_csv("saeb_rend.csv", delimiter=',')
df_saeb_rend

#Características das variáveis do dataset
df_saeb_rend.info()

#Estatísticas univariadas
df_saeb_rend.describe()


# In[ ]: Tabela de frequências absolutas das variáveis 'uf' e rede'

df_saeb_rend['uf'].value_counts()
df_saeb_rend['rede'].value_counts()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com linear fit

x = df_saeb_rend['rendimento']
y = df_saeb_rend['saeb']
plt.plot(x, y, 'o', color='#FDE725FF', markersize=5, alpha=0.6)
sns.regplot(x="rendimento", y="saeb", data=df_saeb_rend)
plt.title('Dispersão dos dados com linear fit')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento',
#com destaque para 'rede' escolar

sns.scatterplot(x="rendimento", y="saeb", data=df_saeb_rend,
                hue="rede", alpha=0.6, palette = "viridis")
plt.title('Dispersão dos dados por rede escolar')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento',
#com destaque para 'rede' escolar e linear fits

sns.lmplot(x="rendimento", y="saeb", data=df_saeb_rend,
           hue="rede", ci=None, palette="viridis")
plt.title('Dispersão dos dados por rede escolar e com linear fits')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()


# In[ ]: Estimação do modelo de regressão e diagnóstico de heterocedasticidade

# Estimando o modelo
modelo_saeb = sm.OLS.from_formula('saeb ~ rendimento', df_saeb_rend).fit()

# Parâmetros do modelo
modelo_saeb.summary()


# In[ ]: Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

from scipy import stats

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value


# In[ ]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_saeb)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!


# In[ ]: Dummizando a variável 'uf'

df_saeb_rend_dummies = pd.get_dummies(df_saeb_rend, columns=['uf'],
                                      drop_first=True)

df_saeb_rend_dummies.head(10)


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_saeb_rend_dummies.drop(columns=['municipio',
                                                        'codigo',
                                                        'escola',
                                                        'rede',
                                                        'saeb']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "saeb ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_saeb_dummies_uf = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_saeb_rend_dummies).fit()

#Parâmetros do modelo
modelo_saeb_dummies_uf.summary()


# In[ ]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_saeb_dummies_uf'

breusch_pagan_test(modelo_saeb_dummies_uf)


# In[ ]: Plotando 'saeb' em função de 'rendimento',
#com destaque para UFs e linear fits

sns.lmplot(x="rendimento", y="saeb", data=df_saeb_rend,
           hue="uf", ci=None, palette="viridis")
plt.title('Dispersão dos dados por UF e com linear fits')
plt.xlabel('rendimento')
plt.ylabel('saeb')
plt.show()


# In[ ]:
#############################################################################
#               REGRESSÃO NÃO LINEAR MÚLTIPLA COM DUMMIES                   #
#               EXEMPLO 08 - CARREGAMENTO DA BASE DE DADOS                  #
#############################################################################

df_planosaude = pd.read_csv("planosaude.csv", delimiter=',')
df_planosaude

#Características das variáveis do dataset
df_planosaude.info()

#Estatísticas univariadas
df_planosaude.describe()


# In[ ]: Transformação da variável 'plano' para o tipo categórico

df_planosaude['plano'] = df_planosaude['plano'].astype('category')
df_planosaude['plano']


# In[ ]: Tabela de frequências absolutas da variável 'plano'

df_planosaude['plano'].value_counts()


# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_planosaude.loc[:,"despmed":"renda"], diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Dummizando a variável 'plano'

df_planosaude_dummies = pd.get_dummies(df_planosaude, columns=['plano'],
                                      drop_first=True)

df_planosaude_dummies.head(10)


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_planosaude_dummies.drop(columns=['id',
                                                         'despmed']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "despmed ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_planosaude = sm.OLS.from_formula(formula_dummies_modelo,
                                        df_planosaude_dummies).fit()

#Parâmetros do modelo
modelo_planosaude.summary()


# In[ ]: Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'stepwise_process.statsmodels'
#pip install "stepwise-process==2.5"
# Autores: Helder Prado Santos e Luiz Paulo Fávero
from stepwise_process.statsmodels import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_planosaude = stepwise(modelo_planosaude, pvalue_limit=0.05)


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiroFrancia' do pacote
#'sfrancia'
# Autores: Luiz Paulo Fávero e Helder Prado Santos
#pip install sfrancia==1.0.8
from sfrancia import shapiroFrancia
shapiroFrancia(modelo_step_planosaude.resid)


# In[ ]: Plotando os resíduos do 'modelo_step_planosaude',
#com curva normal teórica

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_planosaude.resid, fit=norm, kde=True, bins=15)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()


# In[ ]: Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

plt.figure(figsize=(15,10))
sns.kdeplot(data=modelo_step_planosaude.resid, multiple="stack",
            color='#55C667FF')
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()


# In[ ]: Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

#from scipy import stats

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value


# In[ ]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_step_planosaude)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!


# In[ ]: Adicionando fitted values e resíduos do 'modelo_step_planosaude'
#no dataset 'df_planosaude_dummies'

df_planosaude_dummies['fitted_step'] = modelo_step_planosaude.fittedvalues
df_planosaude_dummies['residuos_step'] = modelo_step_planosaude.resid
df_planosaude_dummies


# In[ ]: Gráfico que relaciona resíduos e fitted values
#do 'modelo_step_planosaude'

xdata = df_planosaude_dummies['fitted_step']
ydata = df_planosaude_dummies['residuos_step']

plt.figure(figsize=(15,10))
plt.scatter(xdata, ydata, alpha=0.6)
plt.xlabel('Fitted Values do Modelo Stepwise', fontsize=16)
plt.ylabel('Resíduos do Modelo Stepwise', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#x é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
x, lmbda = boxcox(df_planosaude_dummies['despmed'])

print("Primeiros valores: ",x[:5])
print("Lambda: ",lmbda)


# In[ ]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_planosaude_dummies['bc_despmed'] = x
df_planosaude_dummies


# In[ ]: Estimando um novo modelo com todas as variáveis e a
#variável dependente transformada
modelo_bc_planosaude = sm.OLS.from_formula('bc_despmed ~ idade + dcron +\
                                           renda + plano_esmeralda +\
                                               plano_ouro',
                                               df_planosaude_dummies).fit()

#Parâmetros do modelo
modelo_bc_planosaude.summary()


# In[ ]: Procedimento Stepwise

modelo_step_bc_planosaude = stepwise(modelo_bc_planosaude, pvalue_limit=0.05)


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

#Teste de Shapiro-Francia
shapiroFrancia(modelo_step_bc_planosaude.resid)


# In[ ]: Plotando os novos resíduos do 'modelo_step_bc_planosaude'
#com curva normal teórica

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_bc_planosaude.resid, fit=norm, kde=True, bins=15)
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()


# In[ ]: Kernel density estimation (KDE)

plt.figure(figsize=(15,10))
sns.kdeplot(data=modelo_step_bc_planosaude.resid, multiple="stack",
            color='#440154FF')
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequências', fontsize=16)
plt.show()


# In[ ]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_step_bc_planosaude'

breusch_pagan_test(modelo_step_bc_planosaude)


# In[ ]: Adicionando fitted values e resíduos do 'modelo_step_bc_planosaude'
#no dataset 'df_planosaude_dummies'

df_planosaude_dummies['fitted_step_bc'] = modelo_step_bc_planosaude.fittedvalues
df_planosaude_dummies['residuos_step_bc'] = modelo_step_bc_planosaude.resid
df_planosaude_dummies


# In[ ]: Gráfico que relaciona resíduos e fitted values
#do 'modelo_step_bc_planosaude'

xdata = df_planosaude_dummies['fitted_step_bc']
ydata = df_planosaude_dummies['residuos_step_bc']

plt.figure(figsize=(15,10))
plt.scatter(xdata, ydata, alpha=0.6)
plt.xlabel('Fitted Values do Modelo Stepwise com Transformação de Box-Cox',
           fontsize=16)
plt.ylabel('Resíduos do Modelo Stepwise com Transformação de Box-Cox',
           fontsize=16)
plt.show()


################################### FIM ######################################