# MBA DATA SCIENCE & ANALYTICS USP/Esalq
# SUPERVISED MACHINE LEARNING: MODELOS DE REGRESSÃO PARA DADOS DE CONTAGEM
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários

import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy import stats
from statsmodels.iolib.summary2 import summary_col
from math import exp, factorial
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')


# In[ ]:
##############################################################################
#                 A DISTRIBUIÇÃO POISSON - PARTE CONCEITUAL                  #
##############################################################################

#Estabelecendo uma função da distribuição Poisson para determinados valores
#de lambda
def poisson_lambda(lmbda,m):
    return (exp(-lmbda) * lmbda ** m) / factorial(m)


# In[ ]: Plotagem das funções estabelecidas para diferentes valores de lambda

m = np.arange(0,21)

lmbda_1 = []
lmbda_2 = []
lmbda_4 = []

for item in m:
    # Estabelecendo a distribuição com lambda = 1
    lmbda_1.append(poisson_lambda(1,item))
    # Estabelecendo a distribuição com lambda = 2
    lmbda_2.append(poisson_lambda(2,item))
    # Estabelecendo a distribuição com lambda = 4
    lmbda_4.append(poisson_lambda(4,item))

#Criando um dataframe com m variando de 0 a 20 e diferentes valores de lambda
df_lambda = pd.DataFrame({'m':m,
                          'lambda_1':lmbda_1,
                          'lambda_2':lmbda_2,
                          'lambda_4':lmbda_4})
df_lambda


# In[ ]: Plotagem propriamente dita

from scipy.interpolate import interp1d

def smooth_line_plot(x,y):
    x_new = np.linspace(x.min(), x.max(),500)
    f = interp1d(x, y, kind='quadratic')
    y_smooth=f(x_new)
    return x_new, y_smooth

x_new, lambda_1 = smooth_line_plot(df_lambda.m, df_lambda.lambda_1)
x_new, lambda_2 = smooth_line_plot(df_lambda.m, df_lambda.lambda_2)
x_new, lambda_4 = smooth_line_plot(df_lambda.m, df_lambda.lambda_4)

plt.figure(figsize=(15,10))
plt.plot(x_new,lambda_1, linewidth=5, color='#440154FF')
plt.plot(x_new,lambda_2, linewidth=5, color='#22A884FF')
plt.plot(x_new,lambda_4, linewidth=5, color='#FDE725FF')
plt.xlabel('m', fontsize=20)
plt.ylabel('Probabilidades', fontsize=20)
plt.legend([r'$\lambda$ = 1',r'$\lambda$ = 2',r'$\lambda$ = 4'], fontsize=24)
plt.show


# In[6]:
##############################################################################
#                      REGRESSÃO PARA DADOS DE CONTAGEM                      #
#                  CARREGAMENTO DA BASE DE DADOS corruption                  #
##############################################################################

#Fisman, R.; Miguel, E. Corruption, Norms, and Legal Enforcement:
#Evidence from Diplomatic Parking Tickets.
#Journal of Political Economy, v. 15, n. 6, p. 1020-1048, 2007.
#https://www.journals.uchicago.edu/doi/abs/10.1086/527495

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
df_corruption

#Características das variáveis do dataset
df_corruption.info()

#Estatísticas univariadas
df_corruption.describe()


# In[ ]: Tabela de frequências da variável dependente 'violations'
#Função 'values_counts' do pacote 'pandas' sem e com normalização
#para gerar as contagens e os percentuais, respectivamente
contagem = df_corruption['violations'].value_counts(dropna=False)
percent = df_corruption['violations'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=True)


# In[ ]: Histograma da variável dependente 'violations'

plt.figure(figsize=(15,10))
sns.histplot(data=df_corruption, x="violations", bins=20, color='darkorchid')
plt.show()


# In[ ]: Diagnóstico preliminar para observação de eventual igualdade entre a
#média e a variância da variável dependente 'violations'

pd.DataFrame({'Média':[df_corruption.violations.mean()],
              'Variância':[df_corruption.violations.var()]})


# In[ ]: Comportamento das variáveis 'corruption' e 'violations' antes e
#depois do início da vigência da lei

fig, axs = plt.subplots(ncols=2, figsize=(20,10), sharey=True)

fig.suptitle('Diferença das violações de trânsito em NY antes e depois da vigência da lei',
             fontsize = 20)

post = ['no','yes']

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y']+.1, str(point['val']))

for i, v in enumerate(post):
    df = df_corruption[df_corruption.post==v]
    df['violations'] = np.log(df.violations)
    df.loc[df['violations']==np.inf, 'violations'] = 0
    df.loc[df['violations']==-np.inf, 'violations'] = 0
    sns.regplot(data=df, x='corruption', y='violations',order=3, ax=axs[i],
                ci=False, color='darkorchid')
    axs[i].set_title(v)
    axs[i].set_ylabel("Violações de Trânsito em NY (logs)", fontsize = 17)
    axs[i].set_xlabel("Índice de corrupção dos países", fontsize = 17)
    label_point(df.corruption, df.violations, df.code, axs[i])  

plt.show()


# In[ ]: Estimação do modelo Poisson

#O argumento 'family=sm.families.Poisson()' da função 'smf.glm' define a
#estimação de um modelo Poisson
modelo_poisson = smf.glm(formula='violations ~ staff + post + corruption',
                         data=df_corruption,
                         family=sm.families.Poisson()).fit()

#Parâmetros do modelo
modelo_poisson.summary()


# In[ ]: Outro modo mais completo de apresentar os outputs do modelo,
#pela função 'summary_col'

summary_col([modelo_poisson],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
        })


# In[ ]: Todas as variáveis preditoras se mostraram estatisticamente
#diferentes de zero, considerando-se um nível de significância de 5%,
#ceteris paribus. Porém, já se pode afirmar que a estimação Poisson é a mais
#adequada?

################################################################################
#            TESTE DE SUPERDISPERSÃO DE CAMERON E TRIVEDI (1990)               #
################################################################################
#CAMERON, A. C.; TRIVEDI, P. K. Regression-based tests for overdispersion in
#the Poisson model. Journal of Econometrics, v. 46, n. 3, p. 347-364, 1990.

#1º Passo: estimar um modelo Poisson;
#2º Passo: criar uma nova variável (Y*) utilizando os fitted values do modelo
#Poisson estimado anteriormente;
#3º Passo: estimar um modelo auxiliar OLS, com a variável Y* como variável
#dependente, os fitted values do modelo Poisson como única variável preditora e 
#sem o intercepto;
#4º Passo: Observar a significância do parâmetro beta.

#Adicionando os fitted values do modelo Poisson (lambda_poisson) ao dataframe:
df_corruption['lambda_poisson'] = modelo_poisson.fittedvalues
df_corruption

#Criando a nova variável Y*:
df_corruption['ystar'] = (((df_corruption['violations']
                            -df_corruption['lambda_poisson'])**2)
                          -df_corruption['violations'])/df_corruption['lambda_poisson']
df_corruption

#Estimando o modelo auxiliar OLS, sem o intercepto:
modelo_auxiliar = smf.ols(formula='ystar ~ 0 + lambda_poisson',
                          data=df_corruption).fit()

#Parâmetros do 'modelo_auxiliar'
modelo_auxiliar.summary()

#Caso o p-value do parâmetro do lambda_poisson seja maior que 0.05,
#verifica-se a existência de equidispersão nos dados.
#Caso contrário, diagnostica-se a existência de superdispersão nos dados, fato
#que favorecerá a estimação de um modelo binomial negativo, como ocorre nesse
#caso.


# In[ ]: Função 'overdisp'
# Instalação e carregamento da função 'overdisp' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import overdisp

#Elaboração direta do teste de superdispersão
overdisp(modelo_poisson, df_corruption)


# In[ ]: Apenas para fins didáticos, caso considerássemos a estimação Poisson
#como a mais adequada, qual seria a quantidade média esperada de violações
#de trânsito para um país cujo corpo diplomático fosse composto por 23 membros,
#considerando o período anterior à vigência da lei e cujo índice de corrupção
#seja igual a 0.5?

modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['no'],
                                     'corruption':[0.5]}))

#Qual seria a quantidade média esperada de violações de trânsito para o mesmo
#país, porém agora considerando a vigência da lei?

modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['yes'],
                                     'corruption':[0.5]}))


# In[ ]:
##############################################################################
#            A DISTRIBUIÇÃO BINOMIAL NEGATIVA - PARTE CONCEITUAL             #
##############################################################################

#Estabelecendo uma função da distribuição binomial negativa para determinados
#valores de theta e delta
#theta: parâmetro de forma da distribuição Poisson-Gama (binomial negativa)
#delta: parâmetro de taxa de decaimento da distribuição Poisson-Gama

def bneg(theta, delta, m):
    return ((delta ** theta) * (m ** (theta - 1)) * (exp(-m * delta))) / factorial(theta - 1)


# In[ ]: Plotagem das funções estabelecidas para diferentes valores de
#theta e delta

m = np.arange(1,21)

bneg_theta2_delta2 = []
bneg_theta3_delta1 = []
bneg_theta3_delta05 = []

for item in m:
    # Estabelecendo a distribuição binomial negativa com theta=2 e delta=2
    bneg_theta2_delta2.append(bneg(2,2,item))
    # Estabelecendo a distribuição binomial negativa com theta=3 e delta=1
    bneg_theta3_delta1.append(bneg(3,1,item))
    # Estabelecendo a distribuição binomial negativa com theta=3 e delta=0.5
    bneg_theta3_delta05.append(bneg(3,0.5,item))
   
#Criando um dataframe com m variando de 1 a 20 e diferentes valores de
#theta e delta
df_bneg = pd.DataFrame({'m':m,
                        'bneg_theta2_delta2':bneg_theta2_delta2,
                        'bneg_theta3_delta1':bneg_theta3_delta1,
                        'bneg_theta3_delta05':bneg_theta3_delta05})

df_bneg


# In[ ]: Plotagem propriamente dita

def smooth_line_plot(x,y):
    x_new = np.linspace(x.min(), x.max(),500)
    f = interp1d(x, y, kind='quadratic')
    y_smooth=f(x_new)
    return x_new, y_smooth

x_new, bneg_theta2_delta2 = smooth_line_plot(df_bneg.m,
                                             df_bneg.bneg_theta2_delta2)
x_new, bneg_theta3_delta1 = smooth_line_plot(df_bneg.m,
                                             df_bneg.bneg_theta3_delta1)
x_new, bneg_theta3_delta05 = smooth_line_plot(df_bneg.m,
                                              df_bneg.bneg_theta3_delta05)

plt.figure(figsize=(15,10))
plt.plot(x_new,bneg_theta2_delta2, linewidth=5, color='#440154FF')
plt.plot(x_new,bneg_theta3_delta1, linewidth=5, color='#22A884FF')
plt.plot(x_new,bneg_theta3_delta05, linewidth=5, color='#FDE725FF')
plt.xlabel('m', fontsize=20)
plt.ylabel('Probabilidades', fontsize=20)
plt.legend([r'$\theta$ = 2 e $\delta$ = 2',
            r'$\theta$ = 3 e $\delta$ = 1',
            r'$\theta$ = 3 e $\delta$ = 0.5'],
           fontsize=24)
plt.show


# In[ ]: Estimação do modelo binomial negativo do tipo NB2

#O argumento 'family=sm.families.NegativeBinomial(alpha=2.0963)' da função
#'smf.glm' define a estimação de um modelo binomial negativo do tipo NB2
#com valor de 'fi' ('alpha' no Python) igual a 2.0963. Lembramos que 'fi' é o
#inverso do parâmetro de forma 'theta' da distribuição Poisson-Gama.

modelo_bneg = smf.glm(formula='violations ~ staff + post + corruption',
                      data=df_corruption,
                      family=sm.families.NegativeBinomial(alpha=2.0963)).fit()

#Parâmetros do modelo
modelo_bneg.summary()

#Construção de função para a definição do 'alpha' ('fi') ótimo que gera a
#maximização do valor de Log-Likelihood
n_samples = 10000
alphas = np.linspace(0, 10, n_samples)
llf = np.full(n_samples, fill_value=np.nan)
for i, alpha in enumerate(alphas):
    try:
        model = smf.glm(formula = 'violations ~ staff + post + corruption',
                        data=df_corruption,
                        family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except:
        continue
    llf[i] = model.llf
alpha_ótimo = alphas[np.nanargmax(llf)]
alpha_ótimo

#Plotagem dos resultados
plt.plot(alphas, llf, label='Log-Likelihood')
plt.axvline(x=alpha_ótimo, color='#440154FF',
            label=f'alpha: {alpha_ótimo:0.5f}')
plt.legend()

#Reestimação do modelo binomial negativo com o parâmetro 'alpha_ótimo'
modelo_bneg = smf.glm(formula='violations ~ staff + post + corruption',
                      data=df_corruption,
                      family=sm.families.NegativeBinomial(alpha=alpha_ótimo)).fit()

#Parâmetros do modelo
modelo_bneg.summary()


# In[ ]: Comparando os modelos Poisson e binomial negativo

summary_col([modelo_poisson, modelo_bneg], 
            model_names=["Poisson","BNeg"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf),
                'Pseudo-R2':lambda x: "{:.4f}".format(x.pseudo_rsquared()),
        })


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_poisson, modelo_bneg])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson e
#binomial negativo

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','BNeg'],
                      'loglik':[modelo_poisson.llf, modelo_bneg.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ['#440154FF', '#22A884FF']

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimação", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)


# In[ ]: COMPARAÇÕES ENTRE AS PREVISÕES:
#Qual seria a quantidade média esperada de violações de trânsito para um país
#cujo corpo diplomático seja composto por 23 membros, considerando o período
#anterior à vigência da lei e cujo índice de corrupção seja igual 0.5?

#Modelo Poisson:
modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['no'],
                                     'corruption':[0.5]}))

#Modelo binomial negativo:
modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                     'post':['no'],
                                     'corruption':[0.5]}))

#Qual seria a quantidade média esperada de violações de trânsito para o mesmo
#país, porém agora considerando a vigência da lei?

#Modelo Poisson:
modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['yes'],
                                     'corruption':[0.5]}))

#Modelo binomial negativo:
modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                     'post':['yes'],
                                     'corruption':[0.5]}))


# In[ ]: Adicionando os fitted values dos modelos estimados até o momento,
#para fins de comparação

df_corruption['fitted_poisson'] = modelo_poisson.fittedvalues
df_corruption['fitted_bneg'] = modelo_bneg.fittedvalues

df_corruption[['country','code','violations','fitted_poisson','fitted_bneg']]


# In[ ]: Fitted values dos modelos Poisson e binomial negativo, considerando,
#para fins didáticos, apenas a variável preditora 'staff':

plt.figure(figsize=(20,10))
sns.relplot(data=df_corruption, x='staff', y='violations',
            ci=False, color='black', height=8)
sns.regplot(data=df_corruption, x='staff', y='fitted_poisson', order=3,
            color='#440154FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_bneg', order=3,
            color='#22A884FF')
plt.xlabel('Number of Diplomats (staff)', fontsize=17)
plt.ylabel('Unpaid Parking Violations (violations)', fontsize=17)
plt.legend(['Observado', 'Poisson', 'Fit Poisson', 'CI Poisson',
            'BNeg', 'Fit BNeg', 'CI BNeg'],
           fontsize=17)
plt.show


# In[ ]: Estimações muito próximas para Poisson e BNeg sem superdispersão!

#Para fins didáticos, vamos gerar novo dataset 'corruption2', com quantidades
#de violações de trânsito iguais, no máximo, a 3. Este procedimento poderá,
#eventualmente, eliminar o fenômeno da superdispersão nos dados da variável
#dependente e, consequentemente, tornar as estimações dos modelos POISSON e
#BINOMIAL NEGATIVO praticamente iguais.

#Gerando novo dataset 'corruption2' com violations <= 3
df_corruption2 = df_corruption[df_corruption.violations <= 3]
df_corruption2 = df_corruption2.iloc[:, 0:6] 
df_corruption2


# In[ ]: Histograma da variável dependente 'violations' no dataset
#'corruption2'

plt.figure(figsize=(15,10))
sns.histplot(data=df_corruption2, x="violations", bins=4, color='darkorchid')
plt.show()


# In[ ]: Diagnóstico preliminar para observação de eventual igualdade entre
#a média e a variância da variável dependente 'violations' no dataset
#'corruption2'

pd.DataFrame({'Média':[df_corruption2['violations'].mean()],
              'Variância':[df_corruption2['violations'].var()]})


# In[ ]: Estimação do 'modelo_poisson2'

modelo_poisson2 = smf.glm(formula='violations ~ staff + post + corruption',
                          data=df_corruption2,
                          family=sm.families.Poisson()).fit()

#Parâmetros do modelo
modelo_poisson2.summary()


# In[ ]: Teste de superdispersão no dataset 'corruption2'

#Adicionando os fitted values do 'modelo_poisson2' (lambda_poisson2)
#ao dataframe 'df_corruption2':
df_corruption2['lambda_poisson2'] = modelo_poisson2.fittedvalues
df_corruption2

#Criando a nova variável Y*:
df_corruption2['ystar'] = (((df_corruption2['violations']
                            -df_corruption2['lambda_poisson2'])**2)
                          -df_corruption2['violations'])/df_corruption2['lambda_poisson2']
df_corruption2

#Estimando o 'modelo_auxiliar2' OLS, sem o intercepto:
modelo_auxiliar2 = smf.ols(formula='ystar ~ 0 + lambda_poisson2',
                           data=df_corruption2).fit()

#Parâmetros do 'modelo_auxiliar2'
modelo_auxiliar2.summary()

#Como o p-value do parâmetro de 'lambda_poisson2' é maior que 0.05,
#verifica-se a existência de equidispersão nos dados.


# In[ ]: Teste de superdispersão no dataset 'corruption2'

# Função 'overdisp'
# Instalação e carregamento da função 'overdisp' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import overdisp

#Elaboração direta do teste de superdispersão
overdisp(modelo_poisson2, df_corruption2)


# In[ ]: Estimação do 'modelo_bneg2'
#Construção de função para a definição do 'alpha' ('fi') ótimo que gera a
#maximização do valor de Log-Likelihood
n_samples = 10000
alphas = np.linspace(0, 10, n_samples)
llf = np.full(n_samples, fill_value=np.nan)
for i, alpha in enumerate(alphas):
    try:
        model = smf.glm(formula = 'violations ~ staff + post + corruption',
                        data=df_corruption2,
                        family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except:
        continue
    llf[i] = model.llf
alpha_ótimo = alphas[np.nanargmax(llf)]
alpha_ótimo

#Plotagem dos resultados
plt.plot(alphas, llf, label='Log-Likelihood')
plt.axvline(x=alpha_ótimo, color='#440154FF',
            label=f'alpha: {alpha_ótimo:0.5f}')
plt.legend()

#Estimação do 'modelo_bneg2' com o parâmetro 'alpha_ótimo'
modelo_bneg2 = smf.glm(formula='violations ~ staff + post + corruption',
                       data=df_corruption2,
                       family=sm.families.NegativeBinomial(alpha=alpha_ótimo)).fit()

#Parâmetros do 'modelo_bneg2'
modelo_bneg2.summary()


# In[ ]: Comparando os 'modelo_poisson2' e 'modelo_bneg2'

summary_col([modelo_poisson2, modelo_bneg2], 
            model_names=["Poisson2","BNEG2"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf),
                'Pseudo-R2':lambda x: "{:.2f}".format(x.pseudo_rsquared()),
        })


# In[ ]: likelihood ratio test para comparação de LL's entre os modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_poisson2, modelo_bneg2])
# Quando não há superdispersão, não existem diferenças significantes entre os
#modelos Poisson e binomial negativo!


# In[ ]:
##############################################################################
#       A DISTRIBUIÇÃO ZERO-INFLATED POISSON (ZIP) - PARTE CONCEITUAL        #
##############################################################################

#LAMBERT, D. Zero-inflated Poisson regression, with an application to defects
#in manufacturing. Technometrics, v. 34, n. 1, p. 1-14, 1992.

#Exemplo de uma função da distribuição ZI Poisson, com lambda=1 e plogit=0,7
def zip_lambda1_plogit07(m):
    lmbda = 1
    plogit = 0.7
    
    if m == 0:
        return (plogit) + ((1 - plogit) * exp(-lmbda))
    else:
        return (1 - plogit) * ((exp(-lmbda) * lmbda ** m) / factorial(m))


# In[ ]: Plotagem das funções estabelecidas

m = np.arange(0,21)

zip_lambda1_plogit07 = [zip_lambda1_plogit07(i) for i in m]

#Criando um dataframe com m variando de 0 a 20

df_zip = pd.DataFrame({'m':m,
                       'zip_lambda1_plogit07':zip_lambda1_plogit07})
df_zip


# In[ ]: Plotagem propriamente dita
#Comparando as distribuições Poisson, BNeg e ZIP

def smooth_line_plot(x,y):
    x_new = np.linspace(x.min(), x.max(),500)
    f = interp1d(x, y, kind='quadratic')
    y_smooth=f(x_new)
    return x_new, y_smooth

x_new, zip_lambda1_plogit07 = smooth_line_plot(df_zip.m,
                                               df_zip.zip_lambda1_plogit07)

plt.figure(figsize=(15,10))
plt.plot(x_new,lambda_1, linewidth=3, color='#404688FF')
plt.plot(x_new,lambda_2, linewidth=3, color='#2C728EFF')
plt.plot(x_new,lambda_4, linewidth=3, color='#20A486FF')
plt.plot(x_new,bneg_theta2_delta2, linewidth=3, color='#75D054FF')
plt.plot(x_new,bneg_theta3_delta1, linewidth=3, color='#C7E020FF')
plt.plot(x_new,bneg_theta3_delta05, linewidth=3, color='#FDE725FF')
plt.plot(x_new,zip_lambda1_plogit07, linewidth=7, color="#440154FF")
plt.xlabel('m', fontsize=20)
plt.ylabel('Probabilidades', fontsize=20)
plt.legend([r'Poisson: $\lambda$ = 1',
            r'Poisson: $\lambda$ = 2',
            r'Poisson: $\lambda$ = 4',
            r'BNeg: $\theta$ = 2 e $\delta$ = 2',
            r'BNeg: $\theta$ = 3 e $\delta$ = 1',
            r'BNeg: $\theta$ = 3 e $\delta$ = 0.5',
            r'ZIP: $\lambda$ = 1 e plogit = 0.7'],
           fontsize=24)
plt.show


# In[ ]:
##############################################################################
#              ESTIMAÇÃO DO MODELO ZERO-INFLATED POISSON (ZIP)               #
##############################################################################

#Definição da variável dependente (voltando ao dataset 'df_corruption')
y = df_corruption.violations

#Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_corruption[['staff','post','corruption']]
X1 = sm.add_constant(x1)

#Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_corruption[['corruption']]
X2 = sm.add_constant(x2)

#Se estimarmos o modelo sem dummizar as variáveis categórias, o modelo retorna
#um erro
X1 = pd.get_dummies(X1, columns=['post'], drop_first=True)

#Estimação do modelo ZIP pela função 'ZeroInflatedPoisson' do pacote
#'Statsmodels'

#Estimação do modelo ZIP
#O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zip = sm.ZeroInflatedPoisson(y, X1, exog_infl=X2,
                                    inflation='logit').fit()

#Parâmetros do modelo
modelo_zip.summary()


# In[ ]: Gráfico para comparar os valores previstos x valores reais de
#'violations' pelo modelo ZIP

zip_predictions = modelo_zip.predict(X1, exog_infl=X2)
predicted_counts = np.round(zip_predictions)
actual_counts = df_corruption['violations']

plt.figure(figsize=(15,10))
plt.plot(df_corruption.index, predicted_counts, 'go-',
         color='orange')
plt.plot(df_corruption.index, actual_counts, 'go-',
         color='#440154FF')
plt.xlabel('Observação', fontsize=20)
plt.ylabel('Violações de Trânsito', fontsize=20)
plt.legend(['Valores Previstos pelo ZIP', 'Valores Reais no Dataset'],
           fontsize=20)
plt.show()


# In[ ]: Comparando os modelos Poisson e ZIP

summary_col([modelo_poisson, modelo_zip], 
            model_names=["Poisson","ZIP"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf),
                'Pseudo-R2':lambda x: "{:.4f}".format(x.pseudo_rsquared()),
        })


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_poisson, modelo_zip])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson, BNeg e ZIP

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','ZIP','BNeg'],
                      'loglik':[modelo_poisson.llf,
                                modelo_zip.llf,
                                modelo_bneg.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ["#440154FF", "#453781FF", "#22A884FF"]

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimação", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)


# In[ ]: COMPARAÇÕES ENTRE AS PREVISÕES:
#Supondo que considerássemos a estimação ZIP como a mais adequada, qual seria a 
#quantidade média esperada de violações de trânsito para um país cujo corpo 
#diplomático seja composto por 23 membros, considerando o período anterior à 
#vigência da lei e cujo índice de corrupção seja igual a 0.5?

#Modelo Poisson:
modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['no'],
                                     'corruption':[0.5]}))

#Modelo binomial negativo:
modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                  'post':['no'],
                                  'corruption':[0.5]}))

#Modelo ZIP:
#Obs.: manter a ordem dos parâmetros nos argumentos da função 'predict'
modelo_zip.params

modelo_zip.predict(pd.DataFrame({'const':[1],
                                 'staff':[23],
                                 'corruption':[0.5],
                                 'post_yes':[0]}),
                   exog_infl=pd.DataFrame({'const':[1],
                                           'corruption':[0.5]}))

#Qual seria a quantidade média esperada de violações de trânsito para o mesmo
#país ao se considerar o início da vigência da lei?

#Modelo Poisson:
modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['yes'],
                                     'corruption':[0.5]}))

#Modelo binomial negativo:
modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                  'post':['yes'],
                                  'corruption':[0.5]}))

#Modelo ZIP:
modelo_zip.predict(pd.DataFrame({'const':[1],
                                 'staff':[23],
                                 'corruption':[0.5],
                                 'post_yes':[1]}),
                   exog_infl=pd.DataFrame({'const':[1],
                                           'corruption':[0.5]}))


# In[ ]:
##############################################################################
#  A DISTRIBUIÇÃO ZERO-INFLATED BINOMIAL NEGATIVA (ZINB) - PARTE CONCEITUAL  #
##############################################################################

#Exemplo de uma função da distribuição ZI Binomial Negativa, com theta = 2,
#delta = 2, plogit = 0.7 e lambda_bneg = 2
def zinb_theta2_delta2_plogit07_lambda2(m):
    lambda_bneg = 1
    plogit = 0.7
    theta = 2
    delta = 2
    if m == 0:
        return (plogit) + ((1 - plogit) *
                           (((1) / (1 + 1/theta * lambda_bneg)) ** theta))
    else:
        return (1 - plogit) * ((delta ** theta) * (m ** (theta - 1)) *
                               (exp(-m * delta))) / factorial(theta - 1)


# In[ ]: Plotagem das funções estabelecidas

m = np.arange(0,21)

zinb_theta2_delta2_plogit07_lambda2 = [zinb_theta2_delta2_plogit07_lambda2(i)
                                       for i in m]

#Criando um dataframe com m variando de 0 a 20

df_zinb = pd.DataFrame({'m':m,
                       'zinb_theta2_delta2_plogit07_lambda2':zinb_theta2_delta2_plogit07_lambda2})
df_zinb


# In[ ]: Plotagem propriamente dita
#Comparando as distribuições Poisson, BNeg, ZIP e ZINB

def smooth_line_plot(x,y):
    x_new = np.linspace(x.min(), x.max(),500)
    f = interp1d(x, y, kind='quadratic')
    y_smooth=f(x_new)
    return x_new, y_smooth

x_new, zinb_theta2_delta2_plogit07_lambda2 = smooth_line_plot(df_zinb.m,
                                                              df_zinb.zinb_theta2_delta2_plogit07_lambda2)

plt.figure(figsize=(15,10))
plt.plot(x_new,lambda_1, linewidth=3, color='#404688FF')
plt.plot(x_new,lambda_2, linewidth=3, color='#2C728EFF')
plt.plot(x_new,lambda_4, linewidth=3, color='#20A486FF')
plt.plot(x_new,bneg_theta2_delta2, linewidth=3, color='#75D054FF')
plt.plot(x_new,bneg_theta3_delta1, linewidth=3, color='#C7E020FF')
plt.plot(x_new,bneg_theta3_delta05, linewidth=3, color='#FDE725FF')
plt.plot(x_new,zip_lambda1_plogit07, linewidth=5, color="#440154FF")
plt.plot(x_new,zinb_theta2_delta2_plogit07_lambda2, linewidth=7, color="red")
plt.xlabel('m', fontsize=20)
plt.ylabel('Probabilidades', fontsize=20)
plt.legend([r'Poisson: $\lambda$ = 1',
            r'Poisson: $\lambda$ = 2',
            r'Poisson: $\lambda$ = 4',
            r'BNeg: $\theta$ = 2 e $\delta$ = 2',
            r'BNeg: $\theta$ = 3 e $\delta$ = 1',
            r'BNeg: $\theta$ = 3 e $\delta$ = 0.5',
            r'ZIP: $\lambda$ = 1 e plogit = 0.7',
            r'ZINB: $\lambda$$_{bneg}$ = 1, plogit = 0.7, $\theta$ = 2 e $\delta$ = 2'],
           fontsize=24)
plt.show


# In[ ]:
##############################################################################
#        ESTIMAÇÃO DO MODELO ZERO-INFLATED BINOMIAL NEGATIVO (ZINB)          #
##############################################################################

#Definição da variável dependente (voltando ao dataset 'df_corruption')
y = df_corruption.violations

#Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_corruption[['staff','post','corruption']]
X1 = sm.add_constant(x1)

#Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_corruption[['corruption']]
X2 = sm.add_constant(x2)

#Se estimarmos o modelo sem dummizar as variáveis categórias, o modelo retorna
#um erro
X1 = pd.get_dummies(X1, columns=['post'], drop_first=True)

#Estimação do modelo ZINB pela função 'ZeroInflatedNegativeBinomialP' do
#pacote 'statsmodels.discrete.count_model'

from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

#Estimação do modelo ZINB
#O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zinb = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2,
                                            inflation='logit').fit()

#Parâmetros do modelo
modelo_zinb.summary()

#O parâmetro 'alpha' apresentado é o inverso do parâmetro 'theta', ou seja,
#o inverso do parâmetro de forma da distribuição Poisson-Gama.
#Como 'alpha' (e da mesma forma 'theta') é estatisticamente diferente de
#zero, podemos afirmar que há superdispersão nos dados (outra forma de
#verificar o fenômeno da superdispersão!)


# In[ ]: Gráfico para comparar os valores previstos x valores reais de
#'violations' pelo modelo ZINB

zinb_predictions = modelo_zinb.predict(X1, exog_infl=X2)
predicted_counts = np.round(zinb_predictions)
actual_counts = df_corruption['violations']

plt.figure(figsize=(15,10))
plt.plot(df_corruption.index, predicted_counts, 'go-',
         color='orange')
plt.plot(df_corruption.index, actual_counts, 'go-',
         color='#440154FF')
plt.xlabel('Observação', fontsize=20)
plt.ylabel('Violações de Trânsito', fontsize=20)
plt.legend(['Valores Previstos pelo ZINB', 'Valores Reais no Dataset'],
           fontsize=20)
plt.show()


# In[ ]: Comparando os modelos BNeg e ZINB

summary_col([modelo_bneg, modelo_zinb], 
            model_names=["BNeg","ZINB"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf),
                'Pseudo-R2':lambda x: "{:.4f}".format(x.pseudo_rsquared()),
        })


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_bneg, modelo_zinb])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson, BNeg, ZIP e
#ZINB

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','ZIP','BNeg','ZINB'],
                      'loglik':[modelo_poisson.llf,
                                modelo_zip.llf,
                                modelo_bneg.llf,
                                modelo_zinb.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ["#440154FF", "#453781FF", "#22A884FF", "orange"]

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimação", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)


# In[ ]: COMPARAÇÕES ENTRE AS PREVISÕES:
#Supondo que considerássemos a estimação ZINB como a mais adequada, qual seria
#a quantidade média esperada de violações de trânsito para um país cujo corpo 
#diplomático seja composto por 23 membros, considerando o período anterior à 
#vigência da lei e cujo índice de corrupção seja igual a 0.5?
   
#Modelo Poisson:
modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['no'],
                                     'corruption':[0.5]}))

#Modelo binomial negativo:
modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                  'post':['no'],
                                  'corruption':[0.5]}))

#Modelo ZIP:
#Obs.: manter a ordem dos parâmetros nos argumentos da função 'predict'
modelo_zip.params

modelo_zip.predict(pd.DataFrame({'const':[1],
                                 'staff':[23],
                                 'corruption':[0.5],
                                 'post_yes':[0]}),
                   exog_infl=pd.DataFrame({'const':[1],
                                           'corruption':[0.5]}))

#Modelo ZINB:
#Obs.: manter a ordem dos parâmetros nos argumentos da função 'predict'
modelo_zinb.params

modelo_zinb.predict(pd.DataFrame({'const':[1],
                                  'staff':[23],
                                  'corruption':[0.5],
                                  'post_yes':[0]}),
                    exog_infl=pd.DataFrame({'const':[1],
                                            'corruption':[0.5]}))

#Qual seria a quantidade média esperada de violações de trânsito para o mesmo
#país, porém agora considerando a vigência da lei?

#Modelo Poisson:
modelo_poisson.predict(pd.DataFrame({'staff':[23],
                                     'post':['yes'],
                                     'corruption':[0.5]}))

#Modelo binomial negativo:
modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                  'post':['yes'],
                                  'corruption':[0.5]}))

#Modelo ZIP:
modelo_zip.predict(pd.DataFrame({'const':[1],
                                 'staff':[23],
                                 'corruption':[0.5],
                                 'post_yes':[1]}),
                   exog_infl=pd.DataFrame({'const':[1],
                                           'corruption':[0.5]}))

#Modelo ZINB:
modelo_zinb.predict(pd.DataFrame({'const':[1],
                                  'staff':[23],
                                  'corruption':[0.5],
                                  'post_yes':[1]}),
                    exog_infl=pd.DataFrame({'const':[1],
                                            'corruption':[0.5]}))


# In[ ]: Adicionando os fitted values dos modelos estimados para fins de
#comparação

df_corruption['fitted_zip'] = modelo_zip.predict(X1, exog_infl=X2)
df_corruption['fitted_zinb'] = modelo_zinb.predict(X1, exog_infl=X2)
df_corruption


# In[75]: Fitted values dos modelos POISSON, BNEG, ZIP e ZINB, considerando,
#para fins didáticos, a variável dependente 'violations' em função apenas da
#variável preditora 'staff'

plt.figure(figsize=(20,10))
sns.relplot(data=df_corruption, x='staff', y='violations',
            ci=False, color='black', height=8)
sns.regplot(data=df_corruption, x='staff', y='fitted_poisson', order=3,
            color='#440154FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_bneg', order=3,
            color='#22A884FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_zip', order=3,
            color='#453781FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_zinb', order=3,
            color='orange')
plt.xlabel('Number of Diplomats (staff)', fontsize=17)
plt.ylabel('Unpaid Parking Violations (violations)', fontsize=17)
plt.legend(['Observado', 'Poisson', 'Fit Poisson', 'CI Poisson',
            'BNeg', 'Fit BNeg', 'CI BNeg',
            'ZIP', 'Fit ZIP', 'CI ZIP',
            'ZINB', 'Fit ZINB', 'CI ZINB'],
           fontsize=14)
plt.show


################################## FIM ######################################