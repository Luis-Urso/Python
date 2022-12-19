## Exemplo de como usar o PANDAS para 
## ler arquivos CSV e transformar em 
## Dataset

import pandas as pd

f_path="C:/Users/WB02554/Downloads/"
df = pd.read_csv(f_path+"Hire.csv")

df_xls = pd.read_excel(f_path+"LATAM - Hire Details.xlsx",index_col=0)

print(df_xls)


#print(df["Worker"])

df['Hire Date']= pd.to_datetime(df['Hire Date'])

print(df.info())

newdf = df.sort_values(by='Legal Name - First Name')

print(newdf["Legal Name - First Name"])
