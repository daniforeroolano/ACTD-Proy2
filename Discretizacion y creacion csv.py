# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:28:41 2023

@author: 57314
"""

from pgmpy . estimators import PC
import pandas as pd
from ucimlrepo import fetch_ucirepo 

#importo los datos
dataset = fetch_ucirepo(id=697)
#Separo para que no me quede una tabla
data = dataset['data']
targets = dataset['data']['targets']

# Convierto a dataframe
df_data = pd.DataFrame(data['features'])

# meto el target al data frame
df = pd.concat([df_data, pd.DataFrame({'Target': targets['Target']})], axis=1)

# Especificar las variables que queremos discretizar
variables_a_discretizar = ['Previous qualification (grade)', 'Admission grade','Unemployment rate','Inflation rate','GDP',"Curricular units 2nd sem (grade)","Curricular units 1st sem (grade)","Age at enrollment","Curricular units 1st sem (evaluations)"]

# Especificicar el número de intervalos deseados para cada variable
num_bins = 4

# Crear un nuevo DataFrame para almacenar las variables originales y discretizadas
df_nuevo = pd.DataFrame()

# Discretizar las variables seleccionadas y añadirlas al nuevo DataFrame
for col in variables_a_discretizar:
    df[col+'_bin'] = pd.cut(df[col], bins=num_bins, labels=False)
    df_nuevo[col+'_bin'] = df[col+'_bin']

# Añadir las variables que no se han discretizado al nuevo DataFrame
for col in df.columns:
    if col not in variables_a_discretizar and 'bin' not in col:
        df_nuevo[col] = df[col]

# Obtener los límites de los intervalos para cada variable
bin_limits = {col: pd.cut(df[col], bins=num_bins).unique() for col in variables_a_discretizar}

# Imprimir el nuevo DataFrame con las variables originales y discretizadas
print(df_nuevo)
print("Bin limits:")
print(bin_limits)

df=df_nuevo

ruta_archivo = r'C:\Users\57314\Documents\universidad\Analitica\proyecto 2\mi_archivo.csv'

df.to_csv(ruta_archivo, index=False)

#