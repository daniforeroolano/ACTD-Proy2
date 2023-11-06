# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:33:44 2023

@author: 57314
"""
from pgmpy . estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy . estimators import BayesianEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
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

# Cambiar los encabezados
nuevos_encabezados = ['G','M','AH','AI','AJ','AF','Z','T','X',"A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "N", "O", "P", "Q", "R", "S", "U", "V", "W", "Y", "AA", "AB", "AC", "AD", "AE", "AG", "Target"]
df.columns = nuevos_encabezados
df_filt=df[["Z","Target","AF","AE","Y","X"]]
X_train, X_test = train_test_split(df_filt,test_size=0.2, random_state=43)



from pgmpy.estimators import HillClimbSearch, K2Score

scoring_method = K2Score ( data =X_train)
esth = HillClimbSearch ( data =X_train)
estimated_modelk = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print(estimated_modelk)
print(estimated_modelk.nodes())
print(estimated_modelk.edges())
print(scoring_method.score(estimated_modelk))

from pgmpy . estimators import BicScore
scoring_method = BicScore ( data =X_train)
esth = HillClimbSearch ( data =X_train)
estimated_modelb = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print(estimated_modelb)
print(estimated_modelb.nodes())
print(estimated_modelb.edges())
print(scoring_method.score(estimated_modelb))

df_filtv2=df[["Z","Target","AE","Y","X"]]
X_train2, X_test2 = train_test_split(df_filtv2,test_size=0.2, random_state=43)
df_filtv3=df[["Z","Target","AE","Y"]]
X_train3, X_test3 = train_test_split(df_filtv3,test_size=0.2, random_state=43)

scoring_method = K2Score ( data =X_train2)
esth = HillClimbSearch ( data =X_train2)
estimated_modelk2 = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print(estimated_modelk2)
print(estimated_modelk2.nodes())
print(estimated_modelk2.edges())
print(scoring_method.score(estimated_modelk2))

from pgmpy . estimators import BicScore
scoring_method = BicScore ( data =X_train3)
esth = HillClimbSearch ( data =X_train3)
estimated_modelb2 = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print(estimated_modelb2)
print(estimated_modelb2.nodes())
print(estimated_modelb2.edges())
print(scoring_method.score(estimated_modelb2))

df_filtv4=df[["Target","AE","Y","X"]]
X_train4, X_test4 = train_test_split(df_filtv4,test_size=0.2, random_state=43)

scoring_method = K2Score ( data =X_train4)
esth = HillClimbSearch ( data =X_train4)
estimated_modelk3 = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print(estimated_modelk3)
print(estimated_modelk3.nodes())
print(estimated_modelk3.edges())
print(scoring_method.score(estimated_modelk3))



from pgmpy . models import BayesianNetwork
from pgmpy . estimators import MaximumLikelihoodEstimator
modelo1= BayesianNetwork(estimated_modelk3)
modelo1.fit(data =X_train4 , estimator = MaximumLikelihoodEstimator)
for i in modelo1.nodes():
    print(modelo1.get_cpds(i))
valido = modelo1.check_model()
print(valido)


new_column_values = []



for index, row in X_test4.iterrows():  
    Y_value = row["Y"]
    X_value = row["X"]
    AE_value = row["AE"]
   
    
    infer = VariableElimination(modelo1)
    result= infer.query(["Target"], evidence={"Y":Y_value,"X":X_value,"AE":AE_value})
    result_values = result.values
    max_prob_index = np.argmax(result.values)
    max_prob_option = modelo1.get_cpds('Target').state_names['Target'][max_prob_index]
              
    new_column_values.append(max_prob_option)
 
X_test4["Estimado"] = new_column_values



df2=X_test4[["Target","Estimado"]]
print(df2.head())

# Agrupar por las columnas y contar las combinaciones únicas
combination_counts = df2.groupby(["Target", "Estimado"]).size().reset_index(name="count")

print(combination_counts)


from pgmpy.readwrite import BIFWriter
print("modelo completo")
# Escribir el modelo en un archivo BIF
writer = BIFWriter(modelo1)
bif_file_path = r'C:\Users\oem\Documents\universidad de los andes\octavo\Analitica computacional para la toma de decisiones\proyecto 2'
# Corrected line to save the BIF file
writer.write_bif(filename=bif_file_path + "/modelo1.bif")