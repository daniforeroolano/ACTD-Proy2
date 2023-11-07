# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:23:25 2023

@author: 57314
"""

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy . estimators import BayesianEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt

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

modelo1=BayesianNetwork([ ("Z", "Target"),("AF", "Target"),("AE", "Target"),("Y", "Target"),("Z", "AF"),("AE", "AF"),("X", "AF")])
X = df[["Z","Target","AF","AE","Y","X"]]
#V
#AB
#-R
#-T
#-Y
#-AE
#-I
#-J
X_train, X_test = train_test_split(X,test_size=0.2, random_state=43)
#modelo1.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 
estimador_bayesiano = BayesianEstimator(model=modelo1, data=X_train)
# Estimar las CPDs (tablas de probabilidades condicionales) para todas las variables
cpd_Z = estimador_bayesiano.estimate_cpd('Z')
cpd_Target = estimador_bayesiano.estimate_cpd('Target')
cpd_AF = estimador_bayesiano.estimate_cpd('AF')
cpd_AE = estimador_bayesiano.estimate_cpd('AE')
cpd_Y = estimador_bayesiano.estimate_cpd('Y')
cpd_X = estimador_bayesiano.estimate_cpd('X')


# Añadir las CPDs estimadas al modelo
modelo1.add_cpds(cpd_Z, cpd_Target, cpd_AF, cpd_AE, cpd_Y, cpd_X)

# Verificar si el modelo es válido
valido = modelo1.check_model()
print(valido)


new_column_values = []
# Inicializa listas para almacenar las probabilidades y las etiquetas reales
y_true = []


for index, row in X_test.iterrows():  
    Z_value = row["Z"]
    AF_value = row["AF"]
    Y_value = row["Y"]
    X_value = row["X"]
    AE_value = row["AE"]
   
    
    infer = VariableElimination(modelo1)
    result= infer.query(["Target"], evidence={"Z": Z_value, "AF": AF_value, "Y":Y_value,"X":X_value,"AE":AE_value})
    result_values = result.values
    max_prob_index = np.argmax(result.values)
    max_prob_option = modelo1.get_cpds('Target').state_names['Target'][max_prob_index]
              
    new_column_values.append(max_prob_option)
    y_true.append(row["Target"])
 
X_test["Estimado Target"] = new_column_values
print(X_test["Estimado Target"])


df2=X_test[["Target","Estimado Target"]]
print(df2.head())

# Agrupar por las columnas y contar las combinaciones únicas
combination_counts = df2.groupby(["Target", "Estimado Target"]).size().reset_index(name="count")

print(combination_counts)

print(len(X_test))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, new_column_values)  # y_true son las etiquetas reales, y_pred son las etiquetas predichas
print(f"Exactitud Global (Accuracy): {accuracy}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_true, new_column_values)  # y_true son las etiquetas reales, y_pred son las etiquetas predichas
class_names = ["Dropout", "Enrolled", "Graduate"]
# Crea un gráfico de calor para visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# Agrega etiquetas a los ejes
plt.xlabel('Clases Predichas')
plt.ylabel('Clases Reales')
plt.title('Matriz de Confusión Multiclase')

plt.show()

from sklearn.metrics import f1_score

f1_macro = f1_score(y_true, new_column_values, average='macro')
f1_micro = f1_score(y_true, new_column_values, average='micro')

print(f"F1-Score Macro: {f1_macro}")
print(f"F1-Score Micro: {f1_micro}")

from sklearn.metrics import classification_report

report = classification_report(y_true, new_column_values, target_names=class_names)  # class_names son los nombres de las clases
print("Reporte de Clasificación:")
print(report)


#modelo 2
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy . estimators import BayesianEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import matplotlib.pyplot as plt
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
modelo1= BayesianNetwork(estimated_modelk3)
modelo1.fit(data =X_train4 , estimator = MaximumLikelihoodEstimator)
for i in modelo1.nodes():
    print(modelo1.get_cpds(i))
valido = modelo1.check_model()
print(valido)
print(modelo1)

new_column_values = []
y_true = []


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
    y_true.append(row["Target"])
X_test4["Estimado"] = new_column_values



df2=X_test4[["Target","Estimado"]]
print(df2.head())

# Agrupar por las columnas y contar las combinaciones únicas
combination_counts = df2.groupby(["Target", "Estimado"]).size().reset_index(name="count")

print(combination_counts)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, new_column_values)  # y_true son las etiquetas reales, y_pred son las etiquetas predichas
print(f"Exactitud Global (Accuracy): {accuracy}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_true, new_column_values)  # y_true son las etiquetas reales, y_pred son las etiquetas predichas
class_names = ["Dropout", "Enrolled", "Graduate"]
# Crea un gráfico de calor para visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# Agrega etiquetas a los ejes
plt.xlabel('Clases Predichas')
plt.ylabel('Clases Reales')
plt.title('Matriz de Confusión Multiclase')

plt.show()

from sklearn.metrics import f1_score

f1_macro = f1_score(y_true, new_column_values, average='macro')
f1_micro = f1_score(y_true, new_column_values, average='micro')

print(f"F1-Score Macro: {f1_macro}")
print(f"F1-Score Micro: {f1_micro}")

from sklearn.metrics import classification_report

report = classification_report(y_true, new_column_values, target_names=class_names)  # class_names son los nombres de las clases
print("Reporte de Clasificación:")
print(report)
infer = VariableElimination(modelo1)
result= infer.query(["Target"], evidence={"X":0})
result_values = result.values
max_prob_index = np.argmax(result.values)
max_prob_option = modelo1.get_cpds('Target').state_names['Target'][max_prob_index]
print(result_values)
