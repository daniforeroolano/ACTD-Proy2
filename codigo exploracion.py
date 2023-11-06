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

# Exploracion inicial
print("columns")
print(df_nuevo.columns) 
print("size")
print(df_nuevo.size) 
print("info")
print(df_nuevo.info())
print("shape")
print(df_nuevo.shape)
print("describe")
print(df_nuevo.describe)
print("nulos")
print(df_nuevo.isnull().sum())




print("Grafico del target")
# Calcular la frecuencia de cada valor en la columna "target"
frecuencia_target = df_nuevo['Target'].value_counts()

# Crear el gráfico de frecuencia
plt.bar(frecuencia_target.index, frecuencia_target.values)
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Frecuencia de la columna "target"')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mayor legibilidad

# Mostrar el gráfico
plt.show()

#Comparar el comportamiento de cada uno de las variables contra la objetivo
for feature in df_nuevo.columns:
    if feature != 'Curricular units 2nd sem (grade)_bin':
        plt.figure(figsize=(10, 6))
        df_nuevo.boxplot(column=feature, by='Curricular units 2nd sem (grade)_bin', ax=plt.gca())
        plt.title(f'Boxplot of {feature} by Curricular units 2nd sem (grade)_bin')
        plt.suptitle('')  # Suppress the default title
        plt.xlabel('Curricular units 2nd sem (grade)_bin')
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

