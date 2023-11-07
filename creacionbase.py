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
df.columns = df.columns.str.replace(r'[ ()/\'"]', '_')
import pandas as pd

nuevo_df = df.head(5)

#df=nuevo_df
# Carga tu DataFrame (df_nuevo) desde tu fuente de datos

# Establece el nombre de la tabla
table_name = 'estudiantes2'

# Crear la sentencia SQL para la creación de la tabla
create_table_sql = f"CREATE TABLE {table_name} ("

# Recorre las columnas del DataFrame y genera las columnas de la tabla
for column_name, data_type in zip(df.columns, df.dtypes):
    # Reemplaza caracteres no válidos en los nombres de columna
    formatted_column_name = column_name.replace('(', '_').replace(')', '_').replace(' ', '_').replace('/', '_').replace("'", '_')
    
    if data_type == 'object':
        data_type = 'TEXT'
    elif data_type == 'int64':
        data_type = 'INTEGER'
    elif data_type == 'float64':
        data_type = 'REAL'
    create_table_sql += f'"{formatted_column_name}" {data_type}, '  # Agrega comillas dobles a los nombres de columna

# Elimina la última coma y agrega el paréntesis de cierre
create_table_sql = create_table_sql.rstrip(', ') + ");"

# Ruta donde se creará el archivo SQL
sql_file_path = 'C:\\Users\\oem\\Documents\\universidad de los andes\\octavo\\Analitica computacional para la toma de decisiones\\proyecto 2\\base_datos\\estudiantes.sql'

# Guarda la sentencia SQL de creación de la tabla en un archivo
with open(sql_file_path, 'w') as sql_file:
    sql_file.write(create_table_sql)

# Iterar a través de las filas de datos en el DataFrame
for index, row in df.iterrows():
    # Generar un comando SQL INSERT para cada fila
    formatted_columns = ', '.join(map(lambda x: x.replace('(', '_').replace(')', '_').replace(' ', '_').replace('/', '_').replace("'", '_'), row.index))
    formatted_columns = ', '.join([f'"{col}"' for col in formatted_columns.split(', ')])
    values = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
    insert_sql = f"INSERT INTO {table_name} ({formatted_columns}) VALUES ({values});\n"
    # Escribir el comando SQL INSERT en el archivo
    with open(sql_file_path, 'a') as sql_file:
        sql_file.write(insert_sql)

