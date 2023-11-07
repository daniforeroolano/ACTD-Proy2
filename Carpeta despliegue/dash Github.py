import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from pgmpy.inference import VariableElimination
import numpy as np
from dash.dependencies import Input, Output, State
from pgmpy.readwrite import BIFReader
import plotly.express as px
#from geopy.geocoders import Nominatim

# Importa tus datos de PostgreSQL
import psycopg2
import pandas as pd
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Conexión a la base de datos
engine = psycopg2.connect(
    dbname="students",
    user="postgres",
    password="********",
    host="dfo-jfm.c2pleeo1r93e.us-east-1.rds.amazonaws.com",
    port='5432'
)

# Consultas SQL
query1 = '''
SELECT
  CASE "Nacionality"
    WHEN 1 THEN 'Portugal'
    WHEN 2 THEN 'Germany'
    WHEN 6 THEN 'Spain'
    WHEN 11 THEN 'Italy'
    WHEN 13 THEN 'Netherlands'
    WHEN 14 THEN 'England'
    WHEN 17 THEN 'Lithuania'
    WHEN 21 THEN 'Angola'
    WHEN 22 THEN 'Cape Verde'
    WHEN 24 THEN 'Guinea'
    WHEN 25 THEN 'Mozambique'
    WHEN 26 THEN 'saint tome and principe'
    WHEN 32 THEN 'Turkey'
    WHEN 41 THEN 'Brazil'
    WHEN 62 THEN 'Romania'
    WHEN 100 THEN 'Moldavia'
    WHEN 101 THEN 'Mexico'
    WHEN 103 THEN 'Ukraine'
    WHEN 105 THEN 'Russia'
    WHEN 108 THEN 'Cuba'
    WHEN 109 THEN 'Colombia'
  END AS "Country",
  COUNT(*) AS "Quantity"
FROM estudiantes2
GROUP BY "Nacionality"
ORDER BY "Quantity" desc;
'''

query2 = '''
SELECT
  CASE "Gender"
    WHEN 1 THEN 'Male'
    else 'Female'
  END AS "Gender",
  COUNT(*) AS "Quantity"
FROM estudiantes2
GROUP BY "Gender"
ORDER BY "Quantity" desc;
'''

query3 = '''
SELECT ROUND((sum(case "Target" when 'Graduate' THEN 1 else 0 end )*100 / count(*)) , 4) as "percentage of graduate"
FROM estudiantes2;
'''

# Ejecuta las consultas y almacena los resultados en DataFrames
df1 = pd.read_sql_query(query1, engine)
df2 = pd.read_sql_query(query2, engine)
df3 = pd.read_sql_query(query3, engine)


# Agrega columnas para latitud y longitud
data = {
    "Lat": [39.662165, -10.333333, 0.326412, 39.326068, 16.000055, 10.722623, 42.638426, 47.287961, 49.487197, 19.43263, -19.302233, -11.877577, 40.233821, 40.420348, 45.985213, 38.959759, 52.531021, 4.099917, 23.013134, 52.243498, 55.35],
    "Lon": [-8.135352, -53.2, 6.730095, -4.837979, -24.008395, -10.708359, 12.674297, 28.567094, 31.271832, -99.133178, 34.914498, 17.569124, -84.409673, -79.116698, 24.685923, 34.924965, -1.264906, -72.908813, -80.832875, 5.634323, 23.75]
}

# Convertir el nuevo DataFrame en un DataFrame de pandas
new_columns_df = pd.DataFrame(data)

# Concatenar los DataFrames (df1 y new_columns_df) a lo largo de las columnas
df1 = pd.concat([df1, new_columns_df], axis=1)


fig = px.choropleth(
    data_frame=df1,  # Reemplaza "df" con el nombre de tu DataFrame
    locations="Country",  # Columna con los nombres de los países
    locationmode="country names",  # Modo de ubicación basado en nombres de países
    color="Quantity",  # Columna con los valores numéricos a visualizar
    color_continuous_scale="Viridis",  # Escala de colores (puedes cambiarla)
    labels={"Quantity": "Quantity"},  # Etiqueta para la leyenda
    title="Students per country"  # Título del mapa
)
# Read model from BIF file
nombre_archivo_bif = 'modelo1.bif'
reader = BIFReader(nombre_archivo_bif )
modelo1 = reader.get_model()
# Print model 
print("bif leido")

# Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
modelo1.check_model()
infer = VariableElimination(modelo1)

#dash
print("inferencia establecida")


########
panel_1_content = html.Div([
    html.H2("Instructions"),
    html.P("This aplication is a tool that predicts the academic success of a student, based on personal, economic and academic atributes."),
    html.P("Follow these instructions:"),
    html.P("1. fill up all the information required"),
    html.P("2. click on the execution boton"),
    html.P("3. then the results will show up"),
    html.P("4. Spaces "'"Curricular units 1st sem (approved)"'" and "'"Curricular units 2nd sem (approved):"'" can be left empty."),
])
panel_2_content = html.Div([
    html.H2("Data record"),
    html.P("This application is a tool that predicts the academic success of a student, based on personal, economic, and academic attributes."),
    html.Div([  # Agregamos un contenedor para organizar los elementos
        html.H6("Curricular units 1st sem (approved):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='Y',  # Identificador único para la lista desplegable
            options=[
                {'label': '(Valor Vacío)', 'value': "None"},
                {'label':'0','value':"0"},
                {'label':'1','value':"1"},
                {'label':'2','value':"2"},
                {'label':'3','value':"3"},
                {'label':'4','value':"4"},
                {'label':'5','value':"5"},
                {'label':'6','value':"6"},
                {'label':'7','value':"7"},
                {'label':'8','value':"8"},
                {'label':'9','value':"9"},
                {'label':'10','value':"10"},
                {'label':'11','value':"11"},
                {'label':'12','value':"12"},
                {'label':'13','value':"13"},
                {'label':'14','value':"14"},
                {'label':'15','value':"15"},
                {'label':'16','value':"16"},
                {'label':'17','value':"17"},
                {'label':'18','value':"18"},
                {'label':'19','value':"19"},
                {'label':'20','value':"20"},
                {'label':'21','value':"21"},
                {'label':'26','value':"26"},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            #value=6,  # Opción preseleccionada
            style={'display': 'inline-block','width': '200px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 2nd sem (approved):", style={'display': 'inline-block', 'margin-right': '15px', 'margin-left':'100px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='AE',  # Identificador único para la lista desplegable
            options=[
                {'label': '(Valor Vacío)', 'value': "None"},
                {'label':'0','value':"0"},
                {'label':'1','value':"1"},
                {'label':'2','value':"2"},
                {'label':'3','value':"3"},
                {'label':'4','value':"4"},
                {'label':'5','value':"5"},
                {'label':'6','value':"6"},
                {'label':'7','value':"7"},
                {'label':'8','value':"8"},
                {'label':'9','value':"9"},
                {'label':'10','value':"10"},
                {'label':'11','value':"11"},
                {'label':'12','value':"12"},
                {'label':'13','value':"13"},
                {'label':'14','value':"14"},
                {'label':'16','value':"16"},
                {'label':'17','value':"17"},
                {'label':'18','value':"18"},
                {'label':'19','value':"19"},
                {'label':'20','value':"20"},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            #value=5,  # Opción preseleccionada
            style={'display': 'inline-block','width': '200px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 1st sem (evaluations):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='X',  # Identificador único para la lista desplegable
            options=[
                {'label':'Less than 12','value':"0"},
                {'label':'Between 12 and 23','value':"1"},
                {'label':'Between 23 and 34','value':"2"},
                {'label':'More than 34','value':"3"},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=0,  # Opción preseleccionada
            style={'display': 'inline-block','width': '200px'}  # Establecemos el estilo inline-block
        ),
        
    ], className='panel'),
])

panel_3_content = html.Div([
    html.H2("Resultados"),
    html.P("This aplication is a tool that predicts the academic success of a student, based on personal, economic and academic atributes."),
])
panel_4_content = html.Div([
    html.H1("About the data"),
    
    # Mapa de calor
    dcc.Graph(
        id='world-map',
        figure=fig,  # Esto es la figura que creaste previamente
        style={'width': '1200px', 'height': '800px'}
    ),
    
    # Gráfico de barras
    html.H2("Male and female proportion:", style={'font-size': '30px'}),
    dcc.Graph(
        id='bar-chart',
        figure=px.bar(df2, x="Gender", y="Quantity", labels={"Quantity": "Quantity"}),
        style={'width': '800px', 'height': '600px'}
    ),
    
    # Texto grande
    html.Div([
        html.H2("Percentage of graduate:", style={'font-size': '50px'}),
        html.P(f"{df3.iloc[0]['percentage of graduate']}%", style={'font-size': '60px'}),
    ]),
])
print("paneles creados")

# Diseño de la aplicación con los paneles
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Pestaña 1', children=[
            html.H1(children='Academic success predictor'),  # Encabezado principal

            # Contenedor de Paneles
            html.Div([
                panel_1_content,  # Agrega aquí otros paneles si es necesario
                panel_2_content,
            ], className="panel-container"),
            # Agrega tus componentes aquí, como el botón y cualquier otra interfaz que desees
            html.Button('Realizar inferencia', id='inferencia-button'),
            # Agrega un espacio para mostrar el resultado
            html.Div([
                html.H2("Results"),
                html.P(id='resultado-inferencia'),
            ])
        ]),
        dcc.Tab(label='Pestaña 2', children=[
            panel_4_content,
        ]),
    ])
])
print("layout creado")

@app.callback(
    Output('resultado-inferencia', 'children'),
    [Input('inferencia-button', 'n_clicks')],
    [State('Y', 'value'),
     State('AE', 'value'),
     State('X', 'value')],
    allow_duplicate=True
)
def realizar_inferencia(n_clicks, Y, AE, X):
    if n_clicks is None:
        return "Esperando a que se haga clic en el botón..."

    try:
        if str(Y)=='None' and str(AE) =='None':
            result = infer.query(
                variables=["Target"],
                evidence={"X": str(X)}
            )
        elif str(Y)=='None':
            result = infer.query(
                variables=["Target"],
                evidence={"AE": str(AE), "X": str(X)}
            )
        elif str(AE) =='None':
            result = infer.query(
                variables=["Target"],
                evidence={"Y": str(Y), "X": str(X)}
            )
        else:
            result = infer.query(
                variables=["Target"],
                evidence={"Y": str(Y), "AE": str(AE), "X": str(X)}
            )
        result_values = result.values
        max_prob_index = np.argmax(result.values)
        max_prob_option = modelo1.get_cpds('Target').state_names['Target'][max_prob_index]
        
        # Crear la gráfica de barras
        opciones = modelo1.get_cpds('Target').state_names['Target']
        probabilidades = result_values.tolist()
        
        # Creating the bar chart
        grafico = {
            'data': [{'x':opciones , 'y': probabilidades, 'type': 'bar'}],
            'layout': {'title': 'Probabilidad de cada opción', 'xaxis': {'title': 'Opciones'}, 'yaxis': {'title': 'Probabilidad'}}
        }
        
        return [
            html.Div(f"La opción más probable es: {opciones[np.argmax(probabilidades)]}"),
            dcc.Graph(figure=grafico),
        ]

    except Exception as e:
        return f"Error durante la inferencia: {str(e)}"

if __name__ == '__main__':
   app . run_server ( host = "0.0.0.0", debug = True )
#if __name__ == '__main__':
#    app.run_server(debug=True)
print("fin")
