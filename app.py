from flask import Flask, render_template, request
import joblib
import os
import numpy as np

# Crear una aplicación Flask
app = Flask(__name__)

# Obtener la ruta del directorio actual
current_directory = os.path.dirname(__file__)

# Ruta completa del modelo
model_path = os.path.join(current_directory, 'models', 'modelo_arbol.pkl')

# Cargar el modelo entrenado
model = joblib.load(model_path)  # Cargar el modelo previamente guardado

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'

# Definir la ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperatura = int(request.form['Temp'])
    humedad = int(request.form['Hum'])
    ph = int(request.form['PH'])
    precipitacion = int(request.form['Preci'])
    
    # Valores futuros
    pred_probabilities = np.array([[N, P, K, temperatura, humedad, ph, precipitacion]])
    
    # Realiza predicciones en las nuevas muestras utilizando el modelo de árbol de decisión
    prediccion = model.predict(pred_probabilities)

    # prediccion contiene las etiquetas de clase predichas para las nuevas muestras
    mensaje = "La Clasificación de la etiqueta es:  "
    mensaje+=prediccion[0]
    
    # Renderizar la plantilla 'result.html' y pasar el mensaje a la plantilla
    return render_template('result.html', pred=mensaje)

# Iniciar la aplicación si este script es el punto de entrada
if __name__ == '__main__':
    app.run(debug=True)  # Iniciar la aplicación Flask