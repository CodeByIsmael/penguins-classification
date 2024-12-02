# Penguins Classification Project

## Descripción
Este proyecto implementa un sistema de clasificación para predecir la especie de un pingüino en base a características físicas y geográficas. Se utilizaron los siguientes modelos de aprendizaje automático:

- **Regresión Logística**
- **Máquinas de Soporte Vectorial (SVM)**
- **Árboles de Decisión**
- **K Vecinos Más Cercanos (KNN)**

El proyecto incluye:
1. Preprocesamiento de datos
2. Entrenamiento y serialización de los modelos
3. Implementación de una API REST con Flask para predicciones
4. Un cliente que realiza peticiones a los servicios de predicción

## Estructura del proyecto

```plaintext
penguins-classification/
├── dataset/                       # Dataset original
│   ├── penguins_size.csv
├── models/                     # Modelos serializados
│   ├── log_reg_model.pkl
│   ├── svm_model.pkl
│   ├── dt_model.pkl
│   ├── knn_model.pkl
├── notebooks/                  # Notebooks Jupyter para exploración y cliente
│   ├── penguins_classification.ipynb
│   ├── client.ipynb
├── penguins_classification/               # Aplicación Flask
│   ├── penguin_app.py
├── pyproject.toml              # Configuración de dependencias con Poetry
├── README.md                   # Este archivo
```

## Instalación

1. **Clona este repositorio:**

    ```bash
    git clone https://github.com/tu-usuario/penguins-classification.git
    cd penguins-classification
    ```

2. **Instala dependencias con Poetry:**

    ```bash
    poetry install
    ```

3. **Activa el entorno virtual:**

    ```bash
    poetry shell
    ```

## Uso

### 1. Entrenamiento de los modelos
- Utiliza el notebook `penguins_classification.ipynb` en la carpeta `notebooks` para entrenar y serializar los modelos.
- Los modelos se guardarán en la carpeta `models`.

### 2. Inicia el servidor Flask
- Ve a la carpeta `penguins_app` y ejecuta:

    ```bash
    python app.py
    ```

- Esto levantará el servidor en `http://127.0.0.1:8000`.

### 3. Cliente para predicciones
- Utiliza el notebook `client.ipynb` para realizar peticiones a los endpoints del servidor Flask y obtener predicciones.

### 4. Endpoints disponibles
#### `/predict/log_reg`  
Predicciones con el modelo de Regresión Logística.
#### `/predict/svm`  
Predicciones con el modelo SVM.  
#### `/predict/dt`  
Predicciones con el modelo de Árboles de Decisión.  
#### `/predict/knn`  
Predicciones con el modelo KNN.

### Ejemplo de petición
```python
import requests

data = {
    "bill_length_mm": 50.4,
    "bill_depth_mm": 15.6,
    "flipper_length_mm": 222.0,
    "body_mass_g": 5750.0,
    "island": "Biscoe",
    "sex": "Male"
}

response = requests.post("http://127.0.0.1:8000/predict/log_reg", json=data)
print(response.json())
