print("ENTRO A VALIDAR.PY")

import mlflow
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys

THRESHOLD = 0.8

# --- Configurar MLflow igual que en train.py ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
mlflow.set_tracking_uri("file://" + os.path.abspath(mlruns_dir))

#mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
# --- Cargar dataset ---
print("--- Debug: Cargando dataset categorias Vinos Rojos---")

### carga los datos
datos = pd.read_csv("data/raw/winequality-red.csv", sep=";")
X = datos.drop("quality", axis=1)

### convirtiendo diferentes clases a dos para hacer clasificación binaria
y = datos["quality"].apply(lambda x: 1 if x >= 7 else 0) 

# partiendo el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow local
# --- Cargar modelo desde MLflow ---
print("--- Debug: Cargando modelo desde MLflow ---")

try:
    experiment_name = "Clasificacion-vino"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise Exception(f"Experimento '{experiment_name}' no encontrado")
    
    # Obtener el último run
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    
    if runs.empty:
        raise Exception("No se encontraron corridas en el experimento")
    
    run_id = runs.iloc[0].run_id

    model_uri = f"runs:/{run_id}/modelo_random_forest"
    print(f"--- Debug: Cargando modelo desde URI: {model_uri} ---")
    
    model = mlflow.sklearn.load_model(model_uri)
    print("--- Debug: Modelo cargado exitosamente desde MLflow ---")

except Exception as e:
    print(f"--- ERROR al cargar modelo desde MLflow: {str(e)} {experiment}---")
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    print(os.listdir(os.getcwd()))
    sys.exit(1)

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_test)
    print(f"🔍 Predicciones: {y_pred[:5]}")


    prec = accuracy_score(y_test,y_pred)
    f1   = f1_score(y_test,y_pred)

    print(f"📊 Precisión: {prec:.4f} | F1-score: {f1:.4f} (umbral: {THRESHOLD})")

    # Validación
    if prec >= THRESHOLD:
        print("El modelo cumple los criterios de calidad.")
        sys.exit(0)
    else:
        print("El modelo no cumple el umbral. Deteniendo pipeline.")
        sys.exit(1)

except Exception as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    if hasattr(model, 'n_features_in_'):
        print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)