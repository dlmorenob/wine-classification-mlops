print("Entra a archivo de entrenamiento")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

### carga los datos
datos = pd.read_csv("data/raw/winequality-red.csv", sep=";")
X = datos.drop("quality", axis=1)

### convirtiendo diferentes clases a dos para hacer calsificación binaria
y = datos["quality"].apply(lambda x: 1 if x >= 7 else 0) 

# partiendo el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

workspace_dir = os.getcwd() 
mlruns_dir = os.path.join(workspace_dir, "mlruns")

# MLflow local
#tracking_uri = "http://127.0.0.1:8089"  
## MLflow remoto
tracking_uri = "file://" + os.path.abspath(mlruns_dir)

mlflow.set_tracking_uri(tracking_uri)  

mlflow.set_experiment("Clasificacion de vino rojo")


with mlflow.start_run():
    # registro de hiper parametros
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # entrenado el modelo
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predict and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1
    })

    print(f"Precisión: {accuracy} - f1 score: {f1}")

    #ahora registramos el modelo 
    mlflow.sklearn.log_model(model, "modelo_random_forest")

    # Guardando el modelo localmente
    joblib.dump(model, "models/model.joblib")

    print(f"Logged run: {mlflow.active_run().info.run_id}")