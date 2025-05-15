from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI()
model = None  # Stockera le modèle globalement


# -------- Schéma d'entrée ----------
class SleepInput(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Heart_Rate: int
    Daily_Steps: int
    Systolic: float
    Diastolic: float


# -------- Récupérer le dernier modèle ----------
def get_latest_model_uri(experiment_name="sleep_disorder_classification"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise Exception("Experiment not found")
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise Exception("No runs found")
    return f"runs:/{runs[0].info.run_id}/model", runs[0].info.run_id


# -------- Charger modèle au démarrage ----------
@app.on_event("startup")
def load_initial_model():
    global model
    try:
        uri, _ = get_latest_model_uri()
        model = mlflow.sklearn.load_model(uri)
    except Exception as e:
        print(f"⚠️ Impossible de charger le modèle au démarrage : {e}")


# -------- Endpoint de prédiction ----------
@app.post("/predict")
def predict(input_data: SleepInput):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        df = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------- Entraînement du modèle ----------
@app.post("/train")
def train():
    global model
    try:
        # Charger les données
        data = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
        data[['Systolic', 'Diastolic']] = data['Blood_Pressure'].str.split('/', expand=True).astype(float)
        data.drop(columns=['Blood_Pressure', 'Person ID'], inplace=True)

        label_encoder = LabelEncoder()
        data['Sleep_Disorder'] = label_encoder.fit_transform(data['Sleep_Disorder'])

        X = data.drop('Sleep_Disorder', axis=1)
        y = data['Sleep_Disorder']

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
        
        mlflow.set_tracking_uri("file:/home/site/wwwroot/mlruns")

        mlflow.set_experiment("sleep_disorder_classification")

        with mlflow.start_run() as run:
            pipeline.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipeline.predict(X_test))

            mlflow.log_param("n_estimators", 200)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            # Recharger le modèle entraîné
            model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")

        return {"message": f"✅ Modèle entraîné avec précision {acc:.4f}", "run_id": run.info.run_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
