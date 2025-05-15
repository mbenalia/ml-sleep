# train_and_log.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger les données
data = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Feature engineering
data[['Systolic', 'Diastolic']] = data['Blood_Pressure'].str.split('/', expand=True).astype(float)
data.drop(columns=['Blood_Pressure', 'Person ID'], inplace=True)

label_encoder = LabelEncoder()
data['Sleep_Disorder'] = label_encoder.fit_transform(data['Sleep_Disorder'])

X = data.drop('Sleep_Disorder', axis=1)
y = data['Sleep_Disorder']

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
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

# Lancer une expérience MLflow
mlflow.set_experiment("sleep_disorder_classification")

with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Logguer les métriques et le modèle
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    print(f"✅ Modèle loggé avec une précision de {acc:.4f}")
