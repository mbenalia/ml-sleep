from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
import pytest
import warnings
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Créer une instance du client de test FastAPI
client = TestClient(app)

# Test de l'endpoint /predict
def test_predict():
    # Données d'exemple simulées pour le test
    test_data = {
        "Gender": "Male",
        "Age": 25,
        "Occupation": "Engineer",
        "Sleep_Duration": 7.5,
        "Quality_of_Sleep": 4,
        "Physical_Activity_Level": 3,
        "Stress_Level": 2,
        "BMI_Category": "Normal",
        "Heart_Rate": 70,
        "Daily_Steps": 8000,
        "Systolic": 120.0,
        "Diastolic": 80.0
    }

    # Envoi de la requête POST à l'endpoint /predict
    response = client.post("/predict", json=test_data)

    # Vérification du code de statut et de la réponse
    assert response.status_code == 200
    assert "prediction" in response.json()
