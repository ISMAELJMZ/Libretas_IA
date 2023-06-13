import uvicorn 
from fastapi import FastAPI
import pickle
from ParametrosSVM import parametrosSVM


app = FastAPI()
pickle_in = open("modelo_svm.pkl","rb")
modelo_svm= pickle.load(pickle_in)

from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    
    resultado = modelo_svm.predict(data)
    return {"prediction": resultado}

@app.post("/predict")
def predict_svm(data:parametrosSVM):
    data = data.dict()
    Pregnancies = data["Pregnancies"]
    Glucose = data["Glucosa"]
    BloodPressure = data["Presion sanguinea"]
    SkinThickness = data["piel"]
    Insulin = data["insulina"]
    BMI = data["masa corporal"]
    DiabetesPedigreeFunction = data["diabetes"]
    Age = data["edad"]    
    
    prediction = modelo_svm.predict([[Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age]])

if __name__== "__main__":
    uvicorn.run(app, host="127.0.0.2", port=8000)
