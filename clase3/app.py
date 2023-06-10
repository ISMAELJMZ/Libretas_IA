import uvicorn 
from fastapi import FastAPI
from BankNotes import BankNotes 
import pickle


app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return{"mensaje":"hola, bienvenidio al modelo"}

@app.get("/Bienvenida")
def fun_nombre(name:str):
    return{"hola bienvenido": f"{name}"}

@app.post("/predict")
def predict_banknote(data:BankNotes):
    data = data.dict()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy =  data["entropy"]
    
    prediction = classifier.predict([[variance, skewness,curtosis, entropy]])

    if(prediction[0]>0.5):
        prediction = "nota falsa "
    else: 
        prediction = "es nota de banco"
    return {"prediction":prediction}

if __name__== "__main__":
    uvicorn.run(app, host="127.0.0.2", port=8000)

    #levanatr servicio en la carptea de clase 3
    # uvicorn app:app --reload en este ejemplo es app ya que el archivo principal es app 
    
    # cargamos el conjunto de datos csv dentro de la carpeta 3
    # tambien instalar sk learn pip install scikit-learn

    