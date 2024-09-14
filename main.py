import uvicorn
from fastapi import FastAPI, Body
from joblib import load

app = FastAPI(title="Predict Titanic Survival",
              description="API for Predict Titanic Survival",
              version="1.0")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

@app.get("/")
async def read_root():
    return {"message": "Predict Titanic Survival!"}

@app.post('/prediction', tags=["predictions"])
async def get_prediction(Pclass: int,   #Passenger Class (1 = 1st class หรู, 2 = 2nd class, 3 = 3rd class ชั้นประหยัด)
                        Sex: int,       #Gender (0 = female, 1 = male)
                        Age: float,     #Age of Passenger
                        SibSp: int,     #Number of Siblings/Spouses Aboard (จำนวนพี่น้อง/คู่สมรสบนเรือ)
                        Parch: int,     #Number of Parents/Children Aboard (จำนวนพ่อแม่/ลูกบนเรือ)
                        Fare: float,    #Fare Paid for the Ticket (ค่าโดยสาร)
                        Embarked: int): #Port of Embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton) (ท่าเรือที่ผู้โดยสารขึ้นเรือ)

    # Load model from model_knn.pkl
    model = load('d:/FinalML/FastAPI/app/model_knn.pkl')

    # Collect conditions as input to the model
    conditions = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]

    # Make prediction
    prediction = model.predict([conditions]).tolist()

    # Convert numerical prediction to human-readable format
    if prediction[0] == 1:
        result = "Survived"
    else:
        result = "Not Survived"

    return {"prediction": result}

