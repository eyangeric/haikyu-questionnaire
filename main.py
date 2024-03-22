from fastapi import FastAPI
from pydantic import BaseModel

class User(BaseModel):
    name: str
    gender: str
    race: str
    q1: tuple[int, int]
    q2: tuple[int, int]
    q3: tuple[int, int]
    q4: tuple[int, int]
    q5: tuple[int, int]
    q6: tuple[int, int]
    q7: tuple[int, int]
    q8: tuple[int, int]
    q9: tuple[int, int]
    q10: tuple[int, int]


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/questions/")
async def create_user(user: User):
    return user