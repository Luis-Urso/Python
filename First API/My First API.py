## First API Projec
## by Luis A. Urso
##
## Required:
## pip install fastapi
## pip install uvicorn
##
## to test, access: localhost:8000/my-first-api
##

from fastapi import FastAPI

app = FastAPI()

@app.get("/my-first-api")
def hello(name: str):
  return {'Hello ' + name + '!'} 
