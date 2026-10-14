from fastapi import FastAPI

app = FastAPI()


@app.get("/item/{id}")
def item(id: str):
    eval(id)
