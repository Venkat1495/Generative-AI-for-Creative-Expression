from fastapi import FastAPI
from BackEnd.users.routes import route as user_route
from BackEnd.users.routes import token as login
from BackEnd.users.routes import prediction as prediction
from BackEnd.users.routes import history as history
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(user_route)
app.include_router(login)
app.include_router(prediction)
app.include_router(history)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
