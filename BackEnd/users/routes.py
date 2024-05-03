from fastapi import APIRouter, status, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from BackEnd.core.sercurity import ACCESS_TOKEN_EXPIRE_MINUTES
from BackEnd.core.database import get_database
from BackEnd.users.model import PyObjectId, PredictionModel
from BackEnd.users.schemas import CreateUserRequest, User, Token, prediction_input, Prediction_results, LinkItem
from BackEnd.users.services import create_user_account, authenticate_user, get_current_active_user, create_access_token, get_my_model_predictions, get_my_latest_prediction_id
from pymongo import MongoClient
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from datetime import timedelta
from typing import Optional, List
from bson import ObjectId



route = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

token = APIRouter(
    prefix="/login",
    tags=["login"],
    responses={404: {"description": "Not found"}},
)

prediction = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)

history = APIRouter(
    prefix="/get_history",
    tags=["get_history"],
    responses={404: {"description": "Not found"}},
)


@route.post('', status_code=status.HTTP_201_CREATED)
async def created_user(data: CreateUserRequest, db: MongoClient= Depends(get_database)):
    await create_user_account(user_data=data, db=db["user_details"])
    payload={"message": "User account has been successfully created."}
    return JSONResponse(content=payload)

@token.post("", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: MongoClient= Depends(get_database)):
    print("testing1")
    user = authenticate_user(db["user_details"], form_data.username, form_data.password)
    print("testing2")

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username and password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expries_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@route.get("/me", response_model=User)
async def read_user_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@route.get("/me/items")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": 1, "owner": current_user}]

@prediction.post('', status_code=status.HTTP_201_CREATED)
async def prediction_results(data: prediction_input, db: MongoClient= Depends(get_database), current_user: User = Depends(get_current_active_user)):
    result =await get_my_model_predictions(prediction_input=data, db=db["Generated_Lyrics"], user_id=current_user.id)


    # Retrieve the ID of the inserted prediction
    prediction_id = str(result.id)

    payload = {"message": "User Lyrics has been successfully Genrated."}
    return JSONResponse(content=payload)
    # Redirect the client to the individual prediction page with the prediction ID
    # return RedirectResponse(url=f"/prediction/{prediction_id}", status_code=status.HTTP_303_SEE_OTHER)

@prediction.get("/{prediction_id}", status_code=status.HTTP_200_OK, response_model=Prediction_results)
async def get_prediction_by_id(
    prediction_id: str,
    db: MongoClient = Depends(get_database),
    current_user: User = Depends(get_current_active_user),
):

    """
    Retrieve a specific prediction by its ID for the current user.
    """
    # # If prediction_id is None, use get_my_latest_prediction_id to get the latest ID
    # if prediction_id == "Start":
    #     prediction_id = await get_my_latest_prediction_id(db, current_user)
    if prediction_id is None:
        # Return an empty response if there's nothing to retrieve
        return JSONResponse(
            content={"message": "No predictions found", "latest_prediction": {}, "prediction_id_list": []})

    # Use projection to retrieve only Lyrics and GPT_Lyrics fields
    projection = {"Lyrics": 1, "GPT_Lyrics": 1, "Title": 1,
    "Genre": 1,
    "Artist": 1,
    "Number_of_Samples": 1}

    # Retrieve the prediction by ID
    latest_prediction = db["Generated_Lyrics"].find_one(
        {"_id": PyObjectId(prediction_id), "user_id": PyObjectId(current_user.id)},
        projection
    )

    # # Convert to a dictionary and ensure all ObjectId are converted to strings
    # latest_prediction = {
    #     key: (str(value) if isinstance(value, ObjectId) else value)
    #     for key, value in latest_prediction.items()
    # }

    if not latest_prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # prediction = PredictionModel.from_mongo(latest_prediction)

    print(latest_prediction["GPT_Lyrics"])

    # Return a JSONResponse containing both the specific prediction and the list of the last 10 prediction IDs
    return Prediction_results(**latest_prediction)


@history.get("", response_model=List[LinkItem])
async def get_links(current_user: User = Depends(get_current_active_user), db: MongoClient= Depends(get_database)):
    # Retrieve the latest 10 predictions for the current user, returning only '_id' and 'created_at'
    user_predictions = (
        db["Generated_Lyrics"]
        .find(
            {"user_id": PyObjectId(current_user.id)},  # Query to find user-specific predictions
            {"_id": 1, "Title": 1}  # Projection to return only '_id' and 'created_at'
        )
        .sort("created_at", -1)  # Sort by 'created_at' in descending order
        .limit(10)  # Limit to the latest 10 results
    )

    # Convert database results to LinkItem format
    links = [
        {"title": prediction['Title'], "urlSegment": str(prediction['_id'])}
        for prediction in user_predictions
    ]

    return links



# from bson import ObjectId
#
# # Define a function to convert MongoDB ObjectId to a JSON-friendly format
# def objectid_to_str(data):
#     if isinstance(data, dict):
#         # If the data is a dictionary, convert any ObjectId to a string
#         return {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in data.items()}
#     elif isinstance(data, list):
#         # If the data is a list, apply the conversion recursively
#         return [objectid_to_str(item) for item in data]
#     else:
#         # For single values, convert if it's an ObjectId
#         return str(data) if isinstance(data, ObjectId) else data
