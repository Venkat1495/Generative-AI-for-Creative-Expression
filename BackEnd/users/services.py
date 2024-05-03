from fastapi import HTTPException, Depends, status
from pymongo.collection import Collection
from pymongo import MongoClient
from Generate import generate_song
from BackEnd.users.model import UserModel, PredictionModel, PyObjectId
from BackEnd.users.schemas import CreateUserRequest, UserInDB, TokenData, prediction_input, User
from BackEnd.core.sercurity import get_password_hash, verify_password, SECRET_KEY, ALGORITHM, openai
from BackEnd.core.database import get_database
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer


oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def create_user_account(user_data: CreateUserRequest, db: Collection):
    print(user_data)
    # Check if the email already exists
    if db.find_one({"email": user_data.email}):
        raise HTTPException(status_code=422, detail="Email is already registered with us.")

    hashed_password = get_password_hash(user_data.password)

    # Create new user document
    new_user_data = UserModel(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        username=user_data.email,
        email=user_data.email,
        password=hashed_password,
        is_active=False,
        is_verified=False,
        registered_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
    ).dict(by_alias=True, exclude={"id"})

    # Insert the new user into the database
    result = db.insert_one(dict(new_user_data))

    # Retrieve the new user's data, including the generated ID
    new_user = db.find_one({"_id": result.inserted_id})

    # Return a Pydantic model of the new user
    return UserModel(**new_user)


def get_user(username: str, db: Collection):

    if db.find_one({"email": username}):
        user_data = db.find_one({"email": username})
        return UserModel(**user_data)

def authenticate_user(db, username: str, password: str):
    user = get_user(username, db)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False

    return user

def create_access_token(data: dict, expries_delta: timedelta or None = None):
    to_encode = data.copy()
    if expries_delta:
        expire = datetime.utcnow() + expries_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth_2_scheme), db: MongoClient= Depends(get_database)):

    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception

        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception

    user = get_user(username= token_data.username, db=db["user_details"])

    if user is None:
        raise credential_exception
    return user


async def get_current_active_user(current_user: UserModel = Depends(get_current_user)):
    if current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive User")
    return current_user


async def get_my_model_predictions(prediction_input: prediction_input, db: Collection, user_id: str):
    input = f"Title: {prediction_input.Title}; Tag: {prediction_input.Genre}; Artist: {prediction_input.Artist};\n\n" + "Lyrics: \n"
    Lyrics = generate_song(seed_text=input, num_samples=prediction_input.Number_of_Samples)
    GPT_Lyrics = get_gpt_predictions(prediction_input)
    # latest_prediction = {
    #     key: (str(value) if isinstance(value) else value)
    #     for key, value in GPT_Lyrics.items()
    # }
    print(GPT_Lyrics)
    new_lyrics_data = PredictionModel(
        Title=prediction_input.Title,
        Genre=prediction_input.Genre,
        Artist=prediction_input.Artist,
        Number_of_Samples=prediction_input.Number_of_Samples,
        Lyrics=Lyrics,
        GPT_Lyrics = GPT_Lyrics,
        user_id=PyObjectId(user_id)
    ).dict(by_alias=True, exclude={"id"})

    # Insert the new Lyrics into the database
    result = db.insert_one(dict(new_lyrics_data))

    # Retrieve the new user's data, including the generated ID
    new_lyric = db.find_one({"_id": result.inserted_id})

    # Return a Pydantic model of the new user
    return PredictionModel(**new_lyric)


async def get_my_latest_prediction_id(db: Collection, current_user: User):
    # Retrieve the latest prediction for the current user based on 'created_at'
    latest_prediction= (
        db["Generated_Lyrics"]
        .find({"user_id": PyObjectId(current_user.id)})
        .sort("created_at", -1)
        .limit(1)
        .next()
                            )
    return latest_prediction["_id"] if latest_prediction else None

def get_gpt_predictions(prediction_input):

    input = f"Title: {prediction_input.Title}; Tag: {prediction_input.Genre}; Artist: {prediction_input.Artist}; Number of samples you have to generate  : {prediction_input.Number_of_Samples}"

    print("working")

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "I want your help in the Generating new lyrics bases on the fields i submit. "
            },
            {
                "role": "user",
                "content": "I want your help in the Generating new lyrics bases on the fields i submit. "
            },
            {
                "role": "assistant",
                "content": "Sure. Please share the field you want me use to generate new song lyrics."
            },
            {
                "role": "user",
                "content": "Here is the below sample and from next i will be just sharing the below fields.\n\nTitle: Everybody Needs To Be Somebodys Friend; Tag: pop; Artist: Rustic Overtones; Number of samples you have to generate  : 5\n"
            },
            {
                "role": "assistant",
                "content": "Here are five sample lyrics based on the fields you provided:\n\n1. (Verse 1)\nIn this big wide world, we're all searching for a hand to lend\nA friend to lean on, someone to comprehend\nThrough ups and downs, we stand strong until the end\nBecause everybody needs to be somebody's friend\n\n(Chorus)\nLet's hold each other tight, and never let go\nTogether we can conquer any high or low\nIn this pop melody, our friendships will transcend\nBecause everybody needs to be somebody's friend\n\n2. (Verse 1)\nIn a world so vast, we often feel alone\nBut in this pop realm, our spirits are not disowned\nThrough melodies and harmonies, we find our own melody\nBecause Rustic Overtones reminds us all to be somebody's friend\n\n(Chorus)\nWe'll dance to the rhythm of unity and love\nEmbracing differences, high-fiving from above\nIn this pop masterpiece, let our friendships ascend\nBecause everybody needs to be somebody's friend\n\n3. (Verse 1)\nRustic Overtones invites us to comprehend\nThat friendships are the ties that never truly end\nOn this pop journey, let's build connections that will mend\nBecause everybody needs to be somebody's friend\n\n(Chorus)\nThrough catchy tunes and humming along\nLet's spread kindness, for it can never be wrong\nIn this pop symphony, our bonds we'll transcend\nBecause everybody needs to be somebody's friend\n\n4. (Verse 1)\nWhen darkness falls and we feel lost in the night\nRustic Overtones reminds us, \"It's gonna be alright\"\nThrough pop rhythms, let our friendships ignite\nBecause everybody needs to be somebody's friend\n\n(Chorus)\nLet's sing in unison, our voices strong and true\nCreating a harmony that binds us like glue\nIn this pop anthem, let our love never end\nBecause everybody needs to be somebody's friend\n\n5. (Verse 1)\nIn this pop ballad, Rustic Overtones shine\nUniting hearts and minds, like a divine design\nLet's embrace diversity, leaving no one behind\nBecause everybody needs to be somebody's friend\n\n(Chorus)\nWith every verse and chorus, let's spread the love\nTogether we'll rise, soaring high above\nIn this pop chorus, our friendships won't bend\nBecause everybody needs to be somebody's friend"
            },
            {
                "role": "user",
                "content": "Title: Watching Over Me; Tag: pop; Artist: Canadian Tenors; Number of samples you have to generate  : 3"
            },
            {
                "role": "assistant",
                "content": "Here are three sample lyrics based on the fields you provided:\n\n1. (Verse 1)\nIn the still of the night, when shadows start to creep,\nI feel a presence, a love that's buried deep.\nCanadian Tenors singing melodies so sweet,\nWatching over me, with harmonies that speak.\n\n(Chorus)\nWhen I'm lost and feeling all alone,\nTheir pop tunes guide me back home.\nWatching over me, their voices set me free,\nCanadian Tenors, my guardian melody.\n\n2. (Verse 1)\nFrom the moment I wake till the sun starts to set,\nThe Canadian Tenors' pop tunes I'll never forget.\nTheir lyrics and harmonies, a gentle safety net,\nWatching over me, my heart they'll never let.\n\n(Chorus)\nIn times of darkness, their voices shine so bright,\nGuiding me through the depths of the night.\nWatching over me, their music sets me free,\nCanadian Tenors, my guardian melody.\n\n3. (Verse 1)\nWhen the world feels heavy and my spirit is weak,\nI turn to Canadian Tenors, whose pop songs speak.\nTheir voices carry me to a place so serene,\nWatching over me, like a tranquil dream.\n\n(Chorus)\nWith each heartfelt note, they lift me up high,\nTheir melodies heal, never asking why.\nWatching over me, their music sets me free,\nCanadian Tenors, my guardian melody.\n\nNote: The lyrics provided are original content generated by OpenAI's language model. Any resemblance to existing songs or artists is purely coincidental."
            },
            {
                "role": "user",
                "content": input
            }
        ],
        temperature=1,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(dir(response))

    # latest_prediction = {
    #     key: (str(value) if isinstance(value) else value)
    #     for key, value in response.items()
    # }
    # print(latest_prediction)

    return response.choices[0].message.content


