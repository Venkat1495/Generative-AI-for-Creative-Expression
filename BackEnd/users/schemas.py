from pydantic import BaseModel, EmailStr, validator
from BackEnd.users.model import UserModel, PredictionModel

class CreateUserRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

    @validator('password', pre=True, always=True)
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str or None = None

class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None

class UserInDB(UserModel):
    password: str


class prediction_input(BaseModel):
    Title: str
    Genre: str
    Artist: str
    Number_of_Samples: int

class Prediction_results(BaseModel):
    Title: str
    Genre: str
    Artist: str
    Number_of_Samples: int
    Lyrics: str
    GPT_Lyrics: str

class LinkItem(BaseModel):
    title: str
    urlSegment: str