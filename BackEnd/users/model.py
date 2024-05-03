from pydantic import BaseModel, Field, EmailStr, validator
from bson import ObjectId
from typing import Optional
from datetime import datetime

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema_(cls, field_schema):
        field_schema.update(type="string")

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    first_name: str = Field(...)
    last_name: str = Field(...)
    email: EmailStr = Field(...)
    username: EmailStr = Field(...)
    password: str  # Ensure this is never output to the client
    is_active: bool = Field(default=False)
    is_verified: bool = Field(default=False)
    verified_at: Optional[datetime] = None
    registered_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        orm_mode = True
        schema_extra = {
            "example": {
                "first_name": "Jane",
                "last_name": "Doe",
                "email": "janedoe@example.com",
                "password": "securepassword123",
                "is_active": True,
                "is_verified": False
            }
        }

    @validator('email')
    def validate_email(cls, v):
        # Additional email validation can be placed here if needed
        return v

    @validator('password')
    def validate_password(cls, v, pre=True, always=True):
        if len(v) < 8:
            print(v)
            raise ValueError("Password must be at least 8 characters long")
        return v


class PredictionModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    Title: str
    Genre: str
    Artist: str
    Number_of_Samples: int
    Lyrics: str
    GPT_Lyrics: str
    user_id: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)



    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        orm_mode = True

    # Using this to parse the MongoDB data
    @classmethod
    def from_mongo(cls, data: dict):
        if not data:
            return None
        return cls(Lyrics=data.get("Lyrics", ""), GPT_Lyrics=data.get("GPT_Lyrics", ""))


