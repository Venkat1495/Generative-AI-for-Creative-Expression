from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
import openai
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

openai.api_key = "sk-proj-b5iDdr24YmJrapUY8hxDT3BlbkFJQl6gXTNMYJxK1My4U22w"
SECRET_KEY = "61df5f076a10eca7b66f34240f0fe6de1866cdc4c7b7d5ab835bc125d2ae2751"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30