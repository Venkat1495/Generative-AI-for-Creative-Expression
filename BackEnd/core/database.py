import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from pymongo.collection import Collection
import certifi
from pymongo.errors import ConnectionFailure

username = "admin"
password = "King@456"  # Assuming this password needs URL encoding because it contains '@'
encoded_username = quote_plus(username)
encoded_password = quote_plus(password)
ca = certifi.where()

uri = f"mongodb+srv://{encoded_username}:{encoded_password}@creativewriter.4bebbvo.mongodb.net/?retryWrites=true&w=majority&appName=CreativeWriter"

# env_path = Path("BackEnd") / ".env"
# load_dotenv(dotenv_path=env_path)
#
# MONGODB_URI = os.getenv('MONGODB_URI')
# print(MONGODB_URI)
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)
db = client.CW# Or specify your database name explicitly

try:
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("MongoDB is connected")
except ConnectionFailure:
    print("Server not available")

def get_database():
    collection: Collection = db
    return collection
