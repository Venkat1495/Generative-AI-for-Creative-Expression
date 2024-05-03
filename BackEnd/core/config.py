# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from urllib.parse import quote_plus
# from pydantic_settings import BaseSettings
#
# env_path = Path("BackEnd") / ".env"
# load_dotenv(dotenv_path=env_path)
#
# class settings(BaseSettings):
#
#     # #Database
#     # DB_USER: str = os.getenv('MYSQL_USER')
#     # DB_PASSWORD: str = os.getenv('MYSQL_PASSWORD')
#     # DB_NAME: str = os.getenv('MYSQL_DB')
#     # DB_PORT: str = os.getenv('MYSQL_PORT')
#     # DB_HOST: str = os.getenv('MYSQL_HOST')
#     DATABASE_URL: str = os.getenv('MONGODB_URI')
#
#
# def get_settings() -> settings:
#     return settings()