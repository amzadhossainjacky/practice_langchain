# from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

import getpass
import os

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
response = model.chat("What is the meaning of life?")
print(response)

