from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

messages = [
    HumanMessage("Hello"),
]

response = model.invoke(messages)
print(response.content)
