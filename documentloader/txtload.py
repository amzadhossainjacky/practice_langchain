from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY, temperature=0.6)

# text loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader('./data/text.txt')
mydata = loader.load()

# print(mydata)
# print(mydata[0])
print(mydata[0].page_content)



