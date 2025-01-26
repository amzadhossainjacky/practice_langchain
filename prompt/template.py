
# key fetch from environment variable
from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

# openai model generate
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

# response = model.invoke(messages)
# print(response.content)

# prompt dynamic template initialization
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

#invoke the prompt_template
prompt = prompt_template.invoke({"language": "Bangla", "text": "How are you?"})

#pass the template to the ai model

response = model.invoke(prompt)
print(response.content)