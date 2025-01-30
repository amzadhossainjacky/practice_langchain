from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

# key fetch from environment variable
SECRET_KEY = config('OPENAI_API_KEY')

# openai model generate
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

""" 
# prompt dynamic template initialization
human_template = "Tell me the fact about {topic}"
chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(human_template)
])

#deprecated
# chain = LLMChain(llm=model, prompt=chat_prompt)

chain = chat_prompt | model
response = chain.invoke({"topic": "Python"})
print(response.content) 
"""

#example 2

# Example 2: Different template for a person-city query
chat_prompt1 = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("What is the city {person} is from?")
])

chat_prompt2 = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template( "what country the city {city} is in? response in english",)
])


city_chain = chat_prompt1 | model
country_chain = ({
    "city" : city_chain} | chat_prompt2 | model
)

response = country_chain.invoke({"person": "virat kohli"})
print(response.content)
