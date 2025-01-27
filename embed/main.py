from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY, temperature=0.6)

# embed data by OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)
#query_embedding = embeddings_model.embed_query("What is the meaning of life?")
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
    ]
)

print(len(embeddings))