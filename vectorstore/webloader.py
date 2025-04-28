from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY, temperature=0.6)

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
SECRET_KEY = config('OPENAI_API_KEY')

loader = WebBaseLoader(
    ["https://buyer-helpcenter.daraz.com.bd/s/page/category?categoryId=1000001017&m_station=page"]
)
docs = loader.load()

# Optionally, split documents into smaller chunks using a text splitter
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)

#data based creation
db = Chroma.from_documents(split_docs, embeddings_model, persist_directory='./chroma_db')

# Prompt the user for a query
print(f"Enter your query: ")
query = input()

# Perform similarity search on the loaded documents
similarity_docs = db.similarity_search(query, k=1)

# Output the most similar document
print(similarity_docs[0].page_content)

