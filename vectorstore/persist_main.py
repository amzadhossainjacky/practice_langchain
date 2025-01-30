#necessary imports
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# OpenAI API key
SECRET_KEY = config('OPENAI_API_KEY')

#document load
loader = TextLoader('./data/tech.txt', encoding='utf-8')
tech_doc = loader.load()

#text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=800,
    chunk_overlap=100,
)
tech_split_doc = text_splitter.split_documents(tech_doc)

# embed data by OpenAI
embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)

#data based creation
db_connection = Chroma(embedding_function=embeddings_model, persist_directory='./chroma_db')

# query = "Gain Experience with Advanced Topics:"
print(f"Enter your query: ")
query = input()
similarity_docs = db_connection.similarity_search(query, k=1)
#print(similarity_docs)
print(similarity_docs[0].page_content)
