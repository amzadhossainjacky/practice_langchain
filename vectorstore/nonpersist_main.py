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
# tech_split_doc = text_splitter.create_documents(tech_doc)

# embed data by OpenAI
embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)

#data based creation
#db = Chroma.from_documents(tech_split_doc, embeddings_model)
db = FAISS.from_documents(tech_split_doc, embeddings_model)
query = "Everyday Protection vs. Underwater Adventures"

docs = db.similarity_search(query)
print(docs[0].page_content)






