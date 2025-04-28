#necessary imports
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# OpenAI API key
SECRET_KEY = config('OPENAI_API_KEY')

#document load
loader = TextLoader('./data/tech1.txt', encoding='utf-8')
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
db = Chroma.from_documents(tech_split_doc, embeddings_model, persist_directory='./chroma_db')










