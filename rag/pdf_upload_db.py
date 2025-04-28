#necessary imports
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# OpenAI API key
SECRET_KEY = config('OPENAI_API_KEY')
file_path = './data/test.pdf'
#document load
loader = PyPDFLoader(file_path)
tech_doc = loader.load()

#text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
tech_split_doc = text_splitter.split_documents(tech_doc)

# embed data by OpenAI
embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)

#data based creation
db = Chroma.from_documents(tech_split_doc, embeddings_model, persist_directory='./chroma_db')










