from decouple import config
SECRET_KEY = config('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY, temperature=0.6)

from langchain_text_splitters import CharacterTextSplitter

# Load an example document
with open("./data/sample_eg.txt") as f:
    sample_data = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
)

mydata = text_splitter.create_documents([sample_data])
print(mydata[1])

