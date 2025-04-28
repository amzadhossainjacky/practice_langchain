#necessary imports
from decouple import config
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


# OpenAI API key
SECRET_KEY = config('OPENAI_API_KEY')

# openai model generate
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

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

""" print(f"Enter your query: ")
query = input()
similarity_docs = db_connection.similarity_search(query, k=1)
print(similarity_docs[0].page_content) """

#query = "চ্যাম্পিয়ন ট্রফিতে মাহমুদউল্লাহর সময়টা কেন ভালো যাচ্ছে না?"
#query = "keno mahmudullah champion trophy te valo korte parchen na"
#query = "what speech agent said on topic greetings"
query = "মানিক"

# retrieve the most relevent chunks from the database for the user query
retriver = db_connection.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},  
)

relevennt_chunks = retriver.invoke(query)


# put the relevant chunks + user query into the LLM model openai and generate the answer 
modified_prompt = f'Here is the User Query: "  {query} + "\n" + "And Here are the relevant Information where you can find the ansewer" + "\n" + "do not use other info other than the additional information I provided to you." + "\n" + "Relevat infomations: " + "\n" + {relevennt_chunks[0].page_content} + "\n"'


# generate the answer
response = llm.invoke(modified_prompt)

print(response.content)
