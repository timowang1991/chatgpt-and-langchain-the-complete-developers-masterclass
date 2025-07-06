from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200, # at most 200 characters per chunk
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(
    docs,
    embedding=embeddings, # calls openai embeddings
    persist_directory="emb"
)

# for doc in docs:
#     print(doc.page_content)
#     print('\n')

# getting tuples
# results = db.similarity_search_with_score(
#     'What is an interesting fact about the English language?',
#     # k=2, # return 2 most similar chunks
# )

# for result in results:
#     print('\n')
#     print(result[1])
#     print(result[0].page_content)

# getting only the content
results = db.similarity_search(
    'What is an interesting fact about the English language?',
    k=1, # return the most similar chunk
)

for result in results:
    print('\n')
    print(result.page_content)