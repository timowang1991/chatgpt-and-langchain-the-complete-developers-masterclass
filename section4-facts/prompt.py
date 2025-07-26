from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True  # Enable debug mode for detailed logging

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(persist_directory="emb", embedding_function=embeddings)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",  # "stuff" is the default chain type, there are map reduce, map rerank, and refine as well
    verbose=True,
)

result = chain.run(
    "What is an interesting fact about the English language?",
)

print(result)