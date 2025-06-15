from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Deprecated: Use InMemoryChatMessageHistory instead
# from langchain.memory import ConversationBufferMemory

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

# Deprecated: Use InMemoryChatMessageHistory instead
# memory = ConversationBufferMemory(
#     memory_key="messages", # conversations are added to this key
#     return_messages=True, # wrap the messages in classes (e.g. HumanMessage, AIMessage)
# )
memory = InMemoryChatMessageHistory()

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),  # this will be replaced by the conversation history. This will look into the memory.
        HumanMessagePromptTemplate.from_template("{content}"),
    ]
)

chain = prompt | chat
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="content",
    history_messages_key="messages",
)

while True:
    content = input(">> ")

    result = chain_with_history.invoke({"content": content}, config={"session_id": "1234"})

    print(result)
