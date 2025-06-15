from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='return a list of numbers')
parser.add_argument('--language', default='python')
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template='Write a very short {language} function that will {task}',
    input_variables=['language', 'task']
)

test_prompt = PromptTemplate(
    template='Write some unit tests code for the following {language} code:\n{code}',
    input_variables=['language', 'code']
)

# deprecated LLMChain usage
# code_chain = LLMChain(
#     llm=llm,
#     prompt=code_prompt,
# )

# new LLMChain usage
chain = (
    code_prompt
    | llm
    | (lambda code: { 'code': code, 'language': args.language })
    | test_prompt
    | llm
)

result = chain.invoke({
    'language': args.language,
    'task': args.task
})

print(result)
