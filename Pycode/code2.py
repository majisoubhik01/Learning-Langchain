# Create two chains and connect

from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return a list of numbers")
args = parser.parse_args()

repo_id1 = "bigcode/santacoder"
repo_id2 = "tiiuae/falcon-7b-instruct"

llmA = HuggingFaceHub(
    repo_id = repo_id1,
    model_kwargs = {
        "temperature": 0.5,
        "max_length": 64,
    }
)

llmB = HuggingFaceHub(
    repo_id = repo_id2,
    model_kwargs = {
        "temperature": 0.5,
        "max_length": 64,
    }
)

code_prompt = PromptTemplate(
    template = "Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}"
)

code_chain = LLMChain(
    llm = llmA,
    prompt = code_prompt,
    output_key = "code"
)

test_chain = LLMChain(
    llm=llmB,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language","task"],
    output_variables=["code", "test"]
)

code = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>>>> GENERATED CODE:")
print(code["code"])
print(">>>>>>> GENERATED TEST:")
print(code["test"])