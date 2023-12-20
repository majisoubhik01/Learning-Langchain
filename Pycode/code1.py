# Just one chain in langchain

from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return a list of numbers")
args = parser.parse_args()

repo_id = "bigcode/santacoder"

llm = HuggingFaceHub(
    repo_id = repo_id,
    model_kwargs = {
        "temperature": 0.5,
        "max_length": 64,
    }
)

code_prompt = PromptTemplate(
    template = "Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt
)

result = code_chain.run({
    "language": args.language,
    "task": args.task,
})

print(result)