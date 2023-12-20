import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "google/flan-t5-xxl"

#llm = huggingface_hub.HuggingFaceHub(repo_id, model_kwargs={"temperature":0, "max_length":64})

question = "What is the meaning of life?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(
    template=template, 
    input_variables=["question"]
)

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))