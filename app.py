from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Mapping
from langchain.llms.base import LLM
from transformers import pipeline
import torch
from flask import Flask, request


MAX_TOKENS = 256
MAX_INPUT_SIZE = 1024
NUM_OUTPUT = 256
MAX_CHUNK_OVERLAP = 30

TXTGEN_MODEL_NAME = "facebook/opt-iml-1.3b"


class TextGenLocalOPT(LLM):
    pipeline = pipeline(
        task='text-generation',
        model=TXTGEN_MODEL_NAME,
        model_kwargs={"torch_dtype": torch.float32, 'max_length': 500}
    )

    def _call(self, prompt: str, stop=None) -> str:
        response = self.pipeline(prompt, max_new_tokens=MAX_TOKENS)[
            0]["generated_text"]
        return response[len(prompt):]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": TXTGEN_MODEL_NAME}

    @property
    def _llm_type(self) -> str:
        return "custom"


# docs = DirectoryLoader('./data').load()
with open('./data/job_application.txt') as file:
    examples = file.read()

llm = TextGenLocalOPT()
prompt = PromptTemplate(
    input_variables=["q", "examples"],
    template="Given the following examples, answer the Question, as truthfully as possible, if not sure or need additional information reply 'unknown'\n\n### Examples:\n\n{examples}\n\nQ: {q}\nA:",
)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

app = Flask(__name__)


@app.route("/question")
def question():
    q = request.args.get('q')
    return llm_chain.run({"examples": examples, 'q': q}).lstrip()


gen_llm = TextGenLocalOPT()
gen_prompt = PromptTemplate(
    input_variables=["thing", "examples"],
    template="Use the following examples to write a {thing}, as truthfully as possible, if not sure or need additional information reply 'unknown'\n\n### Examples:\n\n{examples}\n\n{thing}:",
)
gen_llm_chain = LLMChain(llm=gen_llm, prompt=gen_prompt, verbose=False)


@app.route("/generate")
def generate():
    thing = request.args.get('q')
    return gen_llm_chain.run({"examples": examples, 'thing': thing}).lstrip()


if __name__ == "__main__":
    app.run(port=8000, debug=True)
