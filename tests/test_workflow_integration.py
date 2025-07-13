from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

def test_llm_chain_initialization():
    llm = OpenAI(model="text-davinci-003")
    prompt = PromptTemplate(template="Say hello to {name}", input_variables=["name"])
    chain = LLMChain(llm=llm, prompt=prompt)
    assert chain is not None
    result = chain.run(name="Andre")
    assert isinstance(result, str)
