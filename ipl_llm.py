# from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
# from langchain.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.document_loaders import BSHTMLLoader
# from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
import requests
from bs4 import BeautifulSoup

# import os
from googlesearch import search

hf = HuggingFacePipeline.from_model_id(model_id='google/flan-t5-xl', task="text2text-generation")
chain = load_qa_chain(llm=hf, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=chain)

s = search("today's ipl match prediction", num_results=10, advanced=True)

url_ls = [i.url for i in s]
# query = "Which team will win today's match between LSG & RR?"

query = input('\n Enter your IPL match question: ')

for url in url_ls:
    print(url)
    r1 = requests.get(url)
    soup1 = BeautifulSoup(r1.content, 'html.parser')
    text=soup1.text
    print(qa_document_chain.run(input_document=text, question = query))
