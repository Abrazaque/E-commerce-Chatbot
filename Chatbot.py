import os
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.memory import ConversationBufferMemory
# from dotenv import load_dotenv
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
app = Flask(__name__)

# load_dotenv('var.env')
key = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4-1106-preview")
memory = ConversationBufferMemory(chat_history="chat_history")

store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace=embeddings.model
)
# list(store.yield_keys())

file_path = Path(r"D:\Data science\Xflow\Module 2 Vectordb\Data (CSV's)\amazon.csv")
loader = CSVLoader(file_path=file_path, encoding='utf-8')
data = loader.load()


r_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000,
    chunk_overlap=200
)
docs = r_splitter.split_documents(data)

vectbd = FAISS.from_documents(docs,cached_embedder)
retriever = vectbd.as_retriever()
retriever=vectbd.as_retriever()

from langchain.prompts import PromptTemplate

prompt_template = """
You are an E-Commerce AI ChatBot, that assists users related to E-Commerec Query Only, other than that you will say sorry.\ 
Response should be short, precise 
in the answer try to provide as much text as possible from "response" section in the source documents 
If a user asks a question that is available in the provided dataset, try to provide the answer from the "response" section in the source documents. If the answer is not available in the dataset but is relevant to E-Commerce, generate a response in your own words. 
For any non-E-Commerce queries, apologize and encourage users to ask about online shopping or products.
first understand the question if its generally related to E-commerce directly of indirectly try to answer from source if not available use your own words
if answer is not available in the source documents, and it is related to "E-commerce" and "tech products", then you can use your own words to answer the query.                          
                        For instance:

                            *Act as a General Assistant for General E-Commerce queries Only n\

                            *Act as an E-Commerce assistant for product-related inquiries. \
                                    
                        %REMEMBER:
                            * for greetings like: Hi, Hello, how are you,Salam and other greetings you have to answer it
                            * You will say sorry, if the query is out of E Commerce domain.
                                    like:
                                        What is the metaverse?
                                        What is AI?
                                        What is blockchain?
                                        Who is the founder of TATA?
                                        What is SpaceX?
                                        And so on for non-e Commerce queries.
                                        For all queries like this, you will not respond and say sorry to the user because you are an assistant who will only assist users with any eCommerce-related query.\


                            EXAMPLE: 
                                User Query: what is AI?
                                Your Response: Sorry, but that query is out of the E-Commerce domain. If you have any questions related tonline shopping, product inquiries, or any assistance with an e-commerce platform, feel free to ask
                                
                                User Query: Who is Elon Musk?
                                Your Response: Sorry, but that query is out of the E-Commerce domain. If you have any questions related to online shopping, product inquiries, or any assistance with an e-commerce platform, feel free to ask
                                
                                User Query: Any moblile phone specifications?
                                Your Response: Answer the query from the source documents. If the answer is not available in the source documents, then you can use your own words to answer the query.
                                
                                
CONTEXT:{context}
QUESTION:{question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=['context', 'question']
)


chain = RetrievalQA.from_chain_type(
    llm=llm,
    memory = memory,
    chain_type='stuff',
    retriever=retriever,
    input_key='query',
    chain_type_kwargs={'prompt': PROMPT}
)

@app.route('/query', methods=['POST'])
def process_query():
    prompt = request.json.get('prompt', '')
    result = chain(prompt)
    return jsonify({'result': result['result']})

if __name__ == '__main__':
    app.run(port=5000) 