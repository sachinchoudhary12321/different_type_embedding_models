from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

docs = ['hello sachin the great','jaipur is beautiful city','hello hello world']

text = 'hello world jb'
vector = embeddings.embed_query(text)
print(str(vector))
vector = embeddings.embed_documents(docs)
 

print(str(vector))