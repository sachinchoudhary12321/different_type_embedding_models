# there is different text and user come and ask the query so this model find which one is more similiar
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

docs = ['sachin the make the 100 century','jaipur is beautiful city','this world is so big','i am very handsome']

query = 'tell me about jaipur'


docs_embeddings = embeddings.embed_documents(docs)
 

query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings]
                  ,docs_embeddings)[0]

print(list(enumerate(scores)))

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)

print(docs[index])
print(score)


