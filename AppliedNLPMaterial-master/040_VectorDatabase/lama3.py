#%% packages
import os
from pprint import pprint
import chromadb
import ollama

# chroma_db = chromadb.Client()  # on the fly
chroma_db = chromadb.PersistentClient(path="db")

#%%
chroma_db.list_collections()
#%% Get / Create Collection
chroma_collection = chroma_db.get_or_create_collection("movies")

res = chroma_collection.query(query_texts=["a monster in closet"], n_results=5)
#%% count of documents in collection
len(chroma_collection.get()['ids'])

# %% Run a Query
def get_query_results(query_text:str, n_results:int=5):
    res = chroma_collection.query(query_texts=[query_text], n_results=3)
    docs = res["documents"][0]
    titles = [item['title'] for item in res["metadatas"][0]]
    res_string = ';'.join([f'{title}: {description}' for title, description in zip(titles, docs)])
    return res_string

query_text = "a monster in the closet"
retrieved_results = get_query_results(query_text)
print(retrieved_results)
