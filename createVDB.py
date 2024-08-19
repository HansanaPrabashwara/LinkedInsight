from preprocess import postsList
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import numpy as np
import pickle

load_dotenv()
os.getenv("GOOGLE_API_KEY")

def get_vector_store(text_chunks):
    """Create a vector database using the text chunks

    Args:
        text_chunks (list): Generated text chunks as a list
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    



get_vector_store(postsList("influencers_data.csv", "content"))

# # print(postsList("influencers_data.csv","content"))


# # for i in 

# # isFile = os.path.exists("./fais_index")
# # print(isFile)


# # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
# # vector_store = FAISS.from_texts(postsList("influencers_data.csv", "content"), embedding=embeddings)

# # for text in postsList("influencers_data.csv", "content"):
# # vector_store.aadd_texts(postsList("influencers_data.csv", "content"), embeddings=embeddings)

# # vector_store.save_local("faiss_index")



# embedder = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
# posts = postsList("influencers_data.csv", "content")

# print("Generating Embeddings")
# embeddings =  embedder.embed_documents(posts)

# print("Saving Embeddings")
# pickle.dump(embeddings, "post_embeddings.pkl")



# embeddings =  np.array(embeddings)




# dimension = embeddings.shape[1]  # Size of embeddings
# index = FAISS.IndexFlatL2(dimension)  # L2 distance index

# # Step 3: Add embeddings to the FAISS index
# print("Adding Embeddings to VDB")
# index.add(embeddings)

# print("Creating the vector store")
# vector_store = FAISS(embedding_function=embedder, index=index)
