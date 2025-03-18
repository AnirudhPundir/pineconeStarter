from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid;

load_dotenv()

file_path = "./data.xlsx"

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "observations"

model = SentenceTransformer('all-MiniLM-L6-v2')


def initialize_pinecone_index(index_name, embeddings, data):

    existing_indexes = pc.list_indexes()

    dimension = embeddings.shape[1]

    exists = any(index["name"] == index_name for index in existing_indexes)

    if not exists:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Ensure that the index is created successfully
        # print(f"Index {index_name} created.")
    else:
        print(f"Index Name: {index_name}")

    # separate function created to create embeddings
    # embeddings = pc.inference.embed(
    #     model="multilingual-e5-large",
    #     inputs=[d['text'] for d in data],
    #     parameters={'input_type': 'passage', 'truncate': 'END'}
    # )

    # print(embeddings[0])

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pc.Index(index_name)

    vectors = []

    for d, e in zip(data, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": e,
            "metadata": {'text': d}
        })

    # print(f"{vectors}")

    index.upsert(vectors=vectors, namespace="ns1")

# Example usage
# data =  [
#     {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
#     {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
#     {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
#     {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
#     {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
#     {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
# ]


def createEmbeddings():
    
    df = pd.read_excel(file_path, sheet_name="ObservationsReduced")

    #Select relevant columns to be converted
    columns_to_embed = df.columns

    #Concatenate the values into a single string
    df["combined_text"] = df[columns_to_embed].astype(str).apply(lambda x: " | ".join(x), axis=1)

    # Store combined_text in a separate variable
    combined_texts = df["combined_text"].tolist()

    createEmbeddingsModel =  SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = createEmbeddingsModel.encode(combined_texts)

    embeddings = np.array(embeddings).astype('float32')

    print(embeddings.shape[1])

    return {"embeddings": embeddings, "data": combined_texts}
    # print(combined_texts)

    # Keep only the combined_text column
    # df = df[["combined_text"]]

    # print(f"{df["combined_text"]}")

def search_pinecone_db(index_name, query, top_k=3):
    
    existing_indexes = pc.list_indexes()

    exists = any(index['name'] == index_name for index in existing_indexes)

    if not exists:
        return "Index Missing"
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    index = pc.Index(index_name)

    #fetch the embedding model
    start_embedding_time = time.time()  # Start time for embedding creation

    #create embedding for requested query
    query_embedding = model.encode([query]).tolist()
    end_embedding_time = time.time()  # End time for embedding creation

    embedding_elapsed_time = end_embedding_time - start_embedding_time  # Calculate elapsed time for embedding creation
    print(f"Embedding Time Taken: {embedding_elapsed_time}")  # Print the time taken for embedding creation

    start_query_time = time.time()  # Start time for the query
    results = index.query(vector=query_embedding[0], top_k=top_k, include_metadata=True, namespace="ns1")
    end_query_time = time.time()  # End time for the query

    query_elapsed_time = end_query_time - start_query_time  # Calculate elapsed time for the query
    print(f"Query Time Taken: {query_elapsed_time}")  # Print the time taken for the query

    #fetch the metadata from retrieved indexes
    retrieved_context = [match["metadata"]["text"] for match in results["matches"]]

    #concatenate the data
    context = "\n".join(retrieved_context)

    return context

# start_time = time.time()


#create embeddings
# embeddedData = createEmbeddings()

# initialize_pinecone_index(index_name, embeddedData["embeddings"], embeddedData["data"])

# print(embeddings)

query = "Observation for oil spill in kitchen"
context = search_pinecone_db(index_name, query, 3)
# print(f"{context}")

# end_time = time.time()

# elapsed_time = end_time - start_time

# print(f"Time Taken: {elapsed_time}")

# initialize_pinecone_index("ESS", embeddings)
