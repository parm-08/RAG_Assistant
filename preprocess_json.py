# # You’ve built a semantic search system:

# # Take text chunks from JSON files

# # Convert each chunk into a vector (embedding) using Ollama

# # Store those embeddings in a DataFrame

# # When a user asks a question:

# # Convert the question into an embedding

# # Compare it with all stored embeddings using cosine similarity

# # Return the most relevant chunks

# # This is the core of RAG (Retrieval-Augmented Generation).
# import requests
# import os
# import json
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics.pairwise import cosine_similarity

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "nomic-embed-text",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# jsons = os.listdir("jsons")  # List all the jsons 
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
       
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk) 
      
# # print(my_dicts)

# df = pd.DataFrame.from_records(my_dicts)
# # print(df)
# #Save This DataFrame
# joblib.dump(df,'embeddings.joblib')
# incoming_query = input("Ask a question:")
# question_embedding=create_embedding([incoming_query])[0]
# # print(question_embedding)


# # FIND SIMILARITIES OF QUESTION_EMBEDDINGS WITH OTHER EMBEDDING S (USING COSINE SIMILARITY)
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding'].shape))

# similarities=cosine_similarity(np.vstack(df[
#     'embedding']),[question_embedding]).flatten()
# # print(similarities)
# top_results=3
# max_indx=similarities.argsort()[: :-1][0:top_results]
# # print(max_indx)

# new_df=df.loc[max_indx]
# print(new_df[["title","number","text"]])


# prompt=f''' I am teaching web development using Sigma web development course.
# Here are video chunks containing video title,video number, start time in seconds, end time in seconds,the text at that time:

# {new_df.to_json()}
# --------------------------------------------

# {incoming_query}   
# The user has asked a question related to the video content. 
# Based on the video chunks, you must identify where the topic is taught by specifying:
# - the video title
# - the exact timestamp (start and end)

# Then guide the user to watch that specific part of the video to understand the concept.
# If the user asks unrelated questions, tell him that you can only answer questiosn related to the course.
# '''
# print("Reached prompt writing section")

# with open ("prompt.txt","w") as f:
#     f.write(prompt)
# # for index,item in new_df.iterrows():
#     # print(index,item["title"],item["number"],item["text"],item["start"],item["end"])



# # a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# # print(a)


import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("newjsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newjsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')

