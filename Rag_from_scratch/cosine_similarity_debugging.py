# This code is for calculating the cosine similarity between the question and the document using HuggingFace embeddings.
# You can change the question and document to test with different inputs. 
# The cosine similarity will give you a value between -1 and 1, where 1 means the vectors are identical,
# 0 means they are orthogonal, and -1 means they are opposite.
# Also different embedding models may give different similarity scores, so you can experiment with different models
# to see how it affects the results.

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

question = "How many retail stores does Nike have outside USA?"
#question = "What pets do I like?"
#question = "Which animals do I like as pets?"
#question = "I like which pets?"
document = "USA RETAIL STORES are 10"

#embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embed = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_1 = embed.embed_query(question)
vector_2 = embed.embed_query(document)

similarity = cosine_similarity(vector_1, vector_2)
print("Cosine Similarity:", similarity)
