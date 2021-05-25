
import faiss
import numpy as np
from .create_embeddings import *
from .process_wordnet import *

df = get_wordnet_vocab_as_dataframe()

def create_faiss_indices(model_code="dsbert", quantizer="HNSWFlat"):
    """
    Creates indices for efficient similarity search 
    Args:
        model_code (str): code for model to be used to create sentence embeddings
        quantizer (str): quantizer for compressing the indices for efficient search
    """
    
    embeddings = create_sentence_transformer_embeddings(model_code, df.definition.to_list())
    if quantizer == "HNSWFlat":
        quantizer = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        index = faiss.IndexIVFPQ(quantizer, embeddings.shape[1], 3, 16, 8)
        # Step 3: Pass the index to IndexIDMap
        index = faiss.IndexIDMap(index)

        # Step 4: Add vectors and their IDs
        index.train(embeddings)
        index.add_with_ids(embeddings, df.id.values)
        faiss.write_index(index,"store/wordnet_" + model_code + ".index")

if __name__ == "__main__":
    create_faiss_indices("dsbert")
        

