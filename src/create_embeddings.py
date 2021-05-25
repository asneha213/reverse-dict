
import torch
import numpy as np



def load_model(model_code):
    """
    Load model and return
    Args:
        model_code (str): short-hand of the model name
    Return:
        model object
    """
    # model name can be chosen from
    # https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
    if model_code == "dsbert":
        model_name = "distilbert-base-nli-stsb-mean-tokens"
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
    return model
        


def create_sentence_transformer_embeddings(model_code, list_defns):
    """
    Creating embeddings out of sentence transformer models:
    https://pypi.org/project/sentence-transformers/
    Args:
        :param model_name: string
        :param list_defns: list of word net word definitions 
    Returns:
        :return embeddings as an array
    """
    model = load_model(model_code)
    embeddings = model.encode(list_defns, show_progress_bar=True)
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    return embeddings






    
