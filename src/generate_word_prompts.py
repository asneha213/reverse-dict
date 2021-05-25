import faiss
import numpy as np
import pickle

def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.

    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx][column]) for idx in I[0]]

def get_word_prompts_from_query(df, user_query, model, model_code, num_results=30):
    """
    Get list of words whose definitions closely match the user query
    Args:
        user_query (str): query prompted by the user
        model (model obj): model loaded by pytorch
        model_code (str): string denoting the shorthand of the model
        num_results (int): number of results to be returned
    Returns:
        List of words and their definitions
    """

    index = faiss.read_index("store/wordnet_" + model_code + ".index")
    D, I = vector_search([user_query], model, index, num_results)
    prediction = id2details(df, I, 'name')
    defn = id2details(df, I, 'definition')
    return list(zip(prediction, defn))


if __name__ == "__main__":
    df = pickle.load(open("store/df.pkl", "rb"))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    user_query = "a very wealthy man"
    model_code = "dsbert"
    list_defns = get_word_prompts_from_query(df, user_query, model, "dsbert")
    print(list_defns)
