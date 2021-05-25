import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn
import pandas as pd

def get_wordnet_vocab_as_dataframe():
    """
    Create a pandas dataframe from wordnet corpus
    :return pandas dataframe
    """
    wordnet_dict = {}
    wordnet_dict['pos'] = []
    wordnet_dict['name'] = []
    wordnet_dict['definition'] = []
    wordnet_dict['id'] = []
    count = 0
    for synset in list(wn.all_synsets()):
        count += 1
        wordnet_dict['pos'].append(synset.pos())
        wordnet_dict['name'].append(synset.name())
        wordnet_dict['definition'].append(synset.definition())
        wordnet_dict['id'].append(count)
    df = pd.DataFrame(wordnet_dict, columns = ['id', 'name', 'pos', 'definition'])
    df.to_pickle("store/df.pkl")
    return df
