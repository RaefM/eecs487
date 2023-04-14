import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import gensim.downloader
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from coherenceModelNews import *

def read_aviation():
    return pd.read_csv('moreAviationPerms.csv')

def read_reddit():
    return pd.read_csv('redditPerms.csv')

def read_wsj():
    return pd.read_csv('wsjPerms.csv')

def read_la():
    return pd.read_csv('LATimesWashPostPerms.csv')

def read_reuters():
    return pd.read_csv('reutersPerms.csv')

def get_statistics(series, title):
    print(f"Mean of {title}: {np.mean(series.values)}")
    print(f"SD of {title}: {np.sqrt(np.var(series.values))}")
    series.rename(title).hist(legend=True, bins=20)
    plt.savefig(title + '_stats.png')

def validate_args(dataset, model_type):
    data_valid = dataset in ['aviation', 'reddit', 'wsj', 'la', 'reuters']
    model_valid = model_type in ['aviation', 'la', 'wsj']

    if not (data_valid and model_valid):
        usage_warning()
        exit(1)

def usage_warning():
    print("Usage: python3 scoreData.py <dataset> <model>")
    print("Datasets: aviation, reddit, wsj, la, reuters")
    print("Models: aviation, la")

def get_model(model_type, device):
    fname = ''
    if model_type == 'la':
        fname = 'best_rnn_latimes.pt'
    elif model_type == 'wsj':
        fname = 'best_rnn_wsj.pt'
    else:
        fname = 'best_rnn_av_complex.pt'

    basic_model = FFNN(5, True, device)
    basic_model.load_state_dict(torch.load(fname))
    basic_model.eval()
    basic_model.to(device)

    return basic_model

def get_dataset(dataset_type):
    if dataset_type == 'aviation':
        return read_aviation()
    elif dataset_type == 'reddit':
        return read_reddit()
    elif dataset_type == 'wsj':
        return read_wsj()
    elif dataset_type == 'la':
        return read_la()
    else:
        return read_reuters()

def get_windows(post, embed, unk):
    window_size = 5
    sentences = get_paragraph_embedding_tup(embed, unk, post)
    num_windows = len(sentences) - window_size + 1
                
    if num_windows < 0:
        print(f"WARNING: Post did not have enough sentences for window size {window_size}; returning None")
        return None
    
    return [tensor_of_tupled_par_embed(sentences[i:i+window_size]) for i in range(num_windows)]

def get_coherence(post, model, device, embed, unk):
    windows_of_post = get_windows(post, embed, unk)
    with torch.no_grad():
        windows = [[s.to(device) for s in window] for window in windows_of_post]
        scores = torch.sigmoid(model(windows))
        return torch.mean(scores).tolist()

def compute_coherence(model, dataset, device, embed, unk):
    def get_coherence_wrapper(par): 
        return get_coherence(par, model, device, embed, unk)
    
    return dataset.paragraph.apply(get_coherence_wrapper)

def main():
    if len(sys.argv) != 3:
        usage_warning()
        exit(1)
    
    dataset_type = sys.argv[1]
    model_type = sys.argv[2]

    validate_args(dataset_type, model_type)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = get_model(model_type, device)
    dataset = get_dataset(dataset_type)

    embed = gensim.downloader.load("glove-wiki-gigaword-100")
    unk = np.mean(embed.vectors, axis=0)

    dataset['coherence'] = compute_coherence(model, dataset, device, embed, unk)
    dataset.to_csv(dataset_type + 'WithCoherenceBy' + model_type + '.csv')


