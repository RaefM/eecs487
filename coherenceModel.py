import time
import copy
from typing import List

from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset
from torch import float32, nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
import gensim
import numpy as np

Word  = List[float]
Sentence = List[Word]
Paragraph = List[Sentence]
ParagraphTensor = torch.Tensor

def get_word_embedding_tup(embed, unk_rep, word: str):
    return tuple(embed[word] if word in embed.key_to_index else unk_rep)

def get_sentence_embedding_tup(embed, unk_rep, sentence):
    return tuple(get_word_embedding_tup(embed, unk_rep, word) for word in word_tokenize(sentence))

def get_paragraph_embedding_tup(embed, unk_rep, paragraph):
    return tuple(get_sentence_embedding_tup(embed, unk_rep, sentence) for sentence in sent_tokenize(paragraph))

def listify_word_embedding(word) -> Word:
    return list(word)

def listify_sentence_embedding(sentence) -> Sentence:
    return [listify_word_embedding(word) for word in sentence]

def listify_paragraph_embedding(par_embed) -> Paragraph:
    return [listify_sentence_embedding(sentence) for sentence in par_embed]

def tensor_of_tupled_par_embed(par_embed) -> ParagraphTensor:
    return [torch.FloatTensor(listify_sentence_embedding(sentence)) for sentence in par_embed]
    
class WindowedParDataset(Dataset):
    def __init__(self, paragraphs, labels, embed, window_size=3):
        super().__init__()
        unk = np.mean(embed.vectors, axis=0)
        
        coherent_windows = set()
        incoherent_windows = set()
        
        for i, (paragraph, is_coherent) in enumerate(zip(paragraphs, labels)):
            sentences: Sentence = get_paragraph_embedding_tup(embed, unk, paragraph)
            num_windows: int = len(sentences) - window_size + 1
                
            if num_windows < 0:
                print(f"WARNING: Paragraph {i} did not have enough sentences for window size {window_size}")
                continue
            
            for i in range(num_windows):
                curr_window: Sentence = sentences[i:i+window_size]
                if is_coherent:
                    coherent_windows.add(curr_window)
                elif curr_window not in coherent_windows:
                    incoherent_windows.add(curr_window)
                    
        coherent_windows = [tensor_of_tupled_par_embed(window) for window in coherent_windows]
        incoherent_windows = [tensor_of_tupled_par_embed(window) for window in incoherent_windows]
                    
        self.data = []
        
        def add_windows_with_label(windows, label):
            for window in windows:
                self.data.append({
                    'window': window,
                    'label': label
                })
        
        add_windows_with_label(coherent_windows, 1)
        add_windows_with_label(incoherent_windows, 0)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def basic_collate_fn(batch):
    """Collate function for basic setting."""
    windows = [i['window'] for i in batch]
    labels = [i['label'] for i in batch]
    labels = torch.cat(labels)

    return windows, labels


#######################################################################
################################ Model ################################
#######################################################################

class FFNN(nn.Module):
    """Basic feedforward neural network"""
    def __init__(
        self, 
        ffnn_hidden_dim: int, 
        lstm_hidden_dim: int, 
        is_bidirectional: bool, 
        window_size: int,
        word_vec_length: int = 200
    ):
        super().__init__()
        
        self.bidirec = is_bidirectional
        self.D = 2 if self.bidirec else 1
        self.lstm_hidden_size = lstm_hidden_dim
        
        self.lstm = nn.LSTM(
            word_vec_length, 
            self.lstm_hidden_size, 
            batch_first=False, 
            bidirectional=self.bidirec
        )
        self.lstm_output_dim = self.lstm_hidden_size * self.D
        self.fc1 = nn.Linear(self.lstm_output_dim * window_size, ffnn_hidden_dim)
        self.output = nn.Linear(ffnn_hidden_dim, 1)
    
    def forward(self, windows: List[ParagraphTensor]):
        def lstmForward(l_of_seqs):
            # l_of_seqs shape: batch length * num_words_per_seq (ragged) * 200
            input_lengths = [seq.size(0) for seq in l_of_seqs]
            padded_input = nn.utils.rnn.pad_sequence(l_of_seqs) # tensor w/ shape (max_seq_len, batch_len, 200)
            total_length = padded_input.size(1)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                padded_input, input_lengths, batch_first=False, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_input) # shape (max_seq_len, batch_len, lstm_hidden_dim)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=False, total_length=total_length
            )
            # compute max pooling along the time dimension to collapse into a single lstm_hidden_dim vector
            return torch.max(output, dim=0).values

        to_be_lstmed = [sentence_embed for window in windows for sentence_embed in window]
        rnn_embeddings = lstmForward(to_be_lstmed)
        vs = torch.zeros(
            [len(windows), self.lstm_output_dim * window_size], # num_windows * length of window vector
            dtype=torch.float32
        )
        
        for i, rnn_embedding in enumerate(embeddings):
            curr_window_idx = i / window_size
            sent_idx_in_curr_window = i % window_size
            curr_sent_embed_start = sent_idx_in_curr_window * self.lstm_output_dim 
            curr_sent_embed_end = (sent_idx_in_curr_window + 1) * self.lstm_output_dim  
            vs[curr_window_idx][curr_sent_embed_start : curr_sent_embed_end] = rnn_embedding
        
        vs = F.relu(self.fc1(vs))
        output = torch.transpose(self.output(vs), dim0=0, dim1=1)[0]
        return output


#########################################################################
################################ Training ###############################
#########################################################################

def calculate_loss(scores, labels, loss_fn):
    return loss_fn(scores, labels)

def get_optimizer(net, lr, weight_decay):
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def get_hyper_parameters():
    hidden_dim = [128, 256]
    lr = [1e-3, 5e-4]
    weight_decay = [0, 0.001]

    return hidden_dim, lr, weight_decay


def train_model(net, trn_loader, val_loader, optim, num_epoch=50, collect_cycle=30,
        device='cpu', verbose=True):
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0

    loss_fn = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopper()
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        # Training:
        net.train()
        for questions, context, labels, sent_ids in trn_loader:
            num_itr += 1
            questions = [i.to(device) for i in questions]
            context = [i.to(device) for i in context]
            labels = labels.to(device)
            
            optim.zero_grad()
            output = net(questions, context)
            loss = calculate_loss(output, labels, loss_fn)
            loss.backward()
            optim.step()
            
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))

        # Validation:
        accuracy, loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
        if early_stopper.early_stop(accuracy):
            break
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
    }

    return best_model, stats


def get_predictions(scores: torch.Tensor):
    return torch.IntTensor([1 if score > 0 else 0 for score in scores])


def get_validation_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for questions, context, labels, sent_ids in data_loader:
            questions = [i.to(device) for i in questions]
            context = [i.to(device) for i in context]
            labels = labels.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            scores = net(questions, context)
            loss = calculate_loss(scores, labels, loss_fn)
            pred = torch.IntTensor(get_predictions(questions, context, scores)).to(device)

            total_loss.append(loss.item())
            y_true.append(sent_ids)
            y_pred.append(pred.cpu())
    
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    accuracy = (y_true == y_pred).sum() / y_pred.shape[0]
    total_loss = sum(total_loss) / len(total_loss)
    
    return accuracy, total_loss


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()

class EarlyStopper:
    # Code inspired from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch u/isle_of_gods
    def __init__(self, patience=8):
        self.patience = patience
        self.iters_since_last_inc = 0
        self.max_acc = -float("inf")

    def early_stop(self, curr_val_acc):
        if curr_val_acc >= self.max_acc:
            self.max_acc = curr_val_acc
            self.iters_since_last_inc = 0
        elif curr_val_acc < self.max_acc:
            self.iters_since_last_inc += 1
            if self.iters_since_last_inc >= self.patience:
                return True
        return False

