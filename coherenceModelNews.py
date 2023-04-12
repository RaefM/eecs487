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
from sklearn.metrics import balanced_accuracy_score, accuracy_score

Word  = List[float]
Sentence = List[Word]
Paragraph = List[Sentence]
ParagraphTensor = torch.Tensor

def get_word_embedding_tup(embed, unk_rep, word: str):
    return tuple(embed[word] if word in embed.key_to_index else unk_rep)

def get_sentence_embedding_tup(embed, unk_rep, sentence):
    return tuple(get_word_embedding_tup(embed, unk_rep, word.lower()) for word in word_tokenize(sentence))

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
        self.windows = []
        self.labels = []
        
        for paragraph, is_coherent in zip(paragraphs, labels):
            sentences: Sentence = get_paragraph_embedding_tup(embed, unk, paragraph)
            num_windows: int = len(sentences) - window_size + 1
                
            if num_windows < 0:
                print(f"WARNING: Paragraph {i} did not have enough sentences for window size {window_size}")
                continue
            
            for i in range(num_windows):
                self.windows.append(tensor_of_tupled_par_embed(sentences[i:i+window_size]))
                self.labels.append(is_coherent)
                
        print("Complete")
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'window': self.windows[idx],
            'label': self.labels[idx]
        }
    


def basic_collate_fn(batch):
    """Collate function for basic setting."""
    windows = [i['window'] for i in batch]
    labels = torch.IntTensor([i['label'] for i in batch])

    return windows, labels


#######################################################################
################################ Model ################################
#######################################################################

class FFNN(nn.Module):
    """Basic feedforward neural network"""
    def __init__(
        self,
        window_size: int,
        bidirec,
        device
    ):
        super().__init__()
        word_vec_length = 100
        ffnn_hidden_dim = 500
        self.lstm_hidden_size = word_vec_length
        self.window_size = window_size
        self.device = device
        
        D = 2 if bidirec else 1
        self.lstm = nn.GRU(
            word_vec_length, 
            self.lstm_hidden_size, 
            batch_first=False, 
            bidirectional=bidirec,
        )
        self.lstm_output_dim = self.lstm_hidden_size * D
        self.fc1 = nn.Linear(self.lstm_output_dim * window_size, ffnn_hidden_dim)
        self.output = nn.Linear(ffnn_hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.uniform_(self.lstm.__getattr__(p), -0.2, 0.2)
    
    def forward(self, windows: List[ParagraphTensor]):
        def rnnForward(l_of_seqs):
            # l_of_seqs shape: batch length * num_words_per_seq (ragged) * 200
            input_lengths = [seq.size(0) for seq in l_of_seqs]
            padded_input = nn.utils.rnn.pad_sequence(l_of_seqs) # tensor w/ shape (max_seq_len, batch_len, 200)
            total_length = padded_input.size(0)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                padded_input, input_lengths, batch_first=False, enforce_sorted=False
            )
#             _, hn = self.lstm(packed_input) # shape (max_seq_len, batch_len, lstm_hidden_dim)
#             return hn[0]
            packed_output, _ = self.lstm(packed_input) # shape (max_seq_len, batch_len, lstm_hidden_dim)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=False, total_length=total_length
            )
            # compute max pooling along the time dimension to collapse into a single lstm_hidden_dim vector
            return torch.max(output, dim=0).values

        to_be_rnned = [sentence_embed for window in windows for sentence_embed in window]
        rnn_embeddings = rnnForward(to_be_rnned)
        vs = torch.zeros(
            [len(windows), self.lstm_output_dim * self.window_size], # num_windows * length of window vector
            dtype=torch.float32
        ).to(self.device)
        
        for i, rnn_embedding in enumerate(rnn_embeddings):
            curr_window_idx = i // self.window_size
            sent_idx_in_curr_window = i % self.window_size
            curr_sent_embed_start = sent_idx_in_curr_window * self.lstm_output_dim 
            curr_sent_embed_end = (sent_idx_in_curr_window + 1) * self.lstm_output_dim  
            vs[curr_window_idx][curr_sent_embed_start : curr_sent_embed_end] = rnn_embedding
        
        vs = torch.tanh(self.fc1(vs))
        output = torch.transpose(self.output(vs), dim0=0, dim1=1)[0]
        return output


#########################################################################
################################ Training ###############################
#########################################################################

def calculate_loss(scores, labels, loss_fn):
    return loss_fn(scores, labels.float())

def get_optimizer(net, lr, weight_decay):
    return optimizer.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay)

def get_hyper_parameters():
    lr = [0.01]
    weight_decay = [Q / 50 for Q in [0.01, 0.1, 0.25, 0.5]]

    return lr, weight_decay


def print_grads(model):
    for name, param in model.named_parameters():
        print(name, param.grad.norm())

def train_model(net, trn_loader, val_loader, optim, pos_weight=None, num_epoch=50, collect_cycle=30,
        device='cpu', verbose=True, patience=8, stopping_criteria='loss'):
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_uar = None, 0

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopperLoss(patience) if stopping_criteria == 'loss' else EarlyStopperAcc(patience)
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        # Training:
        net.train()
        for windows, labels in trn_loader:
            num_itr += 1
            windows = [[s.to(device) for s in window] for window in windows]
            labels = labels.to(device)
            
            optim.zero_grad()
            output = net(windows)
            loss = calculate_loss(output, labels, loss_fn)
            loss.backward()
            optim.step()
            
#             print_grads(net)
            
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
        uar, accuracy, loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation UAR: {:.4f}".format(uar))
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if uar > best_uar:
            best_model = copy.deepcopy(net)
            best_uar = uar
        if patience is not None and early_stopper.early_stop(
            loss if stopping_criteria == 'loss' else uar
        ):
            break
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_uar,
    }

    return best_model, stats


def get_predictions(scores: torch.Tensor):
    probs = torch.sigmoid(scores)
    return torch.IntTensor([1 if prob > 0.5 else 0 for prob in probs])


def get_validation_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for windows, labels in data_loader:
            windows = [[s.to(device) for s in window] for window in windows]
            labels = labels.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            scores = net(windows)
            loss = calculate_loss(scores, labels, loss_fn)
            pred = torch.IntTensor(get_predictions(scores)).to(device)

            total_loss.append(loss.item())
            y_true.append(labels.cpu())
            y_pred.append(pred.cpu())
    
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    uar = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    total_loss = sum(total_loss) / len(total_loss)
    
    return uar, accuracy, total_loss


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()

class EarlyStopperAcc:
    def __init__(self, patience=5):
        self.patience = patience
        self.iters_below = 0
        self.iters_staying_same = 0
        self.max_acc = -float("inf")

    def early_stop(self, curr_acc):
        if curr_acc > self.max_acc:
            self.max_acc = curr_acc
            self.iters_below = 0
            self.iters_staying_same = 0
        elif curr_acc == self.max_acc:
            self.iters_staying_same += 1
            if self.iters_staying_same >= 50:
                return True
        elif curr_acc < self.max_acc:
            self.iters_below += 1
            self.iters_staying_same += 1
            if self.iters_below >= self.patience or self.iters_staying_same >= 50:
                return True
        return False

class EarlyStopperLoss:
    # Code inspired from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch u/isle_of_gods
    def __init__(self, patience=10):
        self.patience = patience
        self.iters_since_last_dec = 0
        self.min_loss = float("inf")

    def early_stop(self, curr_loss):
        if curr_loss < self.min_loss:
            self.min_loss = curr_loss
            self.iters_since_last_dec = 0
        elif curr_loss >= self.min_loss:
            self.iters_since_last_dec += 1
            if self.iters_since_last_dec >= self.patience:
                return True
        return False

