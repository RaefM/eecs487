import pandas as pd
import gensim.downloader
import itertools
import torch
import sys
from tqdm.notebook import tqdm
from coherenceModel import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def main():
    print("READING DATA")
    paragraph_df = pd.read_csv(sys.argv[1])

    train_pars = paragraph_df[paragraph_df['id'] < 1440]
    dev_pars = paragraph_df[(paragraph_df['id'] >= 1440) & (paragraph_df['id'] < 1800)]
    test_pars = paragraph_df[paragraph_df['id'] >= 1800]

    X_train, y_train = train_pars.paragraph.values, train_pars.is_coherent.values
    X_val, y_val = dev_pars.paragraph.values, dev_pars.is_coherent.values
    X_test, y_test = test_pars.paragraph.values, test_pars.is_coherent.values

    print(X_train[0])
    print(y_train[0])

    print("DOWNLOADING GloVe")
    embed = gensim.downloader.load("glove-wiki-gigaword-50")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    print("MAKING DATASETS")

    ################ SET THIS TO CHANGE WINDOW SIZE OF THINGS BELOW:
    wsize = 5
    ################
    train_data = WindowedParDataset(X_train, y_train, embed, wsize)
    dev_data = WindowedParDataset(X_val, y_val, embed, wsize)
    test_data = WindowedParDataset(X_test, y_test, embed, wsize)
    train_loader = DataLoader(train_data, batch_size=25, collate_fn=basic_collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=25, collate_fn=basic_collate_fn, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=25, collate_fn=basic_collate_fn, shuffle=False)

    pos_weight = torch.Tensor([20]).to(device)

    print("BEGINNING TO TRAIN MODEL")
    def search_param_utterance(wsize):
        """Experiemnt on different hyper parameters."""
        learning_rate, weight_decay = get_hyper_parameters()
        window_sizes = [wsize]
        print("learning rate from: {}\nweight_decay from: {}\nwindow from: {}".format(
            learning_rate, weight_decay, window_sizes
        ))
        best_model, best_stats = None, None
        best_accuracy, best_lr, best_wd, best_window_size = 0, 0, 0, 0
        for lr, wd, window_size in tqdm(itertools.product(learning_rate, weight_decay, window_sizes),
                              total=len(learning_rate) * len(weight_decay) * len(window_sizes)):
            net = FFNN(window_size, device).to(device)
            optim = get_optimizer(net, lr=lr, weight_decay=wd)
            model, stats = train_model(net, train_loader, dev_loader, optim, pos_weight=pos_weight, 
                                      num_epoch=100, collect_cycle=500, device=device, 
                                      verbose=True, patience=5, stopping_criteria='accuracy')
            # print accuracy
            print(f"{(lr, wd, window_size)}: {stats['accuracy']}")
            # update best parameters if needed
            if stats['accuracy'] > best_accuracy:
                best_accuracy = stats['accuracy']
                best_model, best_stats = model, stats
                best_lr, best_wd, best_window_size = lr, wd, window_size
                torch.save(best_model.state_dict(), 'best_rnn_av.pt')
        print("\n\nBest learning rate: {}, best weight_decay: {}, best window: {}".format(
            best_lr, best_wd, best_window_size))
        print("Accuracy: {:.4f}".format(best_accuracy))
        plot_loss(best_stats)
        return best_model
    basic_model = search_param_utterance(wsize)


    print("TESTING FINAL MODEL")
    uar, accuracy, total_loss = get_validation_performance(
        basic_model, 
        nn.BCEWithLogitsLoss(pos_weight=pos_weight), 
        test_loader, 
        device
    )
    print("Final selection:")
    print("Test UAR: {:.4f}".format(uar))
    print("Test accuracy: {:.4f}".format(accuracy))
    print("Test loss: {:.4f}".format(total_loss))


if __name__ == '__main__':
    main()
