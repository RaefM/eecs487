import gensim.downloader
import torch
import sys
import pandas as pd
from tqdm.notebook import tqdm
from coherenceModelNews import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 getTestPerformance.py <csv with ids> <model params file>")

    print("READING DATA")
    paragraph_df = pd.read_csv(sys.argv[1])

    test_pars = paragraph_df[paragraph_df['id'] >= 1800]
    X_test, y_test = test_pars.paragraph.values, test_pars.is_coherent.values

    print("LOADING GloVe")
    embed = gensim.models.KeyedVectors.load('glove100.kv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("MAKING DATASETS")

    ################ SET THIS TO CHANGE WINDOW SIZE OF THINGS BELOW:
    wsize = 5
    ################
    test_data = WindowedParDataset(X_test, y_test, embed, wsize)
    test_loader = DataLoader(test_data, batch_size=25, collate_fn=basic_collate_fn, shuffle=False)

    pos_weight = torch.Tensor([20]).to(device)

    print("LOADING SPECIFIED MODEL")
    basic_model = FFNN(5, True, device)
    basic_model.load_state_dict(torch.load(sys.argv[2]))
    basic_model.eval()
    basic_model.to(device)

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