import os
import itertools
from tqdm import tqdm

def main():
    print("Scoring all datasets with each model type")

    datasets = ['reddit']
    models = ['aviation', 'la', 'wsj']

    for dataset, model in tqdm(itertools.product(datasets, models), total=len(datasets) * len(models)):
        os.system(f"python3 scoreData.py {dataset} {model}")

if __name__ == '__main__':
    main()