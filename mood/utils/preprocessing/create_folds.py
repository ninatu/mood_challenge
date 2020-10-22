import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import argparse


def create_folds(input_folder: str, output_path: str, n_folds: int):
    files = os.listdir(input_folder)
    files = [filename for filename in files if os.path.splitext(filename)[1] == '.gz']
    np.random.shuffle(files)
    files = np.array(files)

    df = pd.DataFrame(index=files, columns=['test_fold'])
    df.index.name = 'filename'

    for i_fold, (train_index, test_index) in enumerate(KFold(n_splits=n_folds).split(files)):
        test_patients = files[test_index]
        df.loc[test_patients] = i_fold

    df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_dir",
                        default='./data/original/brain_train/',
                        help='input_dir')
    parser.add_argument("-o", "--output_path",
                        default='./folds/brain/train_folds_10.csv',
                        help='output_path')
    parser.add_argument("-n", "--n_folds",
                        type=int,
                        default=10,
                        help='n_folds')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path
    n_folds = args.n_folds

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    create_folds(input_dir, output_path, n_folds)
