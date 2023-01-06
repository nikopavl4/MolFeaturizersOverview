import argparse


def experiment_args():
    parser = argparse.ArgumentParser(description="Perform Machine Learning.")
    parser.add_argument('--seed', type=int, default=0, help="The seed to initialize the random generators.")
    parser.add_argument('--dataset', type=str, default='sider', help="Dataset Name")
    parser.add_argument('--featurizer', type=str, default='circularfingerprint', help="Featurizer Name")
    parser.add_argument('--model', type=str, default='svm', help="Model Name")
    args = parser.parse_args()
    return args
