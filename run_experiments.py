import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'main.py')

print("Starting Running MolFeaturizers for Graph Classification")
simple_models = ['svm', 'randomforest', 'gradboost']
#simple_featurizers = ['maccsfingerprint', 'circularfingerprint', 'mol2vecfingerprint', 'rdkitdescriptors', 'mordreddescriptors', 'coulombmateig', 'bpsymmetry', 'onehot']
simple_featurizers = ['bpsymmetry']
seeds = [97547, 74184, 21094, 96107, 58890, 45103, 43181, 14704, 67601, 26513, 64972, 56930, 35609, 48968, 49530, 55602, 34754, 97245, 13277, 13575]
#datasets = ['clintox', 'sider', 'toxcast', 'hiv']
datasets = ['clintox']

for dataset in datasets:
    print(f'{dataset} Dataset')

    for model in simple_models:
        for feat in simple_featurizers:
            for seed in seeds:
                #print('--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}')
                sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}'])

    # Graph Convolution
    model = 'convmol'
    feat = 'convmol'
    for seed in seeds:
        #print('--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}'])

    # Weave
    model = 'weave'
    feat = 'weave'
    for seed in seeds:
        #print('--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}')
        sub.call(["python" , filename, '--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}'])
