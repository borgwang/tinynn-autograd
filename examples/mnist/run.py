"""Example code for MNIST. A fully-connected network and a convolutional neural network were implemented."""

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np

from core.evaluator import AccEvaluator
from core.layers import Dense
from core.layers import ReLU
from core.losses import SoftmaxCrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import Adam, SGD
from core.tensor import Tensor
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset(data_dir):
    url = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path)
    except Exception as e:
        print('Error downloading dataset: %s' % str(e))
        sys.exit(1)
    # load the dataset
    with gzip.open(save_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


@profile
def main(args):
    if args.seed >= 0:
        random_seed(args.seed);

    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    train_x = Tensor(train_x)
    train_y = Tensor(train_y)
    test_x = Tensor(test_x).gpu()
    test_y = Tensor(test_y)

    net = Net([
        Dense(200),
        ReLU(),
        Dense(100),
        ReLU(),
        Dense(70),
        ReLU(),
        Dense(30),
        ReLU(),
        Dense(10)
    ]).gpu()

    model = Model(net=net, loss=SoftmaxCrossEntropyLoss(), optimizer=SGD(lr=args.lr))
    loss_layer = SoftmaxCrossEntropyLoss()
    iterator = BatchIterator(batch_size=args.batch_size)
    evaluator = AccEvaluator()
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            model.zero_grad()
            x, y = batch.inputs.gpu(), batch.targets.gpu()
            pred = model.forward(x)
            loss = loss_layer.loss(pred, y)
            loss.backward()
            model.step()
        print("Epoch %d tim cost: %.4f" % (epoch, time.time() - t_start))
        # evaluate
        model.set_phase("TEST")
        test_pred = model.forward(test_x).cpu()
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = test_y.values
        res = evaluator.evaluate(test_pred_idx, test_y_idx)
        print(res)
        model.set_phase("TRAIN")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
