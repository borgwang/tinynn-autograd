"""Example code for MNIST. A fully-connected network and a convolutional neural network were implemented."""

import runtime_path  # isort:skip

import matplotlib.pyplot as plt
import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np

from core.nn.net import SequentialNet
from core.nn.layers import Dense, ReLU
from core.nn.loss import SoftmaxCrossEntropyLoss
from core.nn.optimizer import Adam
from core.tensor import Tensor
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.evaluator import AccEvaluator
from env import DEBUG, GRAPH

import networkx as nx

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
    with gzip.open(save_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def build_graph(node, G):
    if id(node) not in G.nodes:
        G.add_node(id(node), name=node.name, shape=node.shape, outdegree=node.outdegree, bwdcost=node.bwdcost)
        for dep in node.dependency:
            subnode = dep["tensor"]
            G = build_graph(subnode, G)
            edge = (id(node), id(subnode))
            if edge in G.edges:
                cnt = nx.get_edge_attributes(G, "cnt")[edge]["cnt"]
                nx.set_edge_attributes(G, {edge: {"cnt": cnt+1}})
            else:
                G.add_edge(*edge, cnt=1)
    return G

def plot_graph(start):
    G = nx.Graph()
    G = build_graph(start, G)
    plt.figure(figsize=(24, 20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[u, v] = f"{data['cnt']}"

    total_bwdcost = 0
    node_labels = {}
    for n, data in G.nodes(data=True):
        node_labels[n] = f"{data['name']}\n{data['shape']}\nbwdcosst: {data['bwdcost']:.4f}s"
        if GRAPH: print(f"node: {data['name']} cost: {data['bwdcost']:.6f}")
        total_bwdcost += data["bwdcost"]
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    if GRAPH: print(f"total_bwdcost: {total_bwdcost:.4f}")
    plt.savefig("test.png")
    sys.exit()

def main(args):
    if args.seed >= 0:
        np.random.seed(args.seed)

    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)
    train_x = Tensor(train_x)
    train_y = Tensor(train_y)
    test_x = Tensor(test_x).to(args.device)
    test_y = Tensor(test_y)

    net = SequentialNet(
            Dense(256), ReLU(),
            Dense(128), ReLU(),
            Dense(64), ReLU(),
            Dense(32), ReLU(),
            Dense(10)).to(args.device)
    optim = Adam(net.get_parameters(), lr=args.lr)
    loss_fn = SoftmaxCrossEntropyLoss()

    iterator = BatchIterator(batch_size=args.batch_size)
    evaluator = AccEvaluator()
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            net.zero_grad()
            x, y = batch.inputs.to(args.device), batch.targets.to(args.device)
            pred = net.forward(x)
            loss = loss_fn(pred, y)
            if GRAPH: ts = time.time()
            loss.backward()
            if GRAPH: print("loss.backward() cost: ", time.time() - ts)
            if GRAPH: plot_graph(loss)
            optim.step()
            if args.onepass: sys.exit()
        print("Epoch %d tim cost: %.4f" % (epoch, time.time() - t_start))
        if args.eval:
            test_pred = net.forward(test_x).numpy()
            test_pred_idx = np.argmax(test_pred, axis=1)
            test_y_idx = test_y.numpy()
            print(evaluator.evaluate(test_pred_idx, test_y_idx))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)

    parser.add_argument("--onepass", default=0, type=int)
    parser.add_argument("--eval", default=0, type=int)
    parser.add_argument("--device", default="gpu", type=str)
    args = parser.parse_args()
    main(args)
