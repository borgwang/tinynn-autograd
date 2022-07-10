import time
import sys

import numpy as np

from core.tensor import Tensor

np.random.seed(0)

from core.ndarray import GPUArray


BS = 2**6
idim = 2**8
odim = 2**6

data_x = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
data_y = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
data_w = np.random.normal(0, 1, (idim, odim)).astype(np.float32)
data_b = np.zeros((1, odim)).astype(np.float32)

n_ep = 10


def build_graph(node, G):
    if id(node) not in G.nodes:
        G.add_node(id(node), name=node.name)
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
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G = build_graph(start, G)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    edge_labels = nx.get_edge_attributes(G, "cnt")
    nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)
    node_labels = nx.get_node_attributes(G, "name")
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.show()
    sys.exit()


def run_gpu():
    print("---- GPU -----")
    x = Tensor(data_x).gpu()
    y = Tensor(data_y).gpu()
    w = Tensor(data_w, requires_grad=True, name="w").gpu()
    b = Tensor(data_b, requires_grad=True, name="b").gpu()

    G = GraphVis()
    t0 = time.time()
    for epoch in range(n_ep):
        w.zero_grad()
        b.zero_grad()
        pred = x @ w + b
        err = pred - y
        loss = (err**2).sum()
        plot_graph(loss)
        loss.backward()
        w -= 0.0001 * w.grad
        b -= 0.0001 * b.grad

    t1 = time.time()
    print(f"GPU compute cost: {t1 - t0:.5f} s")
    #print(f"err check: {err.values.numpy().sum():.8f}")
    print(f"loss check: {loss.values.numpy():.8f}")


def run_cpu():
    print("---- CPU -----")
    x, y, w, b = data_x, data_y, data_w, data_b

    t0 = time.time()
    for epoch in range(n_ep):
        pred = x @ w + b
        err = pred - y
        loss = (err * err).sum()
        dw = x.T @ (2 * err)
        db = (2 * err).sum(axis=0, keepdims=True)
        w -= 0.0001 * dw
        b -= 0.0001 * db
    t1 = time.time()
    print(f"CPU compute cost: {t1 - t0:.3f}s")
    print(f"err check: {err.sum():.8f}")
    print(f"loss check: {loss:.8f}")

run_gpu()
#run_cpu()

