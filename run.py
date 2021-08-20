from model import RGAT,Model
from dataloader import load_Amazon,load_tutorial,load_Amazon_v2,load_ogbl_collab
from utils import construct_negative_graph,compute_loss,compute_auc
import numpy as np
import dgl
import torch



def run_tutorial():
    hetero_graph=load_tutorial()
    print("[INFO]: Graph Construction Completed!")

    k = 5
    model = Model(128, 20, 5, 1, hetero_graph.etypes)
    print("[INFO]: RGAT model Construction Completed!")


    user_feats=hetero_graph.nodes['user'].data['feature']
    item_feats=hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    print("[INFO]: Data Preparation Completed!")


    opt = torch.optim.Adam(model.parameters())
    print("[INFO]: Optimizer Completed!")


    print("[INFO]: Train Process Begin!")
    for epoch in range(100):
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'click', 'item'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('user', 'click', 'item'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


    with torch.no_grad():
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'click', 'item'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('user', 'click', 'item'))
        print('AUC', compute_auc(pos_score, neg_score))



def run_Amazon():
    hetero_graph = load_Amazon_v2()
    print("[INFO]: Graph Construction Completed!")

    k = 5
    model = Model(128, 20, 5, 1, hetero_graph.etypes)
    print("[INFO]: RGAT model Construction Completed!")

    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}

    user_feats_test = hetero_graph.nodes['user'].data['feature']
    item_feats_test = hetero_graph.nodes['item'].data['feature']
    node_features_test = {'user': user_feats_test, 'item': item_feats_test}

    print("[INFO]: Data Preparation Completed!")

    opt = torch.optim.Adam(model.parameters())
    print("[INFO]: Optimizer Completed!")

    print("[INFO]: Train Process Begin!")
    for epoch in range(1000):
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'rate', 'item'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('user', 'rate', 'item'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print("Epoch[{}] || Loss: {} ".format(epoch, loss))

    with torch.no_grad():
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'rate', 'item'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features_test, ('user', 'rate', 'item'))
        print('AUC', compute_auc(pos_score, neg_score))


def run_ogbl_collab():
    # cannot directly use it!!!
    # Not hetero graph
    hetero_graph=load_ogbl_collab()
    print("[INFO]: Graph Construction Completed!")


    k = 19
    model = Model(128, 20, 19, 1, hetero_graph.etypes)
    print("[INFO]: RGAT model Construction Completed!")

    writer_feats = hetero_graph.nodes['_N'].data['feature']
    node_features = {'writer': writer_feats}
    writer_feats_test = hetero_graph.nodes['_N'].data['feature']
    node_features_test = {'writer': writer_feats_test}
    print("[INFO]: Data Preparation Completed!")

    opt = torch.optim.Adam(model.parameters())
    print("[INFO]: Optimizer Completed!")

    print("[INFO]: Train Process Begin!")
    for epoch in range(10):
        negative_graph = construct_negative_graph(hetero_graph, k, ('_N', '_E', '_N'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('_N', '_E', '_N'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Epoch[{}] || Loss: {} ".format(epoch, loss))

    with torch.no_grad():
        negative_graph = construct_negative_graph(hetero_graph, k, ('_N', '_E', '_N'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features_test, ('_N', '_E', '_N'))
        print('AUC', compute_auc(pos_score, neg_score))








