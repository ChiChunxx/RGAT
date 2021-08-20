import torch
import pandas as pd
import numpy as np
import dgl
from ogb.linkproppred import DglLinkPropPredDataset

def load_Amazon():
    # read all raw files
    item_brand = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/item_brand.dat', sep = ",", header=None).values
    item_category = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/item_category.dat', sep = ",", header=None).values
    item_view = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/item_view.dat', sep = ",", header=None).values
    user_item = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/user_item.dat', sep = "\t", header=None).values

    # stastic
    n_users = user_item[:, 0].max() + 1
    n_items = user_item[:, 1].max() + 1
    n_brands = item_brand[:, 1].max() + 1
    n_categorys = item_category[:, 1].max() + 1
    n_views = item_view[:, 1].max() + 1
    n_rates = user_item.shape[0]

    stastics={
        "n_users": n_users,
        "n_items": n_items,
        "n_brands": n_brands,
        "n_categorys": n_categorys,
        "n_views": n_views,
        "n_rates": n_rates,
    }
    #print(stastics)


    # construct the whole graph
    hetero_graph = dgl.heterograph({
        ('item', 'belong', 'brand'): (item_brand[:,0], item_brand[:,1]),
        ('brand', 'has', 'item'): (item_brand[:,1], item_brand[:,0]),
        ('item', 'attribute', 'category'): (item_category[:,0], item_category[:,1]),
        ('category', 'hold', 'item'): (item_category[:,1], item_category[:,0]),
        ('item', 'look', 'view'): (item_view[:,0], item_view[:,1]),
        ('view', 'looked_at', 'item'): (item_view[:,1], item_view[:,0]),
        ('user', 'rate', 'item'): (user_item[:,0], user_item[:,1]),
        ('item', 'rated-by', 'user'): (user_item[:,1], user_item[:,0]),
    })

    # assign node and edge feature
    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, 128)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, 128)
    hetero_graph.nodes['brand'].data['feature'] = torch.randn(n_brands, 128)
    hetero_graph.nodes['category'].data['feature'] = torch.randn(n_categorys, 128)
    hetero_graph.nodes['view'].data['feature'] = torch.randn(n_views, 128)

    hetero_graph.edges['rate'].data['feature'] = torch.randn(n_rates, 128)
    hetero_graph.edges['rated-by'].data['feature'] = hetero_graph.edges['rate'].data['feature']
    hetero_graph.edges['look'].data['feature'] = torch.randn(item_view.shape[0], 128)
    hetero_graph.edges['looked_at'].data['feature'] = hetero_graph.edges['look'].data['feature']
    hetero_graph.edges['attribute'].data['feature'] = torch.randn(item_category.shape[0], 128)
    hetero_graph.edges['hold'].data['feature'] = hetero_graph.edges['attribute'].data['feature']
    hetero_graph.edges['belong'].data['feature'] = torch.randn(item_brand.shape[0], 128)
    hetero_graph.edges['has'].data['feature'] = hetero_graph.edges['belong'].data['feature']

    hetero_graph.edges['rate'].data['star'] = torch.from_numpy(user_item[:, 2])

    return hetero_graph

def load_Amazon_v2():
    # read all raw files
    item_brand = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/item_brand.dat', sep = ",", header=None).values
    item_category = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/item_category.dat', sep = ",", header=None).values
    item_view = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/item_view.dat', sep = ",", header=None).values
    user_item = pd.read_csv('/home/cchi/cchi/dataset/DGL_Dataset/Amazon/user_item.dat', sep = "\t", header=None).values[0:10000]

    # stastic
    n_users = user_item[:, 0].max() + 1
    n_items = user_item[:, 1].max() + 1
    n_brands = item_brand[:, 1].max() + 1
    n_categorys = item_category[:, 1].max() + 1
    n_views = item_view[:, 1].max() + 1
    n_rates = user_item.shape[0]

    stastics={
        "n_users": n_users,
        "n_items": n_items,
        "n_brands": n_brands,
        "n_categorys": n_categorys,
        "n_views": n_views,
        "n_rates": n_rates,
    }
    #print(stastics)


    # construct the whole graph
    hetero_graph = dgl.heterograph({
        ('user', 'rate', 'item'): (user_item[:,0], user_item[:,1]),
        ('item', 'rated-by', 'user'): (user_item[:,1], user_item[:,0])
    })

    # assign node and edge feature
    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, 128)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, 128)

    hetero_graph.edges['rate'].data['feature'] = torch.randn(n_rates, 128)
    hetero_graph.edges['rated-by'].data['feature'] = hetero_graph.edges['rate'].data['feature']

    hetero_graph.edges['rate'].data['star'] = torch.from_numpy(user_item[:, 2])

    return hetero_graph


def load_tutorial():

    # stastic
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 128
    n_user_classes = 5
    n_max_clicks = 10



    # construct the whole graph
    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    hetero_graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})

    # Stastic
    print("Stastic of DGL Tutorial Dataset:")
    print("[Node]: {}".format(hetero_graph.num_nodes()))
    print("[Edge]: {}".format(hetero_graph.num_edges()))


    # assign node and edge feature
    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
    hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()

    hetero_graph.edges['follow'].data['feature'] = torch.randn(n_follows, n_hetero_features)
    hetero_graph.edges['followed-by'].data['feature'] = hetero_graph.edges['follow'].data['feature']
    hetero_graph.edges['click'].data['feature'] = torch.randn(n_clicks, n_hetero_features)
    hetero_graph.edges['clicked-by'].data['feature'] = hetero_graph.edges['click'].data['feature']
    hetero_graph.edges['dislike'].data['feature'] = torch.randn(n_dislikes, n_hetero_features)
    hetero_graph.edges['disliked-by'].data['feature'] = hetero_graph.edges['dislike'].data['feature']

    # 在user类型的节点和click类型的边上随机生成训练集的掩码
    hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
    hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)

    return hetero_graph


def load_ogbl_collab():
    dataset = DglLinkPropPredDataset('ogbl-collab', root='/home/cchi/cchi/dataset')
    hetero_graph = dataset[0]

    # Param setting
    n_hetero_features=128

    # Stastic
    print("Stastic of OGBL Collab Dataset:")
    print("[Node]: {}".format(hetero_graph.num_nodes()))
    print("[Edge]: {}".format(hetero_graph.num_edges()))
    print("[ntypes]: {}".format(hetero_graph.ntypes))
    print("[etypes]: {}".format(hetero_graph.etypes))
    print("Types of ndata: {}".format(len(hetero_graph.ndata)))
    print("Types of edata: {}".format(len(hetero_graph.edata)))

    # assign node feature
    hetero_graph.nodes['_N'].data['feature'] = torch.randn(hetero_graph.num_nodes(), n_hetero_features)
    hetero_graph.edges['_E'].data['feature'] = torch.randn(hetero_graph.num_edges(), n_hetero_features)

    return hetero_graph

def load_FB15k237Dataset():
    dataset = dgl.data.FB15k237Dataset()
    g=dataset[0]

    '''
    g.ndata['ntype'].max()  tensor(0)
    g.edata['etype'].max()  tensor(473)
    '''

    return g