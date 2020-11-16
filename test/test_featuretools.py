# https://github.com/alteryx/featuretools
import sys
# sys.path.append("D:/code/pycallgraph/pycallgraph")
sys.path.append("/mnt/d/code/pycallgraph")
# print(sys.path)

import featuretools as ft
from pycallgraph import PyCallGraph,Config

with PyCallGraph(config=Config(**{"verbose":True})):
    es = ft.demo.load_mock_customer(return_entityset=True)
    feature_matrix, features_defs = ft.dfs(entityset=es, target_entity="customers")

print("done")
