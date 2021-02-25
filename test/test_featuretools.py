# https://github.com/alteryx/featuretools
import sys
# ROOT_DIR = 'D:/code/pycallgraph'
ROOT_DIR = '/mnt/d/code/pycallgraph'
sys.path.append(f"{ROOT_DIR}")
print(sys.path)

import featuretools as ft
from pycallgraph import PyCallGraph,Config
from pycallgraph.output import GraphvizOutput
from pycallgraph import GlobbingFilter

graphviz = GraphvizOutput()
graphviz.output_file = f"{ROOT_DIR}/featuretools.png"

config = Config(**{"verbose":True})
config.trace_filter = GlobbingFilter(
            exclude=['pycallgraph.*',
            'pandas.*',
            'numpy.*',
            '_ImportLockContext*',
            'SourceFileLoader.*',
            'ModuleSpec.*',
            'dateutil.*',
            '__main__',
            'featuretools.demo.mock_customer.<listcomp>'],
            include=['featuretools.feature_base.*',
            'featuretools.primitives.*',
            'featuretools.selection.*',
            'featuretools.synthesis.*',
            'featuretools.utils.*',
            'featuretools.variable_types.*',
            'featuretools.entityset.*']
            )

tracker_log = f"{ROOT_DIR}/tracker.pkl"

with PyCallGraph(config=config, output=graphviz, tracker_log=tracker_log, package_prefix = 'featuretools.'):
    es = ft.demo.load_mock_customer(return_entityset=True)
    feature_matrix, features_defs = ft.dfs(entityset=es, target_entity="customers")

# pycg = PyCallGraph(config=config, output=graphviz, tracker_log=tracker_log, package_prefix = 'featuretools.')
# pycg.only_output()

print("done")


# python .\test\test_featuretools.py