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
config.package_prefix = 'featuretools.'
config.full_func_name_file = f"{ROOT_DIR}/full_func_name.log"
config.func_name_prune = {
            'primitives.*.*.*.*':2,
            'variable_types.*.*.*':1,
            'feature_base.*.*.*':1,
            }
config.trace_filter = GlobbingFilter(
            exclude=['pycallgraph.*',
            'pandas.*',
            'numpy.*',
            '_ImportLockContext*',
            'SourceFileLoader.*',
            'ModuleSpec.*',
            'dateutil.*',
            '__main__',
            # 'featuretools.utils.*.*',
            'featuretools.demo.mock_customer.<listcomp>'],
            include=['featuretools.feature_base.*',
            'featuretools.primitives.*',
            'featuretools.selection.*',
            'featuretools.synthesis.*',
            # 'featuretools.utils.*',
            'featuretools.variable_types.*',
            'featuretools.entityset.*']
            )

config.tracker_log = f"{ROOT_DIR}/tracker.pkl"

# with PyCallGraph(config=config, output=graphviz):
#     es = ft.demo.load_mock_customer(return_entityset=True)
#     feature_matrix, features_defs = ft.dfs(entityset=es, target_entity="customers")

pycg = PyCallGraph(config=config, output=graphviz)
pycg.only_output()

print("done")


# python .\test\test_featuretools.py
# sort full_func_name.log -o full_func_name.log