import numpy as np
import pandas as pd
import pulp as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

# raw = pd.read_sas('data/accepts.sas7bdat')
raw = pd.read_csv('data/accepts.csv')

raw.drop(['weight'], axis=1, inplace=True)
y = raw['bad']
X = raw.drop(['bad'], axis=1)

le = LabelEncoder()
col_cat = X.columns[X.dtypes == 'object']
for c in col_cat:
    X[c] = le.fit_transform(X[c])
X.fillna(0, inplace=True)

dtree = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=20, random_state=123)
dtree.fit(X, y)

dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=X.columns,
                                class_names='01',
                                filled=True, node_ids=True, proportion=True)

graph = graphviz.Source(dot_data)
graph.render('tree', format='png', cleanup=True)

nodes = dtree.apply(X)
raw['_nodes_'] = nodes
raw['bad_amt'] = raw['bad'] * raw['loan_amt']

nodes_agg = raw.groupby('_nodes_').agg(
    n_samp=('bad', np.size),
    n_bad=('bad', np.sum),
    bad_amt=('bad_amt', np.sum)
)

# pl.listSolvers(onlyAvailable=True)
# solver = pl.GLPK_CMD(path='C:\\opt\\glpk-4.65\\w64\\glpsol.exe')
solver = pl.PULP_CBC_CMD()

BAD_RATIO_LMT = 0.1
BAD_AMT_RATIO = 5e6

prob = pl.LpProblem('Decison_Tree_Nodes_Selection', sense=pl.LpMaximize)
w = [pl.LpVariable('w'+str(i), 0, 1, pl.LpBinary)
     for i in nodes_agg.index]
prob += pl.lpSum([nodes_agg.n_samp[i] * w[j]
                  for j, i in enumerate(nodes_agg.index)
                  ])
prob += pl.lpSum([nodes_agg.n_bad[i] * w[j]
                  for j, i in enumerate(nodes_agg.index)]) \
    <= pl.lpSum([nodes_agg.n_samp[i] * w[j]
                 for j, i in enumerate(nodes_agg.index)]) * BAD_RATIO_LMT
prob += pl.lpSum([nodes_agg.bad_amt[i] * w[j]
                  for j, i in enumerate(nodes_agg.index)]) <= BAD_AMT_RATIO

prob.writeLP('dt_opt.lp')
prob.solve(solver)
print('Status:', pl.LpStatus[prob.status])

wd = {wi.name: wi.value() for wi in w}
print(wd)
print('bad rate:', np.dot(nodes_agg.n_bad, list(wd.values()))
      / np.dot(nodes_agg.n_samp, list(wd.values())))
print('loss amt:', np.dot(nodes_agg.bad_amt, list(wd.values())))
