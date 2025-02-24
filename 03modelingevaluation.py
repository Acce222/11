# 步骤1：导入模块
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
import warnings

warnings.filterwarnings("ignore")

# 步骤2；数据预处理
df = pd.read_csv("pre1.csv")
y = df["posttest"]
x = df.drop("posttest", 1)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=456)

onehot = OneHotEncoder()
x_train_onehot = onehot.fit_transform(x_train).toarray()
x_test_onehot = onehot.fit_transform(x_test).toarray()

# 步骤3；模型构建和调参
# lasso回归
l1 = Lasso()
pg = {"alpha": [0.01, 0.03, 0.05, 0.08, 0.1]}
model = GridSearchCV(l1, pg)
model.fit(x_train_onehot, y_train)
print("L1最优参数为：{}".format(model.best_params_))
print("L1最优得分为：{}".format(model.best_score_))
print("L1测试得分：{}".format(model.score(x_test_onehot, y_test)))

# 岭回归
l2 = Ridge()
pg = {"alpha": [0.01, 0.03, 0.05, 0.08, 0.1]}
model = GridSearchCV(l2, pg)
model.fit(x_train_onehot, y_train)
print("L2最优参数为：{}".format(model.best_params_))
print("L2最优得分为：{}".format(model.best_score_))
print("L2测试得分：{}".format(model.score(x_test_onehot, y_test)))

# 决策树
tree = DecisionTreeRegressor()
pg = {"max_depth": [7, 8, 9, 10]}
model = GridSearchCV(tree, pg)
model.fit(x_train_onehot, y_train)
print("决策树最优参数:{}".format(model.best_params_))
print("决策树最优得分:{}".format(model.best_score_))
print("决策树测试得分:{}".format(model.score(x_test_onehot, y_test)))

# 随机森林
rf = RandomForestRegressor()
pg = {"max_depth": [6, 7, 8], "n_estimators": [100, 150, 200]}
model = GridSearchCV(rf, pg)
model.fit(x_train_onehot, y_train)
print("RF最优参数:{}".format(model.best_params_))
print("RF最优得分:{}".format(model.best_score_))
print("RF测试得分:{}".format(model.score(x_test_onehot, y_test)))

# kNN
knn = KNeighborsRegressor()
pg = {"n_neighbors":[5,6,7,8]}
model.fit(x_train_onehot,y_train)
print("KNN最优参数:{}".format(model.best_params_))
print("KNN最优得分:{}".format(model.best_score_))
print("KNN测试得分:{}".format(model.score(x_test_onehot, y_test)))

# 最优算法
model = RandomForestRegressor(max_depth=8,n_estimators=100)
model.fit(x_train_onehot,y_train)
print("最优得分：{}".format(model.score(x_test_onehot,y_test)))

y_ = model.predict(x_test_onehot)
num = len(y_)
plt.plot(np.arange(num),y_,label="预测值")
plt.plot(np.arange(num),y_test,label="实际值")
plt.legend()
plt.show()