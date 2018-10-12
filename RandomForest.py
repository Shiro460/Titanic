#import packages
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import pip

# datasetが格納されているdatasetディレクトリまでのパス指定
#os.chdir("C:\Users\shiro\Dropbox\main\Training\Python\RandomForest\Titanic\dataset")
df = pd.read_csv('train.csv')
print(df)

plt.show(sns.countplot('Sex', hue='Survived', data=df))

# データ加工
from sklearn.model_selection import train_test_split
#欠損値処理
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna('S')

#カテゴリ変数の変換
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
train_X = df.drop('Survived', axis=1)
train_y = df.Survived
(train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 666)

# scikit-learnの中のtreeライブラリを使用
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)

# 検証データを使ってテスト
from sklearn.metrics import (roc_curve, auc, accuracy_score)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
print("決定木による分析結果")
print("AUC: ")
print(auc(fpr, tpr))
print("Accuracy score ")
print(accuracy_score(pred, test_y))

import pydotplus

from IPython.display import Image
from graphviz import Digraph
from sklearn.externals.six import StringIO
from sklearn import tree

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=train_X.columns, max_depth=3)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
'''
*******************************
       ここでエラー発生
       あきらめました
******************************* 
#graph.write_pdf("graph.pdf")
#Image('graph.pdf')
'''

#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
print("ランダムフォレストを使った結果")
print("AUC: ")
print(auc(fpr, tpr))
print("Accuracy score ")
print(accuracy_score(pred, test_y))

'''
*******************************
あまり精度は上がらなかった。
元が良すぎるのかも？
でも今後試してみる価値はある。
特に込み入った議論の時はまず機械学習をやってみて
data-drivenで考慮してみるとブレークスルーが起きるかも
******************************* 
'''





