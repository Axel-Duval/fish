import statistics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

with open('./fish.csv', newline='') as file:
    dataset = pd.read_csv(file, delimiter=';')

    print(dataset.shape)
    print(dataset.head())
    print(dataset.keys())
    print(dataset.info())
    print(dataset.describe())

    sns.pairplot(dataset, hue='Species')

    dataset.hist(bins=30)
    plt.show()

    weight_mean = dataset['Weight'].mean()
    dataset['Weight'] = dataset['Weight'].replace(to_replace=0, value=weight_mean)

    X = dataset.drop('Species', axis=1)
    y = dataset['Species']

    lr = LogisticRegression(penalty='l2', tol=0.0000001)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=555)

    lr.fit(X_train, y_train)

    prediction = lr.predict(X_test)
    print(prediction)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    ########################## reproductibility ##########################
    # p = []

    # for i in range(100):

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    #     lr.fit(X_train, y_train)

    #     prediction = lr.predict(X_test)

    #     p.append(classification_report(y_test, prediction,output_dict=True)["accuracy"])
    
    # print(statistics.mean(p))
