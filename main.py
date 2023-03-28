"""
1. name: Name of the country concerned
2. landmass: 1=N.America, 2=S.America, 3=Europe, 4=Africa, 4=Asia, 6=Oceania
3. zone: Geographic quadrant, based on Greenwich and the Equator; 1=NE, 2=SE, 3=SW, 4=NW
4. area: in thousands of square km
5. population: in round millions
6. language: 1=English, 2=Spanish, 3=French, 4=German, 5=Slavic, 6=Other Indo-European, 7=Chinese, 8=Arabic, 9=Japanese/Turkish/Finnish/Magyar, 10=Others
7. religion: 0=Catholic, 1=Other Christian, 2=Muslim, 3=Buddhist, 4=Hindu, 5=Ethnic, 6=Marxist, 7=Others
8. bars: Number of vertical bars in the flag
9. stripes: Number of horizontal stripes in the flag
10. colours: Number of different colours in the flag
11. red: 0 if red absent, 1 if red present in the flag
12. green: same for green
13. blue: same for blue
14. gold: same for gold (also yellow)
15. white: same for white
16. black: same for black
17. orange: same for orange (also brown)
18. mainhue: predominant colour in the flag (tie-breaks decided by taking the topmost hue, if that fails then the most central hue, and if that fails the leftmost hue)
19. circles: Number of circles in the flag
20. crosses: Number of (upright) crosses
21. saltires: Number of diagonal crosses
22. quarters: Number of quartered sections
23. sunstars: Number of sun or star symbols
24. crescent: 1 if a crescent moon symbol present, else 0
25. triangle: 1 if any triangles present, 0 otherwise
26. icon: 1 if an inanimate image present (e.g., a boat), otherwise 0
27. animate: 1 if an animate image (e.g., an eagle, a tree, a human hand) present, 0 otherwise
28. text: 1 if any letters or writing on the flag (e.g., a motto or slogan), 0 otherwise
29. topleft: colour in the top-left corner (moving right to decide tie-breaks)
30. botright: Colour in the bottom-left corner (moving left to decide tie-breaks)
"""
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from matplotlib import cm
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from table import Table

DATA_FILE = 'flag.data'
COLUMNS = ['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours',
           'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'circles', 'crosses', 'saltires',
           'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright']
NUMERIC_COLUMNS = ['landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours',
                   'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'circles', 'crosses', 'saltires',
                   'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text']


def main():
    # ----------------------------
    t = Table(DATA_FILE, COLUMNS)

    # ----------------------------
    print(t.value.describe())

    # ----------------------------
    print(t.value.head())
    print(t.value.shape)

    # ----------------------------
    for column in t.value.columns:
        print("Unique values in column {} are: {}".format(column, t.value[column].unique()))

    # ----------------------------
    for column in t.value.columns:
        if column not in NUMERIC_COLUMNS:
            t.value[column].value_counts().plot(kind='bar')
            plt.title("Histogram of {}".format(column))
            plt.show()

    for column in NUMERIC_COLUMNS:
        t.value.drop([column], axis=1).hist(bins=30, figsize=(9, 9))
        plt.suptitle("Histogram of {}".format(column))
        plt.show()

    # ----------------------------
    x = t.value[NUMERIC_COLUMNS]
    x = x.sample(n=5, axis=1)
    y = t.value['name']
    colours = np.arange(len(y))
    cmap = matplotlib.colormaps["gnuplot"]
    scatter = scatter_matrix(x, c=colours, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
    plt.suptitle('Scatter-matrix for each input variable')
    plt.show()

    # ----------------------------
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # ----------------------------
    tree = DecisionTreeClassifier().fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))

    # ----------------------------
    knn = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(knn.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(knn.score(x_test, y_test)))

    # ----------------------------
    lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(lda.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(lda.score(x_test, y_test)))

    # ----------------------------
    gnb = GaussianNB().fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(gnb.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(gnb.score(x_test, y_test)))

    # ----------------------------
    svm = SVC().fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(svm.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(svm.score(x_test, y_test)))


if __name__ == '__main__':
    main()
