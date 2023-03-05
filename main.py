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

from table import Table

DATA_FILE = 'flag.data'
COLUMNS = ['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours',
           'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'circles', 'crosses', 'saltires',
           'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright']
NUMERIC_COLUMNS = ['landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours',
                   'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'circles', 'crosses', 'saltires',
                   'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text']


def main():
    t = Table(DATA_FILE, COLUMNS)

    print(t.value.head())

    # ----------------------------
    # print(table.head()) # to see the data
    # print(table.shape) # to see the shape of the data (rows, columns)
    # print(table.columns)  # to see the columns of the data
    # ----------------------------

    # ----------------------------
    # for non numeric columns use sns.countplot(table['name'], label="Count"), plt.show()
    # for numeric columns use table.drop(['name'], axis=1).hist(bins=30, figsize=(9, 9)), plt.show()
    # ----------------------------

    # ----------------------------
    # x = table[NUMERIC_COLUMNS]
    # y = table['name']
    # colours = np.arange(len(y))
    # scatter = scatter_matrix(x, c=colours, marker='o', s=40,
    #                        hist_kwds={'bins': 15}, cmap='gnuplot')
    # plt.savefig('table_scatter_matrix')
    # ----------------------------


if __name__ == '__main__':
    main()
