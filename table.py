import pandas as pd


class Table:
    def __init__(self, filename, columns):
        self.value = pd.read_table(filename, sep=',', header=None, names=columns)


