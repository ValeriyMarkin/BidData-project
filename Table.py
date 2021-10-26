import numpy as np
import pandas as pd


class Table:
    def __init__(self, schema, n_rows, name, storage="row", filename=None):
        self.name = name
        self.n_cols = len(schema)
        self.storage = storage
        self.schema = schema
        self.col_names = [p[0] for p in schema.items()]
        self.dtypes = [p[1] for p in schema.items()]
        self.n_rows = n_rows
        self.filename = filename
        if self.storage == "row":
            self.data = np.empty(self.n_rows, dtype=object)
        elif self.storage == "col":
            self.data = [np.empty(self.n_rows, dtype=column[1]) for column in self.schema.items()]
        else:
            self.data = None

    def load_csv(self, filename):
        df = pd.read_csv(filename, delimiter='|', header=None).iloc[:, :-1]
        self.n_rows = len(df)
        self.filename = filename
        if self.storage == "row":
            self.data = df.values
        else:
            for i in range(self.n_cols):
                self.data[i][:] = df.iloc[:, i].values[:].astype(self.dtypes[i])


def create_slice(table, idx):
    result = Table(table.schema, len(idx), table.name, table.storage)
    if result.storage == "column":
        result.data = [col[idx] for col in table.data]
    else:
        result.data = table.data[idx]
    return result
