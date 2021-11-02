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

        self.view_number = 0
        self.col_index = {}
        for i, col in enumerate(self.col_names):
            self.col_index[col] = i
            
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


    def projection(self, columns):
            """
            projection
            Projects the data of the table keeping only the columns given as arguments
            and returns a new table without duplicate row

            :param columns: name of the columns selected to perform the projection.
            :type b: List of string

            :return: The view of the table projected on the selected columns.
            :rtype: Table

            """
            # Construction of the name of the projected view
            self.view_number += 1
            view_name = "{}_View_{}".format(self.name, self.view_number)

            # Construction of the schema of the projected view
            projected_schema = {}
            for col in columns:
                projected_schema[col] = self.schema[col]

            # Extraction of the data corresponding to the selected columns
            if self.storage == "row":
                selected_data = np.empty(self.n_rows, dtype=object)
                sel_col = []
                for col in columns:
                    sel_col.append(self.col_index[col])

                for i in range(self.n_rows):
                    selected_data[i] = self.data[i][sel_col]
            else:
                selected_data = []
                for col in columns:
                    selected_data.append(self.data[self.col_index[col]][:])

            # Deletion of the duplicate rows
            if self.storage == "row":
                # We transpose the data to find the duplicate rows ('np.unique' or 'set' doesn't work with data of type object)
                List_index_to_delete = get_index_to_delete(from_rows_to_columns(selected_data, projected_schema))
            else:
                List_index_to_delete = get_index_to_delete(selected_data)

            Nb_rows = self.n_rows - len(List_index_to_delete)

            # View construction
            projected_view = Table(projected_schema, Nb_rows, view_name, storage=self.storage)

            # Updating the data of the projected view
            if self.storage == "row":
                if len(List_index_to_delete) != 0:
                    k = -1
                    for i in range(self.n_rows):
                        if i not in List_index_to_delete:
                            k += 1
                            projected_view.data[k] = selected_data[i]
                else:
                    projected_view.data = selected_data
            else:
                if len(List_index_to_delete) != 0:
                    k = -1
                    for i in range(self.n_rows):
                        if i not in List_index_to_delete:
                            k += 1
                            for j, col in enumerate(columns):
                                projected_view.data[j][k] = self.data[self.col_index[col]][i]
                else:
                    for j, col in enumerate(columns):
                        projected_view.data[j][:] = self.data[self.col_index[col]][:]

            return projected_view


def create_slice(table, idx):
    result = Table(table.schema, len(idx), table.name, table.storage)
    if result.storage == "col":
        result.data = [col[idx] for col in table.data]
    else:
        result.data = table.data[idx]
    return result


def index_duplicate(a, index_to_explore=None):
    """
    index_duplicate
    Find the indexes of the duplicate values in a Numpy ndarray

    :param a: the Numpy array
    :type a: numpy.ndarray

    :param index_to_explore: list of the indexes from the Numpy array that we want to explore.
                             Working on a subset of the array improves performance.
                             (default value=None: the whole array is explored)
    :type index_to_explore: list of int

    :return: A dictionary (dict_duplicate) which associates with each duplicated value
             its index in the Numpy array
    :rtype: dict

    example:
    >>> index_duplicate(np.array(['B', 'C', 'B', 'D', 'D', 'D', 'D', 'A']))
    {'B': (array([0, 2], dtype=int64),), 'D': (array([3, 4, 5, 6], dtype=int64),)}

    >>> index_duplicate(np.array(['B', 'C', 'B', 'D', 'D', 'D', 'D', 'A']), index_to_explore=[1, 2, 3, 4])
    {'D': array([3, 4], dtype=int64)}

    """
    if index_to_explore is None:
        s = np.sort(a, axis=None)
    else:
        s = np.sort(a[index_to_explore], axis=None)

    list_duplicate_values = s[:-1][s[1:] == s[:-1]]

    dict_duplicate = {}
    if index_to_explore is None:
        for i in range(len(list_duplicate_values)):
            dict_duplicate[list_duplicate_values[i]] = np.where(a == list_duplicate_values[i])
    else:
        for i in range(len(list_duplicate_values)):
            dict_duplicate[list_duplicate_values[i]] = np.intersect1d(np.where(a == list_duplicate_values[i]),
                                                                      index_to_explore)

    return dict_duplicate


#   ===================================================

def get_index_to_delete(b):
    """
    get_index_to_delete
    In a projection operation, only one instance of the duplicated rows is kept.
    This function gives the indices of the rows to be deleted

    :param b: List of Numpy arrays. Each array contains the values of a column of the table.
    :type b: List of numpy.ndarray

    :return: the list of the index of the duplicate values to be deleted. This list may be empty.
    :rtype: list of int

    example:
    >>> get_index_to_delete([np.array([1, 2, 3]), np.array(['A', 'B', 'A'])])
    []

    >>> get_index_to_delete([np.array([1, 2, 1, 1]), np.array(['A', 'B', 'A', 'A'])])
    [0, 2]

    """

    index_to_delete = []
    # First column
    dict_duplicate_2 = index_duplicate(b[0])
    if dict_duplicate_2 == {}: return index_to_delete

    # Next columns
    for i in range(1, len(b)):
        dict_duplicate_1 = dict_duplicate_2
        dict_duplicate_2 = {}
        for key in dict_duplicate_1:
            dict_duplicate_2.update(index_duplicate(b[i], dict_duplicate_1[key]))
        if dict_duplicate_2 == {}: return index_to_delete

    for key in dict_duplicate_2:
        # We keep the last element of the duplicate list associated with each key
        list_duplicate = dict_duplicate_2[key][0:len(dict_duplicate_2[key]) - 1]
        for index in list_duplicate:
            index_to_delete.append(index)

    index_to_delete.sort()

    return index_to_delete


#   ===================================================

def from_rows_to_columns(r, schema):
    """
    from_rows_to_columns
    Changes from a representation of data by rows to a representation of data by columns

    :param r: data stored per rows.
    :type r: numpy.ndarray

    :param schema: name and dtype of each column of the Table.
    :type schema: dict

    :return: data stored per columns.
    :rtype: numpy.ndarray

    """
    n_rows = len(r)
    n_columns = len(r[0])

    c = [np.empty(n_rows, dtype=column[1]) for column in schema.items()]

    for i in range(n_columns):
        for j in range(n_rows):
            c[i][j] = r[j][i]

    return c
