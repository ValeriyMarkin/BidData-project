import numpy as np
import pandas as pd
from utilities import *


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
        
       	self.col_index = {}
        for i in range(len(self.col_names)):
            self.col_index[self.col_names[i]] = i
        
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


def create_slice(table, idx):
    result = Table(table.schema, len(idx), table.name, table.storage)
    if result.storage == "column":
        result.data = [col[idx] for col in table.data]
    else:
        result.data = table.data[idx]
    return result


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


