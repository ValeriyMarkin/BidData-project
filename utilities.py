from Table import Table
import numpy as np

def mergeSort(table, attr):
    
    """
    This function implements the merge sort algorithm, adapted to our Table class. 
    
    :param table: an object of class Table, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be sorted
    
    :return: table, return the same object of the class Table, now sorted. To keep the original data, is recommended to do pass a deep copy of table to the function. 

    """
    
    # Getting index of the attribute we want to order our table accordingly 
    col = table.col_names.index(attr)
    
    if table.n_rows > 1:
        
        # Splitting data into 2, creating new tables, then sorting both halves
        half = table.n_rows//2  
        sort_table_L = Table(table.schema, half, "SORT_L", storage = table.storage)
        sort_table_R = Table(table.schema, table.n_rows-half, "SORT_R", storage = table.storage)
        
        if table.storage == "row":
            sort_table_L.data = np.copy(table.data[:half,:])
            sort_table_R.data = np.copy(table.data[half:,:])
        else:
            sort_table_L.data = [np.copy(table.data[column][:half]) for column in range(table.n_cols)]
            sort_table_R.data = [np.copy(table.data[column][half:]) for column in range(table.n_cols)]
          
        mergeSort(sort_table_L, attr)
        mergeSort(sort_table_R, attr)
        
        # Comparing numbers of left and right halves
        # As both halves are already sorted, we don’t need to compare all values
        i, j, k = 0, 0, 0
        while i < sort_table_L.n_rows and j < sort_table_R.n_rows:
            if table.storage == "row":
                if sort_table_L.data[i,col] <= sort_table_R.data[j,col]:
                    table.data[k,:] = sort_table_L.data[i,:]
                    i += 1
                else:
                    table.data[k,:] = sort_table_R.data[j,:]
                    j += 1
            else:
                if sort_table_L.data[col][i] <= sort_table_R.data[col][j]:
                    for column in range(table.n_cols):
                        table.data[column][k] = sort_table_L.data[column][i]
                    i += 1
                else:
                    for column in range(table.n_cols):
                        table.data[column][k] = sort_table_R.data[column][j]
                    j += 1
            k += 1
            
        # For the remaining values
        while i < sort_table_L.n_rows :
            if table.storage == "row":
                table.data[k,:] = sort_table_L.data[i,:]
            else:
                for column in range(table.n_cols):
                    table.data[column][k] = sort_table_L.data[column][i]
            i += 1
            k += 1

        while j < sort_table_R.n_rows:
            if table.storage == "row":
                table.data[k,:] = sort_table_R.data[j,:]
            else:
                for column in range(table.n_cols):
                    table.data[column][k] = sort_table_R.data[column][j]
            j += 1
            k += 1
            
        
    return table

#   ===================================================

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
    for i in range(1,len(b)):        
        dict_duplicate_1 = dict_duplicate_2        
        dict_duplicate_2 = {}
        for key in dict_duplicate_1:
            dict_duplicate_2.update(index_duplicate(b[i], dict_duplicate_1[key]))    
        if dict_duplicate_2 == {}: return index_to_delete
    
    for key in dict_duplicate_2:
        # We keep the last element of the duplicate list associated with each key
        list_duplicate = dict_duplicate_2[key][0:len(dict_duplicate_2[key])-1]
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
