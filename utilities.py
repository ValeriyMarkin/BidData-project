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


