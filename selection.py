from Table import Table, SparkTable
from utilities import mergeSort

import operator
import numpy as np
import threading
from math import ceil


def simpleSelection(table, attr, op, value,thread_res = None):

    """
    This function implements a simple version of the select operator on the Table class. Ideal when the data is not ordered.
    
    :param table: an object of class Table, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be selected
    :param op: str, operator that will be used to make the selection. Can be: “<”,”>”,”<=”,”>=”,”==”,”!=”
    :param value: str, value to which our attribute will be compared
    :param thread_res: list, in case func is called by multi-Thread process, is data structure o save outputs across threads
    
    :return: table, return another object of the class Table, which satisfies the selection condition

    """
    
    # Defining possible operations that will be passed to the function as strings
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq,
       '!=': operator.ne}
    
    selected=[]
    # Determining the index of the attribute of interest
    col = table.col_names.index(attr)
    # If row storage, go row by row selecting entries that satisfy selection
    if table.storage == "row":
        for row in range(table.n_rows):
            if(ops[op](table.data[row,col],value)):
                selected.append(table.data[row,:])
                if thread_res is not None:
                   thread_res.append(table.data[row,:])
        result = Table(table.schema, len(selected), "SELECTION_RESULT", storage=table.storage)
        result.data = np.array(selected)
    
    # If col storage, get indices from the column corresponding to the attribute, 
    # then use indices in the other columns.
    else:
        index=[]
        for i in range(table.n_rows):
            if(ops[op](table.data[col][i],value)):
                index.append(i)
        for column in range(table.n_cols):
            selected.append(table.data[column][index])
            if thread_res is not None:
                   thread_res[column].extend(table.data[column][index])
        result = Table(table.schema, len(index), "SELECTION_RESULT", storage=table.storage)
        result.data = selected
        
    return result


def binarySearchSelection(table, attr, op, value, is_sorted = False):

    """
    This function implements version of the select operator using Binary search on the Table class. Ideal when the data is ordered.
    
    :param table: an object of class Table, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be selected
    :param op: str, operator that will be used to make the selection. Can be: “<”,”>”,”<=”,”>=”,”==”,”!=”
    :param value: str, value to which our attribute will be compared
    :param is_sorted: bool, True if table is already sorted and false if it is not (thus need previous sorting)
    
    :return: table, return another object of the class Table, which satisfies the selection condition

    """
    
    # Sort table if table is not yet sorted
    if not is_sorted:
        mergeSort(table, attr)
        
    col = table.col_names.index(attr)
    
    # Get data related to column of attribute
    if table.storage == "row":
        data = table.data[:,col]
    else:
        data = table.data[col]
        
    # In case the value is not in the table, we find closest higher and lower value
    up_closest_idx = -1
    down_closest_idx = -1
        
    # We will have to find the starting and ending index, as there can be duplicates
    # Finding starting index.
    low = 0
    high = table.n_rows - 1
    start_idx = -1
    
    while(low <= high):
        # Slide array in half
        mid =  (high - low)//2 + low
        # If the middle value is higher than the target value, we ignore right half
        if(data[mid] > value ):
            high = mid - 1
            up_closest_idx = mid
        # If the middle value is the same, we need to check if we are in the start
        elif(data[mid] == value):
            start_idx = mid
            high = mid - 1
        # If the middle value is lower than the target value, we ignore left half
        else:
            low = mid + 1
            down_closest_idx = mid
            
    if(start_idx !=-1):
        # Finding ending index.        
        low = 0
        high = table.n_rows - 1        
        end_idx = -1
        while(low <= high):
            # Slide array in half
            mid =(high - low)//2 + low
            # If the middle value is higher than the target value, we ignore right half
            if(data[mid] > value ):
                high = mid - 1
            # If the middle value is the same, we need to check if we are in the end
            elif(data[mid] == value):
                end_idx = mid
                low = mid + 1
            # If the middle value is lower than the target value, we ignore left half
            else:
                low = mid + 1
    
    # If value was in our table
    if(start_idx != -1 and end_idx != -1):
        # Defining slicing for each possible operation     
        if (op =='>'):
            if table.storage == "row":
                res = table.data[:, end_idx+1:]
            else:
                res = [col[end_idx+1:] for col in table.data]
        elif (op == '<'):
            if table.storage == "row":
                res = table.data[:,:start_idx]
            else:
                res = [col[:start_idx] for col in table.data]
        elif (op == '>='):
            if table.storage == "row":
                res = table.data[:, start_idx:]
            else:
                res = [col[start_idx:] for col in table.data]
        elif (op == '<='):
            if table.storage == "row":
                res = table.data[:, : end_idx+1]
            else:
                res = [col[:end_idx+1] for col in table.data]
        elif (op == '=='):
            if table.storage == "row":
                res = table.data[:, start_idx: end_idx+1]
            else:
                res = [col[start_idx:end_idx+1] for col in table.data]
        elif (op == '!='):
            if table.storage == "row":
                res = np.concatenate((table.data[:,:start_idx], table.data[:, end_idx+1:]), axis=0)
            else:
                res = [np.concatenate((col[:start_idx], col[end_idx+1:]), axis=0) for col in table.data]

    
    # If value was not in  table
    else:
        # Defining slicing for each possible operation     
        if (op =='>'):
            if table.storage == "row":
                res = table.data[:, up_closest_idx:]
            else:
                res = [col[up_closest_idx:] for col in table.data]
        elif (op == '<'):
            if table.storage == "row":
                res = table.data[:,:down_closest_idx+1]
            else:
                res = [col[:down_closest_idx+1] for col in table.data]
        elif (op == '>='):
            if table.storage == "row":
                res = table.data[:, up_closest_idx:]
            else:
                res = [col[up_closest_idx:] for col in table.data]
        elif (op == '<='):
            if table.storage == "row":
                res = table.data[:, : down_closest_idx+1]
            else:
                res = [col[:down_closest_idx+1] for col in table.data]
        elif (op == '=='):
            res = np.array([[]])
        elif (op == '!='):
            res = table.data

    # Turning results into a table object
    if table.storage == "row":
        result = Table(table.schema, res.shape[0], "SELECTION_RESULT", storage=table.storage)
    else:
        result = Table(table.schema, res[0].size, "SELECTION_RESULT", storage=table.storage)
    result.data = res
    
    return result


def diskSelection(table, attr, op, value):

    """
    This function implements a simple version of the select operator on the Table class stored on the disk.
    
    :param table: an object of class Table stored in the disk, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be selected
    :param op: str, operator that will be used to make the selection. Can be: “<”,”>”,”<=”,”>=”,”==”,”!=”
    :param value: str, value to which our attribute will be compared
    
    :return: table, return another object of the class Table saved in another file, which satisfies the selection condition


    """
    
    # Defining possible operations that will be passed to the function as strings
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq,
       '!=': operator.ne}
    
    # Determining the index of the attribute of interest
    col = table.col_names.index(attr)
    n_rows = 0
    # Open new file to which the data will be written 
    res_file = open("TPCH-data/SF-0.5/result_selection.csv", "w")
    # Going through the file line by line. If line respect the selection, add it to new file 
    with open(table.filename) as file:
        for line in file:
            data = line.split('|')
            # Need to convert string to expected type in schema  
            if(ops[op](np.array([data[col]]).astype(table.dtypes[col])[0],value)):
                res_file.write(line)
                n_rows+=1
    res_file.close()
    # Create table object with correct path for new Table
    result = Table(table.schema, n_rows, "SELECTION_RESULT", storage='disk', filename = "TPCH-data/SF-0.5/result_selection.csv")
    
        
    return result


def multiThreadSelection(table, attr, op, value, num_threads):

    """
    This function implements a simple version of the select operator on the Table class using multiThread
    
    :param table: an object of class Table, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be selected
    :param op: str, operator that will be used to make the selection. Can be: “<”,”>”,”<=”,”>=”,”==”,”!=”
    :param value: str, value to which our attribute will be compared
    :param num_threads: int, number of threads used in the process
    
    :return: table, return another object of the class Table, which satisfies the selection condition

    """
    
    # Initializing array of threads and the slices of tables
    threads_list = np.empty(num_threads, dtype=object)
    table_slices = np.empty(num_threads, dtype=object)
    start = 0
    size = ceil(float(table.n_rows)/float(num_threads))
    
    # Initializing results list that will be shared between threads
    r = []
    if table.storage == 'col':
        for column in range(table.n_cols):
            r.append([])
     
    # Initializing threads with simple selection function and arguments
    for i in range(num_threads):
        # Slicing tables
        table_slices[i] = Table(table.schema, size, "TABLE_SLICE", storage=table.storage)
        if table.storage == "row":
            table_slices[i].data = table.data[start:start+size,:]
        else: 
            table_slices[i].data = [col[start:start+size] for col in table.data]
        
        start += size
        if(start+size > table.n_rows ):
            size = table.n_rows - start
        
        # Create threads as follows
        try:
           threads_list[i] = threading.Thread( target=simpleSelection, args=(table_slices[i], attr, op, value, r) )
        except:
           print("Error: unable to start thread")
           
    # Starting Threads       
    for t in threads_list:
        t.start()
    # Waiting for all threads to finish   
    for t in threads_list:
        t.join()
        
    # Formatting results into right type
    if table.storage == "row":
        result = Table(table.schema, len(r), "SELECTION_RESULT", storage=table.storage)
        result.data = np.array(r)
    else:
        result = Table(table.schema, len(r[0]), "SELECTION_RESULT", storage=table.storage)
        result.data = [np.array(col) for col in r]
        
    return result


def SparkSelection(table, attr, op, value, sc):

    """
    This function implements a simple version of the select operator in Spark.
    
    :param table: an object of class SparkTable, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be selected
    :param op: str, operator that will be used to make the selection. Can be: “<”,”>”,”<=”,”>=”,”==”,”!=”
    :param value: str, value to which our attribute will be compared
    
    :return: table, return another object of the class SparkTable, which satisfies the selection condition

    """
    
    # Defining possible operations that will be passed to the function as strings
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq,
       '!=': operator.ne}
    
    selected=[]
    # Determining the index of the attribute of interest
    col = table.col_names.index(attr)
    # Collecting data from spark 
    for row in np.array(table.rdd.collect()):
        if(ops[op](row[col],value)):
                selected.append(row)
                
    # Creating new table and new SparkTable with result           
    result_table = Table(table.schema, len(selected), "SELECTION_RESULT", storage="row")
    result_table.data = np.array(selected)
    
    result_spark = SparkTable(result_table,sc)
        
    return result_spark