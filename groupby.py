from Table import Table,SparkTable, create_slice
import numpy as np
import threading

def groupby(table, by):
    """
    This function implements the group by operator on the Table class. The aggregation function could be applied to the
    resulted tables.

    :param table: an object of class Table, on which the operation is performed
    :param by: str, name of the column, based on which the values in table are grouped
    :return: dict, keys are unique values of the column "by", values are new Table class objects, one for every unique
    value in the target column
    """
    idx = table.col_names.index(by)
    if table.storage == "col":
        target_column = table.data[idx]
        index_dict = {}
        for i, el in enumerate(target_column):
            if el in index_dict.keys():
                index_dict[el].append(i)
            else:
                index_dict[el] = [i]
        grouped_t = {}
        for val in index_dict:
            grouped_t[val] = create_slice(table, index_dict[val])
    else:
        index_dict = {}
        for i, row in enumerate(table.data):
            if row[idx] in index_dict.keys():
                index_dict[row[idx]].append(i)
            else:
                index_dict[row[idx]] = [i]
        grouped_t = {}
        for val in index_dict:
            grouped_t[val] = create_slice(table, index_dict[val])
    return grouped_t


def aggregate(grouped_t, group_key_name, col_to_aggr, func):
    result_schema = {
        group_key_name: grouped_t[list(grouped_t.keys())[0]].schema[group_key_name],
        col_to_aggr: float,
    }
    storage = grouped_t[list(grouped_t.keys())[0]].storage
    result = Table(result_schema, len(grouped_t), "AGGR_" + col_to_aggr, storage=storage)
    if storage == "row":
        for i in range(len(grouped_t)):
            res1 = list(grouped_t.keys())[i]
            res2 = func(grouped_t[list(grouped_t.keys())[i]], col_to_aggr)
            result.data[i] = np.array([res1, res2], dtype=object)
    else:
        for i in range(len(grouped_t)):
            result.data[0][i] = list(grouped_t.keys())[i]
            result.data[1][i] = func(grouped_t[list(grouped_t.keys())[i]], col_to_aggr)
    return result


def groupby_spark(table, by, aggregate, sc):
    
    by_index = table.col_names.index(by)
    aggregate_index = table.col_names.index(aggregate)
    spark_table = SparkTable(table,sc)
    grouped = spark_table.rdd.map( lambda row: (row[by_index], row[aggregate_index]) ) 
    Result = grouped.reduceByKey(lambda a, b :a+b)
    
    Result = np.array(Result)
    
    result_schema = {
        by: table.schema[by],
        aggregate: float,
    }
    storage = table.storage
    result = Table(result_schema, len(Result), "AGGR_" + aggregate, storage=storage)
    result.data = np.array(Result)
    
    result_spark = SparkTable(result,sc)
    
    return result_spark

def multithread_groupby(table, by, num_threads):
    threads_list = np.empty(num_threads, dtype=object)
    idx = table.col_names.index(by)
    if table.storage == "col":
        target_column = table.data[idx]
        index_dict = {}
        for i, el in enumerate(target_column):
            if el in index_dict.keys():
                index_dict[el].append(i)
            else:
                index_dict[el] = [i]

    else:
        index_dict = {}
        for i, row in enumerate(table.data):
            if row[idx] in index_dict.keys():
                index_dict[row[idx]].append(i)
            else:
                index_dict[row[idx]] = [i]

    grouped_t = {}
    print(list(index_dict.keys()))
    thread_keys = np.array_split(list(index_dict.keys()), num_threads)

    def single_thread_groupby(keys, a):
        for k in keys:
            grouped_t[k] = create_slice(table, index_dict[k])

    for i in range(num_threads):
        threads_list[i] = threading.Thread(target=single_thread_groupby, args=(thread_keys[i],1))

    for t in threads_list:
        t.start()
    # Waiting for all threads to finish
    for t in threads_list:
        t.join()
    return grouped_t


def multithread_aggregate(grouped_t, group_key_name, col_to_aggr, func, num_threads):
    result_schema = {
        group_key_name: grouped_t[list(grouped_t.keys())[0]].schema[group_key_name],
        col_to_aggr: float,
    }
    storage = grouped_t[list(grouped_t.keys())[0]].storage
    result = Table(result_schema, len(grouped_t), "AGGR_" + col_to_aggr, storage=storage)
    threads_list = np.empty(num_threads, dtype=object)
    threads_idx = np.array_split(np.arange(len(grouped_t)), num_threads)
    keys_list = np.array(grouped_t.keys())

    def single_thread_aggregate(idx, a):
        if storage == "row":
            for j in idx:
                res1 = list(grouped_t.keys())[j]
                res2 = func(grouped_t[list(grouped_t.keys())[j]], col_to_aggr)
                result.data[j] = np.array([res1, res2], dtype=object)
        else:
            for j in idx:
                result.data[0][j] = list(grouped_t.keys())[j]
                result.data[1][j] = func(grouped_t[list(grouped_t.keys())[j]], col_to_aggr)
        return

    for i in range(num_threads):
        threads_list[i] = threading.Thread(target=single_thread_aggregate, args=(threads_idx[i], 1))

    # Starting Threads
    for t in threads_list:
        t.start()
    # Waiting for all threads to finish
    for t in threads_list:
        t.join()
    return result
