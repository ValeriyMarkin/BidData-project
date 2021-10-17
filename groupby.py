from Table import Table, create_slice


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
    if table.storage == "column":
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
