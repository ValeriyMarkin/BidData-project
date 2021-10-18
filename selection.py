import operator
import numpy as np
from Table import Table

def simpleSelection(table, attr, op, value):

    """
    This function implements a simple version of the select operator on the Table class. Ideal when the data is not ordered.
    
    :param table: an object of class Table, on which the operation is performed
    :param attr: str, name of the column, based on which the values will be selected
    :param op: str, operator that will be used to make the selection. Can be: “<”,”>”,”<=”,”>=”,”==”,”!=”
    :param value: str, value to which our attribute will be compared
    
    :return: table, return another object of the class Table, which satisfies the selection condition

    """
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '==': operator.eq,
       '!=': operator.ne}
    
    selected=[]
    col = table.col_names.index(attr)
    if table.storage == "row":
        for row in range(table.n_rows):
            if(ops[op](table.data[row,col],value)):
                selected.append(table.data[row,:])
        result = Table(table.schema, len(selected), "SELECTION_RESULT", storage=table.storage)
        result.data = np.array(selected)
        
    else:
        index=[]
        for i in range(table.n_rows):
            if(ops[op](table.data[col][i],value)):
                index.append(i)
        for column in table.data:
            selected.append(column[index])
        result = Table(table.schema, len(index), "SELECTION_RESULT", storage=table.storage)
        result.data = selected
        
    return result