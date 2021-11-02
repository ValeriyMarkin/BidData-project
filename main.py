from Table import Table
from schemas import nation_schema,orders_schema
from groupby import groupby
from selection import simpleSelection,diskSelection, multiThreadSelection, binarySearchSelection
import timeit

if __name__ == '__main__':
    nation = Table(nation_schema, 25, "NATION", storage="col")
    nation.load_csv('TPCH-data/SF-0.5/nation.csv')
    
    x = simpleSelection(nation, "N_REGIONKEY", "!=", 1)
    #print(x.data)
    x = binarySearchSelection(x, "N_REGIONKEY", "!=", 1)
    #print(x.data)
    
    orders = Table(orders_schema,75000, "ORDERS", storage="col")
    orders.load_csv('TPCH-data/SF-0.5/orders.csv')
    

    start = timeit.default_timer()
    x = simpleSelection(orders, "O_ORDERSTATUS", "==", "F")
    stop = timeit.default_timer()
    print('Time for simpleSelection: ', stop - start) 
    
    start = timeit.default_timer()
    x = multiThreadSelection(orders, "O_ORDERSTATUS", "==", "F", 3)
    stop = timeit.default_timer()
    print('Time for multiThreadSelection: ', stop - start)
    
    start = timeit.default_timer()
    x = diskSelection(orders, "O_ORDERSTATUS", "==", "F")
    stop = timeit.default_timer()
    print('Time for diskSelection: ', stop - start) 
    
    start = timeit.default_timer()
    x = binarySearchSelection(orders, "O_ORDERSTATUS", "==", "F")
    stop = timeit.default_timer()
    print('Time for binarySearchSelection ordering: ', stop - start) 
    
    start = timeit.default_timer()
    x = binarySearchSelection(orders, "O_ORDERSTATUS", "==", "F", True)
    stop = timeit.default_timer()
    print('Time for binarySearchSelection: ', stop - start)


