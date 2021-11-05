from Table import Table
import numpy as np
from schemas import *

def recup_index_lig(tab, Nom):
    for index, element in enumerate(tab.col_names):
        if element == Nom:
            # print (index)
            return index


def recup_index_col(tab, lig, test):
    liste = []
    if (tab.storage == 'row'):
        for i in range(tab.data.shape[0]):
            if (tab.data[i][lig] == test):
                liste.append(i)
    else:
        for i in range(len(tab.data[lig][:])):
            if (tab.data[lig][i] == test):
                liste.append(i)
    # print(liste)
    return liste


def join_multithread(tab1, key1, tab2, key2, num_threads):
    test_scheme = {**tab1.schema, **tab2.schema}
    del test_scheme[key2]
    
    n_rows = max(tab1.n_rows, tab2.n_rows)
    
    indexlig = recup_index_lig(tab1, key1)
    indexlig2 = recup_index_lig(tab2, key2)
    joint = Table(test_scheme, n_rows, tab1.name + " " + tab2.name, tab1.storage)
    if tab1.storage == "row":
        test = list(set(tab1.data[:, indexlig]))
        joint.data=np.empty((0,tab1.n_cols+tab2.n_cols-1))
    else:
        test = list(set(tab1.data[indexlig][:]))
        joint.data=[]
        for i in range(tab1.n_cols+tab2.n_cols-1):
            joint.data.append([])
    
    threads_list = np.empty(num_threads, dtype=object)
    threads_t = np.array_split(np.array(test), num_threads)
    idx = np.array_split(np.array(range(0,n_rows)), num_threads)
    
    
    def single_thread_join(tt,a):
        tab = []
        if (tab1.storage == 'row'):
            temp= []
            res1=[]
            res2 = []
    
            for t,key in enumerate(tt):
                index = recup_index_col(tab1, indexlig, key)
                index2 = recup_index_col(tab2, indexlig2, key)
                for i in range(len(index)):
                    
                    for j in range(len(index2)):
                        temp = []
                        res1=[]
                        res2 = []
                        # temp.append(key)
                        for k in range(tab1.n_cols):
                            res1.append(tab1.data[index[i], k])

                        for k2 in range(tab2.n_cols):
                            if (k2 != indexlig2):
                                res2.append(tab2.data[index2[j], k2])
                        if res1!=[] and res2!=[]:
                            temp=res1+res2
                            joint.data = np.append(joint.data, [np.array(temp, dtype=object)],axis=0)
        else:

            
            for key in tt:
                temp= []
                res1=[]
                res2 = []
                index = recup_index_col(tab1, indexlig, key)
                index2 = recup_index_col(tab2, indexlig2, key)
                for i in range(len(index)):
                    for j in range(len(index2)):
                        temp = []
                        res1=[]
                        res2 = []
                        # temp.append(key)
                        for k in range(tab1.n_cols):
                            res1.append(tab1.data[k][index[i]])

                        for k2 in range(tab2.n_cols):
                            if (k2 != indexlig2):
                                res2.append(tab2.data[k2][index2[j]])

                    if res1!=[] and res2!=[]:
                            temp=res1+res2
                            for w,val in enumerate(temp):
                                joint.data[w].append(val)
                        
    
    for i in range(num_threads):
        threads_list[i] = threading.Thread(target=single_thread_join, args=(threads_t[i], 1))

    # Starting Threads
    for t in threads_list:
        t.start()
    # Waiting for all threads to finish
    for t in threads_list:
        t.join()

def join(tab1, key1, tab2, key2):
    test_scheme = {**tab1.schema, **tab2.schema}
    del test_scheme[key2]
    tab = []
    indexlig = recup_index_lig(tab1, key1)
    indexlig2 = recup_index_lig(tab2, key2)

    if (tab1.storage == 'row'):
        test = list(set(tab1.data[:, indexlig]))
        for key in test:
            index = recup_index_col(tab1, indexlig, key)
            index2 = recup_index_col(tab2, indexlig2, key)
            for i in range(len(index)):
                for j in range(len(index2)):
                    temp = []
                    # temp.append(key)
                    for k in range(tab1.n_cols):
                        temp.append(tab1.data[index[i], k])

                    for k2 in range(tab2.n_cols):
                        if (k2 != indexlig2):
                            temp.append(tab2.data[index2[j], k2])
                    tab.append(temp)
        tab = np.array(tab)
        joint = Table(test_scheme, tab.shape[0], tab1.name + " " + tab2.name, tab1.storage)
        joint.fill_data(tab)

    else:

        test = list(set(tab1.data[indexlig][:]))
        for key in test:
            index = recup_index_col(tab1, indexlig, key)
            index2 = recup_index_col(tab2, indexlig2, key)
            for i in range(len(index)):
                for j in range(len(index2)):
                    temp = []
                    # temp.append(key)
                    for k in range(tab1.n_cols):
                        temp.append(tab1.data[k][index[i]])

                    for k2 in range(tab2.n_cols):
                        if (k2 != indexlig2):
                            temp.append(tab2.data[k2][index2[j]])

                    tab.append(temp)
        joint = Table(test_scheme, len(tab), tab1.name + " " + tab2.name, tab1.storage)
        joint.fill_data(tab)
    return joint
