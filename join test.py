import numpy as np
import pandas as pd


class Table:
    def __init__(self, schema, n_rows, name, storage="row"):
        self.name = name
        self.n_cols = len(schema)
        self.storage = storage
        self.schema = schema
        self.col_names = [p[0] for p in schema.items()]
        self.dtypes = [p[1] for p in schema.items()]
        self.n_rows = n_rows
        if self.storage == "row":
            self.data = np.empty(self.n_rows, dtype=object)
        else:
            self.data = [np.empty(self.n_rows, dtype=column[1]) for column in self.schema.items()]

    def load_csv(self, filename):
        df = pd.read_csv(filename, delimiter='|', header=None).iloc[:, :-1]
        self.n_rows = len(df)
        if self.storage == "row":
            self.data = df.values
        else:
            for i in range(self.n_cols):
                self.data[i][:] = df.iloc[:, i].values[:].astype(self.dtypes[i])

#%%
nation_schema = {
    "N_NATIONKEY": int,
    "N_NAME": '<U25',
    "N_REGIONKEY": int,
    "N_COMMENT": '<U152'}
              
nation = Table(nation_schema, 25, "NATION",'column')
nation.load_csv('TPCH-data/SF-0.5/nation.csv')

#%%
#print(nation.col_names)
"""
for index,name in enumerate(nation.col_names):
    if (nation.col_names == index_1):
"""        

def recup_index_lig (tab,Nom):
    for index,element in enumerate(tab.col_names):
        if element == Nom :
            
            #print (liste)
            return index

def recup_index_lig2 (tab,Nom):

    for index,element in enumerate(tab.col_names):
        if element == Nom :

            #print (liste)
            return index

def recup_index_col (tab,lig,test):
    liste = []
    for i in range(len(tab.data[lig])):
        if(tab.data[lig][i] == test):
            liste.append(i)
    #print(liste)
    return liste

def recup_index_col2 (tab,lig,test):
    liste = []
    for i in range(len(tab.data[lig])):
        if(tab.data[lig][i] == test):
            liste.append(i)
    #print(liste)
    return liste


# hash phase

mot = "N_REGIONKEY"
test = 1 
tab = []
indexlig = recup_index_lig(nation, mot)
indexlig2 = recup_index_lig2(nation, mot)
index = recup_index_col(nation, indexlig,test)
index2 = recup_index_col2(nation, indexlig2,test)
for i in range (len(index)):
    for j in range (len(index2)):
        temp = []
        temp.append(test)
        temp.append(nation.data[0][index[i]])
        temp.append(nation.data[1][index[i]])
        #temp.append(nation.data[2][index[i]])
        temp.append(nation.data[3][index[i]])
        temp.append(nation.data[0][index2[i]])
        temp.append(nation.data[1][index2[i]])
        #temp.append(nation.data[2][index[i]])
        temp.append(nation.data[3][index2[i]])
        tab.append(temp)
#print([h[r[0]] for r in table2])
print(tab)
