from Table import Table 
from schemas import *
     

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


def join(tab1,key1,tab2,key2):
    tab = []
    indexlig = recup_index_lig(tab1, key1)
    indexlig2 = recup_index_lig2(tab2, key2)
    test = list(set(tab1.data[indexlig][:]))
    
    #enlever les doublons
    
    for key in test : 
        index = recup_index_col(tab1, indexlig,key)
        index2 = recup_index_col2(tab2, indexlig2,key)
        for i in range (len(index)):
            for j in range (len(index2)):
                temp = []
                temp.append(key)
                for k in range(len(tab1.col_names)):
                    if (k != indexlig):
                        temp.append(tab1.data[k][index[i]])
                        
                for k2 in range(len(tab2.col_names)):
                    if (k2 != indexlig2):
                        temp.append(tab2.data[k2][index2[j]])
        
                tab.append(temp)
    #print([h[r[0]] for r in table2])   
    return tab


nation = Table(nation_schema, 25, "NATION",'column')
nation.load_csv('TPCH-data/SF-0.5/nation.csv')

region = Table(region_schema, 5, "REGION",'column')
region.load_csv('TPCH-data/SF-0.5/region.csv')

mot = "N_REGIONKEY"
mot2 = "R_REGIONKEY"  

joined_tab = join(nation,mot,region,mot2)