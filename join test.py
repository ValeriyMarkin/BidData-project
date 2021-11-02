from Table import Table 
import schemas   
import numpy as np 

def recup_index_lig (tab,Nom):
    for index,element in enumerate(tab.col_names):
        if element == Nom :
            
            #print (index)
            return index

def recup_index_lig2 (tab,Nom):

    for index,element in enumerate(tab.col_names):
        if element == Nom :

            #print (liste)
            return index

def recup_index_col (tab,lig,test):
    liste = []
    if (tab.storage == 'row'):
        for i in range(tab.data.shape[0]):
            if(tab.data[i][lig] == test):
                liste.append(i)
    else :   
        for i in range(len(tab.data[lig][:])):
            if(tab.data[lig][i] == test):
                liste.append(i)
    #print(liste)
    return liste

def recup_index_col2 (tab,lig,test):
    liste = []
    if (tab.storage == 'row'):
        for i in range(tab.data.shape[0]):
            if(tab.data[i][lig] == test):
                liste.append(i)
    else : 
        for i in range(len(tab.data[lig])):
            if(tab.data[lig][i] == test):
                liste.append(i)
        #print(liste)
    return liste
 

def join(tab1,key1,tab2,key2):
    test_scheme = {**tab1.schema , **tab2.schema} 
    del test_scheme[key2]
    tab = []
    indexlig = recup_index_lig(tab1, key1)
    indexlig2 = recup_index_lig2(tab2, key2)
    
    if (tab1.storage == 'row'):
        test = list(set(tab1.data[:,indexlig]))
        for key in test : 
            index = recup_index_col(tab1, indexlig,key)
            index2 = recup_index_col2(tab2, indexlig2,key)
            for i in range (len(index)):
                for j in range (len(index2)):
                    temp = []
                    #temp.append(key)
                    for k in range(tab1.n_cols):
                        temp.append(tab1.data[index[i],k])
                            
                    for k2 in range(tab2.n_cols):
                        if (k2 != indexlig2):
                            temp.append(tab2.data[index2[j],k2])                  
                    tab.append(temp) 
        tab = np.array(tab)
        joint=Table(test_scheme,tab.shape[0],tab1.name+" "+tab2.name,tab1.storage)
        joint.fill_data(tab)
    
    else :
       
        test = list(set(tab1.data[indexlig][:]))     
        for key in test : 
            index = recup_index_col(tab1, indexlig,key)
            index2 = recup_index_col2(tab2, indexlig2,key)
            for i in range (len(index)):
                for j in range (len(index2)):
                    temp = []
                    #temp.append(key)
                    for k in range(tab1.n_cols):
                        temp.append(tab1.data[k][index[i]])
                            
                    for k2 in range(tab2.n_cols):
                        if (k2 != indexlig2):
                            temp.append(tab2.data[k2][index2[j]])
            
                    tab.append(temp)
        joint=Table(test_scheme,len(tab),tab1.name+" "+tab2.name,tab1.storage)
        joint.fill_data(tab)
    return joint

#%%

nation = Table(schemas.nation_schema, 25, "NATION",'row')
nation.load_csv('TPCH-data/SF-0.5/nation.csv')

region = Table(schemas.region_schema, 5, "REGION",'row')
region.load_csv('TPCH-data/SF-0.5/region.csv')

mot = "N_REGIONKEY"
mot2 = "R_REGIONKEY"  

joined_tab = join(nation,mot,region,mot2)

