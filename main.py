from Table import Table
from schemas import nation_schema

if __name__ == '__main__':

    nation = Table(nation_schema, 25, "NATION", storage='column')
    nation.load_csv('TPCH-data/SF-0.5/nation.csv')
    print(nation.data)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
