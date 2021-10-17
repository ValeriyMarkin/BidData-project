from Table import Table
from schemas import nation_schema
from groupby import groupby


if __name__ == '__main__':
    nation = Table(nation_schema, 25, "NATION", storage='row')
    nation.load_csv('TPCH-data/SF-0.5/nation.csv')
    r = groupby(nation, "N_REGIONKEY")
    print(r[0].data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
