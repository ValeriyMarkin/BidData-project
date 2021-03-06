import datetime

part_schema = {
    "P_PARTKEY": int,
    "P_NAME": '<U55',
    "P_MFGR": '<U25',
    "P_BRAND": '<U10',
    "P_TYPE": '<U25',
    "P_SIZE": int,
    "P_CONTAINER": '<U10',
    "P_RETAILPRICE": float,
    "P_COMMENT": '<U23'
}

supplier_schema = {
    "S_SUPPKEY": int,
    "S_NAME": '<U25',
    "S_ADDRESS": '<U40',
    "S_NATIONKEY": int,
    "S_PHONE": '<U15',
    "S_ACCTBAL ": float,
    "S_COMMENT ": '<U101'
}

partsupp_schema = {
    "PS_PARTKEY": int,
    "PS_SUPPKEY": int,
    "PS_AVAILQTY": int,
    "PS_SUPPLYCOST": float,
    "PS_COMMENT ": '<U199'
}

customer_schema = {
    "C_CUSTKEY": int,
    "C_NAME": '<U25',
    "C_ADDRESS": '<U40',
    "C_NATIONKEY": int,
    "C_PHONE": '<U15',
    "C_ACCTBAL": float,
    "C_MKTSEGMENT": '<U10',
    "C_COMMENT ": '<U117'
}

orders_schema = {
    "O_ORDERKEY": int,
    "O_CUSTKEY": int,
    "O_ORDERSTATUS": '<U1',
    "O_TOTALPRICE": float,
    "O_ORDERDATE": datetime.date,
    "O_ORDERPRIORITY ": '<U15',
    "O_CLERK": '<U15',
    "O_SHIPPRIORITY": int,
    "O_COMMENT": '<U79'
    
}

lineitem_schema = {
    "L_ORDERKEY": int,
    "L_PARTKEY": int,
    "L_SUPPKEY": int,
    "L_LINENUMBER": int,
    "L_QUANTITY": float,
    "L_EXTENDEDPRICE": float,
    "L_DISCOUNT": float,
    "L_TAX": float,
    "L_RETURNFLAG": '<U1',
    "L_LINESTATUS": '<U1',
    "L_SHIPDATE": datetime.date,
    "L_COMMITDATE ": datetime.date,
    "L_RECEIPTDATE": datetime.date,
    "L_SHIPINSTRUCT": '<U25',
    "L_SHIPMODE": '<U10',
    "L_COMMENT ": '<U44'
}

nation_schema = {
    "N_NATIONKEY": int,
    "N_NAME": '<U25',
    "N_REGIONKEY": int,
    "N_COMMENT": '<U152'
}

region_schema = {
    "R_REGIONKEY": int,
    "R_NAME": '<U25',
    "R_COMMENT": '<U152'
}
