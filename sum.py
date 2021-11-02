def sum_col(table, col):
    idx = table.col_names.index(col)
    S = 0
    if table.storage == "row":
        for row in table.data:
            S = S + row[idx]

        return S
    else:
        column_To_sum = table.data[idx]

        for val in column_To_sum:
            S = S + val

        return S