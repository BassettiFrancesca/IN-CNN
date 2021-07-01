def variance(v):
    m = sum(v) / len(v)
    s = 0
    for i in v:
        q = pow((i - m), 2)
        s += q
    var = s / len(v)

    return var
