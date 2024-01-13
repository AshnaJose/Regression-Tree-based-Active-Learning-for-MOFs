def unbiased_variance(labels):
    n = len(labels)
    if n < 2:
        return 0

    mean = sum(labels)/n
    tot = 0
    for val in labels:
        tot += (mean - val)**2

    return tot/(n-1)
