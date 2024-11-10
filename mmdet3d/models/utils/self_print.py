def print2file(content, suffix='', end='.json',mode='w'):
    f = open('/mnt/data/exps_logs/out_' + suffix + end, mode=mode)
    print(content, file=f)
