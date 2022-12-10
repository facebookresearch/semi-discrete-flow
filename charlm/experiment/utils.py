import copy
from prettytable import PrettyTable


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def get_metric_table(metric_dict, epochs):
    max_len = len(epochs)
    for _, vs in metric_dict.items():
        max_len = min(max_len, len(vs))
    table = PrettyTable()
    table.add_column('Epoch', epochs[:max_len])
    if len(metric_dict)>0:
        for metric_name, metric_values in metric_dict.items():
            table.add_column(metric_name, metric_values[:max_len])
    return table


def clean_dict(d, keys):
    d2 = copy.deepcopy(d)
    for key in keys:
        if key in d2:
            del d2[key]
    return d2
