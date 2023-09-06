import sys
import yaml
import time

try:
    from typing import List
except ImportError:
    from typing_extensions import List

def timer(func: callable) -> callable:
    """ Wrapper that calculates the execution time of functions. """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def flatten_dict(d: dict, separator='_', prefix='') -> dict:
    """ Flattens a nested dictionary. """
    res = {}
    for k,v in d.items():
        if isinstance(v, dict):
            res.update(flatten_dict(v, separator, prefix+k+separator))
        else:
            res[prefix+k] = v
    return res

def fix_dictionary_keys(d: dict) -> dict:
    """ Fix to ensure that all dictionary keys have the same data type. """
    fixed_dict = {}
    for k,v in d.items():
        if isinstance(v,dict):
            v = fix_dictionary_keys(v)
        fixed_dict[str(k)] = v
    return fixed_dict

def read_yaml(yaml_file: str) -> dict:
    """ Reads YAML file from the given path, returns data as python dictionary. """
    with open(yaml_file, "r") as yf:
        try:
            data_dict = yaml.safe_load(yf)
            data_dict = fix_dictionary_keys(data_dict)
            return data_dict
        except yaml.YAMLError as yaml_error:
            sys.exit(yaml_error)

def report_results(results_dict: dict) -> None:
    """ Prints the content of the given dictionary in a prettified format. """
    for k,v in results_dict.items():
        label_id, label_name, score = v
        print(
            f"""####### LEVEL {k}:
            \t- Class:            {label_id}
            \t- Full class name:  {label_name}
            \t- Confidence score: {round(score,4)}"""
        )

def check_consecutive(l: List) -> bool:
    """ Checks if all the elements of a given list are consecutive numbers. """
    try:
        l = [int(elem) for elem in l if not isinstance(elem,int)]
        sorted_list = sorted(l)
        ranged_list = list(range(min(l), max(l)+1))
    except TypeError:
        return False
    return sorted_list == ranged_list
