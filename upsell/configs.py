import yaml


class DotDict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_to_dot_dict(dict_):
    """ Convert a regular dictionary to a DotDict """
    dict_ = DotDict(dict_)
    for key in dict_:
        if type(dict_[key]) == dict:
            dict_[key] = convert_to_dot_dict(dict_[key])
    return dict_


def load_yaml_config(filepath):
    """ Loads config file and converts to nested dot dict """
    configs = yaml.safe_load(open(filepath, 'r'))
    return convert_to_dot_dict(configs)
