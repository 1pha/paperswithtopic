import yaml
import easydict


CONFIG_FILE_PATH = './config.yml'


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, easydict.EasyDict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj 


def save_config(cfg, path=CONFIG_FILE_PATH):

    if isinstance(cfg, easydict.EasyDict):
        cfg = edict2dict(cfg)

    with open(path, 'w') as y:
        yaml.dump(cfg, y)


def load_config(path=CONFIG_FILE_PATH):

    with open(path, 'r') as y:
        return easydict.EasyDict(yaml.load(y, Loader=yaml.FullLoader))
