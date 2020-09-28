# author: kcgarikipati@gmail.com
import argparse
from .general import merge_dicts, hash_of_str
import re
import json


class SettingsManager:
    """manager class for settings
    """

    def __init__(self, description='', string_separator='_'):
        self.parser_cmd = argparse.ArgumentParser(description=description)
        self.settings_dict = {}
        self.string_separator = string_separator

    def __str__(self):
        return self.get_str(select_keys=list(self.settings_dict.keys()))

    def __dict__(self):
        return self.get_dict(select_keys=list(self.settings_dict.keys()))

    def get_str(self, select_keys):
        """get settings in string representation based on selected keys
        """
        ret_str = ""
        for key in select_keys:
            if ret_str != "":
                ret_str += self.string_separator
            if key in self.settings_dict:
                val = self.settings_dict[key]
            else:
                print("Key {} does not exist".format(key))
                break
            if type(val) is list:
                val = '_'.join(val)

            ret_str += str(key) + self.string_separator + str(val)
        return ret_str

    def get_dict(self, select_keys):
        """get settings in dict representation based on selected keys
        """
        select_dict = dict((k, v) for k, v in self.settings_dict.items() if k in select_keys)
        return select_dict

    def get_hash(self, select_keys):
        """get hash of settings in string representation based on selected keys
        """
        dict_select = self.get_dict(select_keys)
        return hash_of_str(json.dumps(dict_select, sort_keys=True))

    def parse_cmd(self):
        """update settings by parsing command line
        """
        args = self.parser_cmd.parse_args()
        # get args from regular parsing
        args_dict = vars(args)
        return args_dict

    def parse_file(self, filename):
        """update settings by parsing settings file
        """
        raise NotImplementedError

    def update(self, input_dict):
        """update the settings dict
        """
        for key, val in input_dict.items():
            if val is not None:
                self.settings_dict[key] = val
        return self.settings_dict

    def create(self, arguments):
        """add arguments to parser
        """
        for item in arguments:
            # add default val to the dict
            self.settings_dict[item[0]] = item[2]
            arg = '--' + item[0].lower()
            dest = item[0]
            help_ = item[3] if item[3] else None
            choices = item[4] if item[4] else None

            if item[1] == bool:
                if item[2] is False:
                    self.parser_cmd.add_argument(arg, dest=dest, action="store_true", help=help_)
                else:
                    self.parser_cmd.add_argument(arg, dest=dest, action="store_false", help=help_)
            elif item[1] == list:
                self.parser_cmd.add_argument(arg, nargs='+', type=str, dest=dest, help=help_, choices=choices)
            else:
                self.parser_cmd.add_argument(arg, type=item[1], dest=dest, help=help_, choices=choices)

    @staticmethod
    def get(arguments, input_dict={}):
        """return object of this class
        """
        cm = SettingsManager()
        cm.create(arguments)
        cm.update(input_dict)
        return cm


def test_settings_manager():
    options =\
    [
        ('exp', str, 'testexp', 'experiment name'),
        ('model', list, 'unet', 'type of model'),
        ('nlayers', int, 2, 'num of layers')
    ]
    cm = SettingsManager.get(options)
    cm.update({'batch_size': 1})
    args_dict = cm.parse_cmd()
    print(args_dict)
    cm.update(args_dict)
    print(cm.__dict__())


if __name__ == '__main__':
    test_settings_manager()