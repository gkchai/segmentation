# author: kcgarikipati@gmail.com
import os
import hashlib
import pandas as pd
import logging
import sys
import numpy as np
import time


def merge_dicts(*dict_args):
    """Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def dict_to_obj(input_dict):
    """Turns a dictionary into a class
    """
    class C(object):
        def __init__(self, dictionary):
            """Constructor"""
            for key in dictionary:
                setattr(self, key, dictionary[key])
    return C(input_dict)


def dict_in_another(subdict, supdict):
    """check if given dict is subset of another
    """
    return subdict.items() <= supdict.items()


def dataset_to_path(filename, base_path, suffix):
    """convert dataset filename(s) to absolute path(s)
    """
    if type(filename) is list:
        return [os.path.join(base_path, fn + '_' + str(suffix) + '.csv') for fn in filename]
    else:
        return os.path.join(base_path, filename + '_' + str(suffix) + '.csv')


def path_to_dataset(filepaths, suffix):
    """convert absolute filepath(s) to dataset(s)
    """
    def extract(basename, suff):
        l = basename.split("_")
        if l[-1] != str(suff)+".csv":
            raise ValueError("can't parse with given suffix")
        return "_".join(l[:-1])

    if type(filepaths) is list:
        return [extract(os.path.basename(fn),suffix) for fn in filepaths]
    else:
        return extract(os.path.basename(filepaths),suffix)


def hash_of_str(string_, len=10):
    """hash of given string of length len.  Check for collision?
    """
    return str(hashlib.sha1(string_.encode('utf-8')).hexdigest()[:len])


def n_in_csv(files_csv):
    """count the number of items in list of csv files
    """
    assert(type(files_csv) == list)
    nums = []
    for file_csv in files_csv:
        data = pd.read_csv(file_csv, header=None)
        nums.append(data.shape[0])
    return sum(nums), nums


def get_logger(modulename, filename=None):
    """Return a logger instance that writes in stdout and filename
    Args:
        filename: (string) path to log.txt
    Returns:
        logger: (instance of logger)
    """
    # set basic config for stdout logging
    log_format = '%(asctime)s:%(name)s:%(levelname)s:%(message)s'
    logger = logging.getLogger(modulename)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format=log_format, level=logging.DEBUG)
    # add file config if specified
    if filename:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
    return logger


def make_dir(dir):
    """mkdir if dir does not exist
    """
    if not os.path.exists(dir):
        os.mkdir(dir)


def silence_tf_logger():
    """silence tf python logger
    """
    logging.getLogger('tensorflow').disabled = True


def k_largest_array(arr, k=2):
    """k largest values in array
    """
    return sorted(list(set(arr.flatten().tolist())))[-k]


def is_equal(a, b, eps = 10^-10):
    """are two (floats) equal
    """
    return abs(a-b) < eps


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


