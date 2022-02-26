import bz2
import pickle
import _pickle as cPickle


def full_pickle(filename=None, data=None):
    """
    Saves the 'data' with the 'filename' as pickle

    """

    f = open(filename + '.pickle', 'wb')
    pickle.dump(data, f)
    f.close()


def unpickle(filename=None):
    """
    Loads and returns a pickled object.

    """

    f = open(filename + '.pickle', 'rb')
    data = pickle.load(f)
    f.close()

    return data


def compress_pickle(filename=None, data=None):
    """
    Pickle a file and then compress it into BZ2 file.

    """

    with bz2.BZ2File(filename + '.pbz2', 'wb') as f:
        cPickle.dump(data, f)


def decompress_pickle(filename=None):
    """
    Load any compressed pickle file.

    """

    data = bz2.BZ2File(filename + '.pbz2', 'rb')
    data = cPickle.load(data)

    return data

# _______________________________NOTES_________________________________________

# 1. http://www.linfo.org/bzip2.html for details on bzip2 file compression
