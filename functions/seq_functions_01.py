import functools
from typing import Final, List, Mapping, Tuple
import numpy as np
import collections
from collections import namedtuple
from collections.abc import Mapping, Sequence
import itertools
import constants

def seq_to_onehot(sequence: str) -> np.ndarray:
    """
    Maps the given sequence of length L into a one-hot 
    encoded matrix.
    
    Args: An RNA sequence with or without alignment gaps.
    Returns: A numpy array of shape 4xL with one-hot 
    encodings of the sequence.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    
    one_hot_arr = np.zeros((len(sequence), 4), dtype=np.int32)
    
    for index, nuc in enumerate(sequence):
        if nuc != "-":
            one_hot_arr[index, mapping[nuc]] = 1
    
    return one_hot_arr

def onehot_to_seq(array: np.ndarray) -> str:
    """
    Maps the given array of shape 4xL to sequence.
    
    Args: An RNA sequence with or without gaps.
    Returns: A sequence str of length L from a 
    one-hot encoding.
    """
    mapping = {0:"A", 1:"C", 2:"G", 3:"U"}
    arr_shape = array.shape
    seq_l = arr_shape[1]
    seq = ""
    
    if arr_shape[0] != 4:
        print(f"ERROR: expected shape (4,{seq_l}), got {arr_shape} instead!")
        print("Fix/adjust input array data.")
    else:
        for i in range(len(arr)):
            if 1 in arr[i]:
                seq = seq+mapping[list(arr[i]).index(1)]
            else:
                seq = seq +"-"
        return seq
