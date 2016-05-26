import logging

import constants
import pygeoutil.util as util
from preprocess_IAM import IAM


class IMAGE:
    """
    Class for IMAGE
    """
    def __init__(self, path_nc):
        IAM.__init__(self, 'IMAGE', path_nc)