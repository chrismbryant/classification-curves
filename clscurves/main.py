

# DBTITLE 1,Imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict, Any
from scipy.integrate import trapz
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pyspark.sql.functions as F

import warnings