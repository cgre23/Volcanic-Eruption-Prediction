import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal, fft
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from astropy.stats import biweight_location, biweight_scale
import random
import os

np.random.seed(1)
random.seed(1)
os.environ['PYTHONHASHSEED'] = '0'

train = pd.read_csv('../Github/Volcanic-Eruption-Prediction/train.csv')
test = pd.read_csv('../GitHub/Volcanic-Eruption-Prediction/sample_submission.csv')
