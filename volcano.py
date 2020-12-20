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

train = pd.read_csv('../GitHub/Volcanic-Eruption-Prediction/train.csv')
test = pd.read_csv('../GitHub/Volcanic-Eruption-Prediction/sample_submission.csv')

def df_parallelize_run(func, t_split, cores=4):
    """
    inspired by https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic
    """
    num_cores = np.min([cores, len(t_split)])
    pool = Pool(processes=num_cores)
    df = pd.concat(pool.starmap(func, t_split), axis=0)
    pool.close()
    pool.join()

    return df

def get_basic_features(df, direct='train'):

    result = pd.DataFrame(dtype=np.float32)
    result['segment_id'] = df['segment_id']
    result['time_to_eruption'] = df['time_to_eruption']
    result.set_index('segment_id', inplace=True)

    for ids in df['segment_id']:

        f = pd.read_csv(f'../GitHub/Volcanic-Eruption-Prediction/{direct}/{ids}.csv')
        f['sensor_all'] = f.sum(1)                          # Adding another read that will be the sum of all sensors
                                                            # Or you can use other statistics as well (min, max, etc)
        for sensor in f.columns:

            s_ = f[sensor].ffill().fillna(0).values         # filling nans
            raw_s = s_                                      # raw signal
            filtered_s = signal.medfilt(s_, 5)              # median filtered signal with window=5
            cos_s = fft.dct(s_)                             # direct cosine transformed signal

            for num, data in enumerate([raw_s, cos_s, filtered_s]):

                mean = np.mean(data)                        # mean
                std = np.std(data, ddof=0)                  # std
                min_ = np.min(data)                         # minimum
                max_ = np.max(data)                         # maximum
                mr = (min_ + max_)*0.5                      # midrange
                ptp = max_ - min_                           # range
                q1, q3 = np.quantile(data, q=[0.25, 0.75])  # 1st & 3rd quartile
                mh = (q3 + q1)*0.5                          # midhinge
                iqr = q3 - q1                               # IQR
                mad_1 = np.median(np.abs(data - mean))      # Median deviation from the mean
                mad_2 = np.mean(np.abs(data - mean))        # Mean deviation from the mean
                bwl = biweight_location(data)               # Biweight location
                bws = biweight_scale(data)                  # Biweight scale

                result.loc[ids, f'{sensor}_mean_{num}'] = mean
                result.loc[ids, f'{sensor}_std_{num}'] = std
                result.loc[ids, f'{sensor}_midrange_{num}'] = mr
                result.loc[ids, f'{sensor}_midhinge_{num}'] = mh
                result.loc[ids, f'{sensor}_min_{num}'] = min_
                result.loc[ids, f'{sensor}_max_{num}'] = max_
                result.loc[ids, f'{sensor}_ptp_{num}'] = ptp
                result.loc[ids, f'{sensor}_q1_{num}'] = q1
                result.loc[ids, f'{sensor}_q3_{num}'] = q3
                result.loc[ids, f'{sensor}_iqr_{num}'] = iqr
                result.loc[ids, f'{sensor}_med_abs_dev_mean_{num}'] = mad_1
                result.loc[ids, f'{sensor}_mean_abs_dev_mean_{num}'] = mad_2
                result.loc[ids, f'{sensor}_biweight_location_{num}'] = bwl
                result.loc[ids, f'{sensor}_biweight_scale_{num}'] = bws

    return result

#What else could be added here?

#Add different modification of signal: abs signal, FFT, DST, MFCC/Cepstrum, different filters with different windows,
# STA/LTA (1s and 60s default, with 100Hz sampling frequency it will be 100 and 6000 samples), 1st/2nd derivative and their abs,
#1st/2nd percentage change and their abs, cumulative and window statistics;
#Add other statistics to compute. I removed a lot just because I do not want to impute NaN's that will emerge, but these features
# also could give additional information. Examples will be kurtosis, skewness, CV, informational entropy. You can also add other
#TS features to the mix (from tsfresh or tslearn libs);
#Watch out for computation times, number of features and collinearity! Use L1/L2 regulariazations (and feature subsampling
#if you are planing to use DT-based models). Blindly adding additional features and signals will lead to bloated dataset
#which will eat your RAM alive. Do not throw everything in the mixer.

N_CORES = 4
chunk_size = train.shape[0] // N_CORES
splits_train = [train[:chunk_size], train[chunk_size:2*chunk_size], train[2*chunk_size:3*chunk_size], train[3*chunk_size:]]
splits_test = [test[:chunk_size], test[chunk_size:2*chunk_size], test[2*chunk_size:3*chunk_size], test[3*chunk_size:]]

%%time

train_df = df_parallelize_run(get_basic_features,
                              [(chunk, 'train') for chunk in splits_train],
                              N_CORES)

%%time

test_df = df_parallelize_run(get_basic_features,
                             [(chunk, 'test') for chunk in splits_test],
                             N_CORES)

print(f'Features containing nans in train are {train_df.columns[(train_df.replace((np.inf, -np.inf), np.nan).isna().sum() > 0)].tolist()}')
print(f'Features containing nans in test are {test_df.columns[(test_df.replace((np.inf, -np.inf), np.nan).isna().sum() > 0)].tolist()}')
print(f'Low variation features in train are {train_df.columns[train_df.var() <= 1].tolist()}')
print(f'Low variation features in test are {test_df.columns[test_df.var() <= 1].tolist()}')

X = train_df.drop('time_to_eruption', axis=1)
X_test = test_df.drop('time_to_eruption', axis=1)
y = train_df['time_to_eruption']

X.corrwith(y).abs().sort_values(ascending=False)
X.corrwith(np.sqrt(y)).abs().sort_values(ascending=False)

%%time

for col in X.columns:
    p = stats.ks_2samp(X[col], X_test[col], mode='exact', alternative='two-sided')[1]
    if p < 0.05:
        print(col)

First of all we will use Lasso to select features.

# We will need StandardScaler to scale our data, Lasso to get coefs, TargetTransformer to transform our targets and
# cross-validation to tune hyperparameters.
# We will use repeated K fold validation scheme with 3 splits and 4 repetitions. Number of splits were chosen based on
# factorization of number of rows in train: 4431 = 3 ∙ 7 ∙ 211. Number of reprtitions were chosen to get number of folds
# divisible by 4 (number of cores) to get (hopefully) the most out of parallel computations.

rkf = RepeatedKFold(n_splits=3, n_repeats=4, random_state=1)

scaler_lasso = StandardScaler()
lasso = Lasso(alpha=2.5, fit_intercept=True, normalize=False, max_iter=20_000, tol=1e-3, positive=False, random_state=1, selection='random')
tt_lasso = TransformedTargetRegressor(regressor=lasso, func=np.sqrt, inverse_func=np.square)
pipe_lasso = Pipeline([('scaler', scaler_lasso), ('reg', tt_lasso)])
cv_lasso = cross_val_score(pipe_lasso,
                           X,
                           y,
                           scoring='neg_mean_absolute_error',
                           cv=rkf,
                           n_jobs=4,
                           verbose=True,
                           error_score='raise')

print(f'Mean MAE is {(cv_lasso*-1).mean()}')
print(f'STD MAE is {(cv_lasso*-1).std()}')

pipe_lasso.fit(X, y)
lasso_coefs = pipe_lasso.named_steps['reg'].regressor_.coef_
important_cols = X.columns[lasso_coefs != 0]

scaler_svr = StandardScaler()
lin_svr = LinearSVR(epsilon=0, tol=1e-2, C=333, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=1, max_iter=1000)
tt_svr = TransformedTargetRegressor(regressor=lin_svr, func=np.sqrt, inverse_func=np.square)
pipe_svr = Pipeline([('scaler', scaler_svr), ('reg', tt_svr)])
cv_svr = cross_val_score(pipe_svr,
                         X[important_cols],
                         y,
                         scoring='neg_mean_absolute_error',
                         cv=rkf,
                         n_jobs=4,
                         verbose=True,
                         error_score='raise')

print(f'Mean MAE is {(cv_svr*-1).mean()}')
print(f'STD MAE is {(cv_svr*-1).std()}')

pipe_svr.fit(X[important_cols], y)
predictions = pipe_svr.predict(X_test[important_cols])

## What else could be done here?

#You can use boosting. I personally do not recommend it. GBDT can only interpolate, it could not extrapolate.
# So, if test set has target values that are out of range of train target values, you won't be able to predict them properly.
# Linear models or NNs can. You can use blend of GBDT and LR, this will be the best combination.
# Another option will be XGBoost with boosting='gblinear';
# You can use different scalers and different target transform functions (log, 1/x, cbrt);
# You can predict hours, minutes, seconds and mseconds, then blend results together.
# Or you can split target into hours-minutes-seconds-ms, predict them and then sum results.

test['time_to_eruption'] = predictions
test
test.to_csv('submission.csv', index=False)
