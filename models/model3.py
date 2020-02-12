import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import uuid
import xgboost as xgb
import lightgbm as lgb
import os
from sklearn.preprocessing import StandardScaler
import pickle

PICKLE_DIR = '../pickle'
os.makedirs(PICKLE_DIR, exist_ok=True)


def pickle_it(expr_name):
    def pickle_func(func):
        @timeit
        def wrapper(*args, **kwargs):
            pickle_file = f'{PICKLE_DIR}/{expr_name}'
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as ifile:
                    return pickle.load(ifile)
            else:
                res = func(*args, **kwargs)
                with open(pickle_file, 'wb') as ofile:
                    pickle.dump(res, ofile)
            return res
        return wrapper
    return pickle_func


class Expr:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)


expr = Expr(expr_name='model3',
            model_name='lgb',
            width1=5,
            width2=10,
            width3=40,
            n_estimators=100,
            max_depth=15,
            metric='mape',
            # metric='mae',
            # metric='mse',
            objective='regression',
            num_leaves=10,
            use_pca=True)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = pd.Timestamp.now()
        res = func(*args, **kwargs)
        elapsed = pd.Timestamp.now() - start
        return res, elapsed

    return wrapper


def smape(satellite_predicted_values, satellite_true_values):
    # the division, addition and subtraction are pointwise
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values)
                          / (np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))


def get_df_stats(df):
    stat_lines = []
    if 'Vx' not in df.columns:
        Vx_col, Vy_col, Vz_col = 'Vx_sim', 'Vy_sim', 'Vz_sim'
    else:
        Vx_col, Vy_col, Vz_col = 'Vx', 'Vy', 'Vz'

    for sat_id, df_sat in df.groupby('sat_id'):
        sat_stats = dict(sat_id=sat_id,
                         start=df_sat['epoch'].min(),
                         end=df_sat['epoch'].max(),
                         n_points=len(df_sat),
                         Vx_mean=df_sat[Vx_col].mean(),
                         Vy_mean=df_sat[Vy_col].mean(),
                         Vz_mean=df_sat[Vz_col].mean())
        stat_lines.append(sat_stats)
    df_stats = pd.DataFrame(stat_lines)
    return df_stats


def get_df_simple_smape_score(y_pred, y_test):
    sat_smape_score = [smape(y_pred[:, i], y_test[:, i]) for i in range(6)]
    mean_smape_score = np.mean(sat_smape_score)
    return 100 * (1 - mean_smape_score)


def get_df_smape_score(sat_id_list, y_pred, y_test):
    if isinstance(sat_id_list, np.ndarray):
        sat_id_list = sat_id_list.tolist()

    reversed_sat_id_list = sat_id_list[::-1]
    smape_scores = []
    sats_smape_scores = []
    for sat_id in sorted(set(sat_id_list)):
        a = sat_id_list.index(sat_id)
        b = len(sat_id_list) - reversed_sat_id_list.index(sat_id)  # dirty hack
        sat_train_vals = y_pred[a:b]
        sat_test_vals = y_test[a:b]
        sat_smape_score = np.mean([smape(sat_train_vals[:, i], sat_test_vals[:, i]) for i in range(6)])
        #         print('sat_id, a, b, smape', sat_id, a, b, sat_smape_score)
        sats_smape_scores.append(dict(sat_id=sat_id,
                                      a=b,
                                      b=b,
                                      sat_smape_score=sat_smape_score,
                                      rows=len(sat_train_vals)))
        smape_scores.append(sat_smape_score)
    mean_smape_score = np.mean(smape_scores)
    score = 100 * (1 - mean_smape_score)

    df_sats_smapes = pd.DataFrame(sats_smape_scores)
    return score, df_sats_smapes


def smape_objective(y_true, y_pred):
    grad = (y_pred - y_true)
    hess = np.ones(len(y_true))
    return grad, hess


@timeit
def baseline(X_train, y_train, X_test, model_name):
    if model_name == 'linear':
        regr_multirf = MultiOutputRegressor(LinearRegression())
    elif model_name == 'ridge':
        regr_multirf = MultiOutputRegressor(Ridge())
    elif model_name == 'lasso':
        regr_multirf = MultiOutputRegressor(Lasso())
    elif model_name == 'randomforest':
        regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                                  max_depth=2,
                                                                  random_state=0))
        # first run local mean smape 0.84345, public 17.47
        # too long
    elif model_name == 'xgb':
        regr_multirf = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100,
                                                             max_depth=2,
                                                             random_state=0))
    elif model_name == 'lgb':
        regr_multirf = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=expr.n_estimators,
                                                              max_depth=expr.max_depth,
                                                              random_state=0,
                                                              # objective=smape_objective,
                                                              objective=expr.objective,
                                                              num_leaves=expr.num_leaves,
                                                              metric=expr.metric
                                                              ))
    else:
        raise Exception('unknown model', model_name)
    regr_multirf.fit(X_train, y_train)
    y_pred = regr_multirf.predict(X_test)
    return regr_multirf, y_pred


def add_features(df, width):
    df_shift = df.shift(width, fill_value=0).add_suffix('_shift')
    df_diff = df.diff().fillna(0).add_suffix('_diff')
    df_diff2 = df_diff.diff().fillna(0).add_suffix('_diff2')
    # df_log = np.log(df).add_suffix('_log')
    df_sin = np.sin(df).add_suffix('_sin')
    df_cos = np.cos(df).add_suffix('_cos')
    df_diff_sin = np.sin(df_diff).add_suffix('_diff_sin')
    df_diff_cos = np.cos(df_diff).add_suffix('_diff_cos')
    df_diff2_sin = np.sin(df_diff2).add_suffix('_diff2_sin')
    df_diff2_cos = np.cos(df_diff2).add_suffix('_diff2_cos')
    # df_exp = np.exp(df).add_suffix('_exp')
    # df_diff_exp = np.exp(df_diff).add_suffix('_diff_exp')
    # df_diff2_exp = np.exp(df_diff2).add_suffix('_diff2_exp')
    df_rolling = df_shift.rolling(window=width)

    if 'y_sim' in df.columns:
        df['center_dist'] = (df.x_sim ** 2 + df.y_sim ** 2 + df.z_sim ** 2) ** 0.5
        df['V_total'] = (df.Vx_sim ** 2 + df.Vy_sim ** 2 + df.Vz_sim ** 2) ** 0.5

    # df_norm = pd.DataFrame(StandardScaler().fit_transform(df)).add_suffix('_norm')

    df_new = pd.concat([df,
                        df_shift,
                        df_rolling.min().add_suffix('_min'),
                        df_rolling.mean().add_suffix('_mean'),
                        df_rolling.max().add_suffix('_max'),
                        df_diff,
                        df_diff2,
                        # df_log,
                        df_sin,
                        df_cos,
                        df_diff_sin,
                        df_diff_cos,
                        df_diff2_sin,
                        df_diff2_cos,
                        # df_diff_exp,
                        # df_diff2_exp,
                        # df_exp,
                        ],
                       axis=1,
                       ignore_index=True)
    print('add features', df_new.columns.tolist())
    df_new = df_new.fillna(0)
    return df_new


def add_features_levels(df_sat, train_cols):
    df_sat_features1 = add_features(df_sat[train_cols], expr.width1)
    df_sat_features2 = add_features(df_sat[train_cols], expr.width2)
    df_sat_features3 = add_features(df_sat[train_cols], expr.width3)
    df_sat_features = pd.concat([df_sat_features1, df_sat_features2, df_sat_features3],
                                ignore_index=True,
                                axis=1)
    # if expr.use_pca:
    #     df_pca = pd.DataFrame(PCA(n_components=6).fit_transform(df_sat_features))\
    #         .add_suffix('_pca')
    #     df_sat_features = pd.concat([df_sat_features, df_pca],
    #                                 ignore_index=True,
    #                                 axis=1)
    return df_sat_features


def get_Xy_data(df, train_cols):
    X_lines = []
    y_lines = []
    sat_id_list = []

    for sat_ind, df_sat in df.groupby('sat_id', as_index=False):
        df_sat.index = df_sat['epoch']
        # add features
        df_sat_features = add_features_levels(df_sat, train_cols)
        sat_train_vals = df_sat_features.values
        sat_test_vals = df_sat[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']].values

        X_lines.extend(sat_train_vals.tolist())
        y_lines.extend(sat_test_vals.tolist())
        sat_id = df['sat_id'].values[0]
        sat_id_list.extend([sat_id] * len(sat_train_vals))

    X = np.array(X_lines)
    y = np.array(y_lines)
    return X, y, sat_id_list


@pickle_it(expr_name=expr.expr_name)
def load_data():
    # input_dir = '/content/drive/My Drive/IDAO 2020/'
    input_dir = '../IDAO 2020/'

    df = pd.read_csv(f'{input_dir}/train.csv')
    df['epoch'] = pd.to_datetime(df.pop('epoch'))
    df.index = df['epoch']
    df.index = df.index.round('1min')
    df['epoch'] = df.index

    test_data = pd.read_csv(f'{input_dir}Track 1/test.csv')

    # !!! save only test satellites
    # test_id_list = list(test_data['sat_id'].unique())
    # print('train data init len', len(df))
    # df = df.query('sat_id in @test_id_list')
    # print('train data final len', len(df))

    test_data['epoch'] = pd.to_datetime(test_data['epoch'])
    # test_subm = pd.read_csv(f'{input_dir}/Track 1/submission.csv')
    # df_stats = get_df_stats(df)
    # df_stats_test = get_df_stats(test_data)
    # train_cols = ['sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
    train_cols = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
    X, y, sat_id_list = get_Xy_data(df, train_cols)

    sat_id_list = np.array(sat_id_list)
    X_train, X_test, y_train, y_test, sat_id_list_train, sat_id_test = train_test_split(X, y, sat_id_list,
                                                                                        shuffle=True,
                                                                                        test_size=0.1)
    print('shapes:',
          X_train.shape,
          X_test.shape,
          y_train.shape,
          y_test.shape,
          sat_id_list_train.shape,
          sat_id_test.shape)

    simple_mean_smape_score = get_df_simple_smape_score(X_train, y_train)
    print('simple mean_smape_score', simple_mean_smape_score)
    mean_smape_score, df_sats_smapes = get_df_smape_score(sat_id_list_train, X_train, y_train)
    print('mean_smape_score', mean_smape_score)
    return X_train, y_train, X_test, y_test, sat_id_test, test_data, train_cols


@timeit
def main():
    # %%
    (X_train, y_train, X_test, y_test, sat_id_test, test_data, train_cols), elapsed_time = load_data()
    # %%
    (regr_multirf, y_pred), elapsed = baseline(X_train, y_train, X_test, model_name=expr.model_name)
    print('elapsed time', elapsed)
    smape_score, df_score_test = get_df_smape_score(sat_id_test.tolist(), y_pred, y_test)
    print('smape local test score:', smape_score)

    expr_id = uuid.uuid1()

    need_save_submission = True
    if need_save_submission:
        # save submission
        X_submission = add_features_levels(test_data, train_cols)
        y_submission = regr_multirf.predict(X_submission)

        df_submission = pd.DataFrame(y_submission, columns=['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
        df_submission['id'] = test_data['id']
        df_submission = df_submission[['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']]
        df_submission.to_csv(f'{expr_id}.csv', index=False)

        model_info = f'MultiOutputRegressor+{expr.model_name}'

        ask_public_score = True

        with open('../expr_log.txt', 'a') as ofile:
            if ask_public_score:
                public_score = input('enter public score: ')
            else:
                public_score = None
            print(expr.expr_name, expr_id, smape_score, public_score,
                  pd.Timestamp.now(), model_info, elapsed, expr,
                  file=ofile, sep='\t')


if __name__ == '__main__':
    main()
