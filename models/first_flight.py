import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import uuid


def timeit(func):
    def wrapper(*args, **kwargs):
        start = pd.Timestamp.now()
        res = func(*args, **kwargs)
        elapsed = pd.Timestamp.now() - start
        return res, elapsed
    return wrapper


class Expr:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)


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
                         Vz_mean=df_sat[Vz_col].mean()
        )
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


@timeit
def baseline(X_train, y_train, X_test, model_name):
    if model_name == 'linear':
        regr_multirf = MultiOutputRegressor(LinearRegression())
    elif model_name == 'ridge':
        regr_multirf = MultiOutputRegressor(Ridge())
    elif model_name == 'lasso':
        regr_multirf = MultiOutputRegressor(Lasso())
    elif model_name == 'xgb':
        regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                                  max_depth=2,
                                                                  random_state=0))
        # first run local mean smape 0.84345, public 17.47
        # too long
    else:
        raise Exception('unknown model', model_name)

    regr_multirf.fit(X_train, y_train)
    y_pred = regr_multirf.predict(X_test)
    return regr_multirf, y_pred


def add_features(df, width):
    # add additional columns
    df_shift = df.shift(width, fill_value=0).add_suffix('_shift')
    df_rolling = df_shift.rolling(window=width)
    df_new = pd.concat([df,
                        df_shift,
                        df_rolling.min().add_suffix('_min'),
                        df_rolling.mean().add_suffix('_mean'),
                        df_rolling.max().add_suffix('_max'),
                        ],
                       axis=1,
                       ignore_index=True)
    df_new = df_new.fillna(0)
    return df_new


def main():
    # input_dir = '/content/drive/My Drive/IDAO 2020/'
    input_dir = 'IDAO 2020/'
    df = pd.read_csv(f'{input_dir}/train.csv')
    df['epoch'] = pd.to_datetime(df.pop('epoch'))
    df.index = df['epoch']
    df.index = df.index.round('1min')
    df['epoch'] = df.index

    test_data = pd.read_csv(f'{input_dir}Track 1/test.csv')
    test_data['epoch'] = pd.to_datetime(test_data['epoch'])
    test_subm = pd.read_csv(f'{input_dir}/Track 1/submission.csv')

    df_stats = get_df_stats(df)

    X_lines = []
    y_lines = []
    sat_id_list = []
    train_cols = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']

    for sat_id, df_sat in df.groupby('sat_id'):
        df_sat.index = df_sat['epoch']
        df_sat1 = df_sat[:pd.to_datetime('2014-01-15 23:59:59.999')]
        df_sat2 = df_sat['2014-01-16 00:00:00':]
        if df_sat2.__len__() > df_sat1.__len__():
            df_sat2 = df_sat2.head(df_sat1.__len__())
        else:
            raise Exception('wrong len')

        df_sat1_features = add_features(df_sat1[train_cols], expr.width)
        sat_train_vals = df_sat1_features.values
        sat_test_vals = df_sat2[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']].values

        X_lines.extend(sat_train_vals.tolist())
        y_lines.extend(sat_test_vals.tolist())
        sat_id_list.extend([sat_id] * len(sat_train_vals))

    X = np.array(X_lines)
    y = np.array(y_lines)

    sat_id_list = np.array(sat_id_list)
    X_train, X_test, y_train, y_test, sat_id_list_train, sat_id_test = train_test_split(X, y, sat_id_list,
                                                                                        shuffle=False)
    print('shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape, sat_id_list_train.shape,
          sat_id_test.shape)
    simple_mean_smape_score = get_df_simple_smape_score(X_train, y_train)
    print('simple mean_smape_score', simple_mean_smape_score)
    mean_smape_score, df_sats_smapes = get_df_smape_score(sat_id_list_train, X_train, y_train)
    print('mean_smape_score', mean_smape_score)
    df_sats_smapes = df_sats_smapes.join(df_stats, on='sat_id', how='inner', lsuffix='_smapes')

    (regr_multirf, y_pred), elapsed = baseline(X_train, y_train, X_test, model_name=expr.model_name)
    print('elapsed time', elapsed)
    smape_score, df_score_test = get_df_smape_score(sat_id_test.tolist(), y_pred, y_test)
    print('smape local test score:', smape_score)

    expr_id = uuid.uuid1()

    # save submission
    X_submission = add_features(test_data[train_cols], width=expr.width)
    y_submission = regr_multirf.predict(X_submission)

    df_submission = pd.DataFrame(y_submission, columns=['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
    df_submission['id'] = test_data['id']
    df_submission = df_submission[['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']]
    df_submission.to_csv(f'{expr_id}.csv', index=False)

    model_info = f'MultiOutputRegressor+{expr.model_name}'
    ask_public_score = False

    with open('../expr_log.txt', 'a') as ofile:
        if ask_public_score:
            public_score = input('enter public score: ')
        else:
            public_score = None
        print(expr.expr_name, expr_id, smape_score, public_score,
            pd.Timestamp.now(), model_info, elapsed, expr,
            file=ofile, sep='\t')


if __name__ == '__main__':
    expr = Expr(expr_name='baseline', model_name='lasso', width=3)
    main()
