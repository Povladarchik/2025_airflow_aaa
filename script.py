import argparse
import os
from datetime import timedelta

import polars as pl
from scipy.sparse import csr_matrix
import numpy as np
import implicit
import mlflow
import mlflow.sklearn


EVAL_DAYS_TRESHOLD = 14
DATA_DIR = os.getenv("DATA_DIR", "data")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "homework-ispopravka")


def get_data():
    df_test_users = pl.read_parquet(f'{DATA_DIR}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{DATA_DIR}/clickstream.pq')
    df_event = pl.read_parquet(f'{DATA_DIR}/events.pq')
    return df_test_users, df_clickstream, df_event


def split_train_test(df_clickstream: pl.DataFrame, df_event: pl.DataFrame):
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)

    df_train = df_clickstream.filter(df_clickstream['event_date'] <= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date'] > treshhold)[['cookie', 'node', 'event']]

    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact') == 1)['event'].unique()
        )
    )
    df_eval = df_eval.filter(
        pl.col('cookie').is_in(df_train['cookie'].unique())
    ).filter(
        pl.col('node').is_in(df_train['node'].unique())
    )

    df_eval = df_eval.unique(['cookie', 'node'])

    return df_train, df_eval


def get_als_pred(users, nodes, user_to_pred,
                 als_alpha=1.0,
                 als_conf_coef=1.0,
                 als_data_prep='binary',
                 als_decay_rate=0.0,
                 als_factors=64,
                 als_iterations=15,
                 als_regularization=0.01,
                 data_frac=1.0):

    if data_frac < 1.0:
        sample_size = int(len(users) * data_frac)
        indices = np.random.choice(len(users), size=sample_size, replace=False)
        users = users[indices]
        nodes = nodes[indices]

    user_ids = users.unique().to_list()
    item_ids = nodes.unique().to_list()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item_id = {v:k for k,v in item_id_to_index.items()}

    rows = [user_id_to_index[user] for user in users]
    cols = [item_id_to_index[item] for item in nodes]

    if als_data_prep == "binary_wtime":
        values = [1.0 + als_decay_rate * i for i in range(len(rows))]
    else:
        values = [1.0] * len(rows)

    values = [v * als_alpha for v in values]

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    model = implicit.als.AlternatingLeastSquares(
        iterations=als_iterations,
        factors=als_factors,
        regularization=als_regularization,
        random_state=42
    )
    model.fit(sparse_matrix)

    user4pred = np.array([user_id_to_index[user] for user in user_to_pred])

    recommendations, scores = model.recommend(
        user4pred,
        sparse_matrix[user4pred],
        N=40,
        filter_already_liked_items=True
    )

    df_pred = pl.DataFrame({
        'node': [[index_to_item_id[i] for i in rec] for rec in recommendations.tolist()],
        'cookie': list(user_to_pred),
        'scores': scores.tolist()
    })
    df_pred = df_pred.explode(['node', 'scores'])

    return df_pred


def recall_at(df_true, df_pred, k=40):
    return df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']],
        how='left',
        on=['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum() / pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()


def main():
    parser = argparse.ArgumentParser(description="Train ALS model with parameters")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--model_type", type=str, default="als", help="Model to use")
    parser.add_argument("--als_factors", type=int, default=64, help="ALS factors")
    parser.add_argument("--als_iterations", type=int, default=15, help="ALS iterations")
    parser.add_argument("--als_regularization", type=float, default=0.01, help="ALS reg")
    parser.add_argument("--als_alpha", type=float, default=1.0, help="ALS alpha")
    parser.add_argument("--als_data_frac", type=float, default=1.0, help="Data fraction")
    parser.add_argument("--als_data_prep", type=str, default="binary", help="Data prep method")
    parser.add_argument("--als_decay_rate", type=float, default=0.0, help="Decay rate for time-weighted binary")

    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location='mlflow-artifacts:/')
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name):
        df_test_users, df_clickstream, df_event = get_data()
        df_train, df_eval = split_train_test(df_clickstream, df_event)
        df_pred = get_als_pred(
            df_train["cookie"], df_train["node"], df_eval['cookie'].unique().to_list(),
            als_alpha=args.als_alpha,
            als_factors=args.als_factors,
            als_iterations=args.als_iterations,
            als_regularization=args.als_regularization,
            data_frac=args.als_data_frac,
            als_data_prep=args.als_data_prep,
            als_decay_rate=args.als_decay_rate
        )

        metric = recall_at(df_eval, df_pred, k=40)

        mlflow.log_params(vars(args))
        mlflow.log_metric("Recall_40", metric)

if __name__ == "__main__":
    main()