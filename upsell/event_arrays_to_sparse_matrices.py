import boto3
import numpy as np
import tempfile
from scipy import sparse
from datetime import datetime
from sklearn.model_selection import KFold

from upsell.s3 import pd_read_s3_multiple_files, s3_key


def convert_to_series_of_sparse_matrices(event_df):
    print(datetime.now())
    event_df['sparse'] = event_df['eventArray'].apply(lambda x: sparse.coo_matrix((x['values'], ([0]*len(x['indices']), x['indices'])), shape=(1, x['size'])))
    print('eventArray to sparse', event_df.shape)
    print(datetime.now())
    events_by_account = event_df.sort_values(['tenant_id', 'account_id', 'dt'])\
        .groupby(['tenant_id', 'account_id'])['sparse']\
        .apply(list)\
        .reset_index()
    print('group by account', events_by_account.shape)
    print(datetime.now())
    event_seqs = events_by_account['sparse'].apply(lambda x: sparse.vstack(x))
    print('vstack', len(event_seqs))
    print(datetime.now())
    return events_by_account[['tenant_id', 'account_id']], event_seqs


def create_save_event_seqs(bucket, tenant_id, run_date, sampling, n_splits=5, random_state=16, training=False):
    key = s3_key(tenant_id, run_date, sampling)
    path = 'sparseEventVectors/' if training else 'predSparseEventVectors/'
    events = pd_read_s3_multiple_files(bucket, key+path, '.parquet')
    print('events', events.shape)

    accounts, event_seqs = convert_to_series_of_sparse_matrices(events)
    print('accounts', accounts.shape)
    print('event_seqs', len(event_seqs))

    train_test_splits = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            range(len(event_seqs))
        )
    )
    print('train_test_splits', len(train_test_splits))

    filename = 'model_train_data.npz' if training else 'model_pred_data.npz'
    with tempfile.TemporaryFile() as outfile:
        np.savez(
            outfile,
            event_seqs=event_seqs,
            train_test_splits=train_test_splits,
            account_ids=accounts,
        )
        outfile.seek(0)
        boto3.Session().resource('s3')\
            .Bucket(bucket)\
            .Object(key+filename)\
            .upload_fileobj(outfile)
    print('saved model_data.npz')
