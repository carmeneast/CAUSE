import boto3
import numpy as np
import tempfile
from scipy import sparse
from datetime import datetime

from upsell.utils.s3 import pd_read_s3_multiple_files, s3_key


def convert_to_series_of_sparse_matrices(event_df):
    print(datetime.now())
    event_df['sparse'] = event_df['event_sparse_vector'].apply(lambda x: sparse.coo_matrix(
        (x['values'], ([0]*len(x['indices']), x['indices'])), shape=(1, x['size'])
    ))
    print('Vectors.sparse to scipy.sparse', event_df.shape)
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


def create_save_event_seqs(bucket, tenant_id, run_date, model_id=None, dataset='train'):
    key = s3_key(tenant_id, run_date, model_id)
    events = pd_read_s3_multiple_files(bucket, key+f'{dataset}_sparse_matrices/', '.parquet')
    print('events', events.shape)

    accounts, event_seqs = convert_to_series_of_sparse_matrices(events)
    print('accounts', accounts.shape)
    print('event_seqs', len(event_seqs))

    filename = f'{dataset}_model_data.npz'
    with tempfile.TemporaryFile() as outfile:
        np.savez(
            outfile,
            event_seqs=event_seqs,
            account_ids=accounts,
        )
        outfile.seek(0)
        boto3.Session().resource('s3')\
            .Bucket(bucket)\
            .Object(key+filename)\
            .upload_fileobj(outfile)
    print(f'saved {filename}')
