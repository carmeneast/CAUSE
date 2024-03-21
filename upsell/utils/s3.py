import boto3
import torch
import numpy as np
import pandas as pd
from io import BytesIO, StringIO


def pd_read_s3_multiple_files(bucket, key, file_suffix='.csv', verbose=False):
    if not key.endswith('/'):
        key = key + '/'  # Add '/' to the end

    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    s3_keys = [item.key for item in s3_resource.Bucket(bucket).objects.filter(Prefix=key)
               if item.key.endswith(file_suffix)]
    if not s3_keys:
        print(f'No {file_suffix} files found in', bucket, key)
    elif verbose:
        print(f'Load {file_suffix} files:')
        for p in s3_keys:
            print(p)
    else:
        print(f'Found {len(s3_keys)} files')

    dfs = []
    for i, key in enumerate(s3_keys):
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        if file_suffix.endswith('.csv'):
            file = pd.read_csv(BytesIO(obj['Body'].read()))
        elif file_suffix == '.json':
            file = pd.read_json(BytesIO(obj['Body'].read()), lines=True)
        elif file_suffix == '.parquet':
            file = pd.read_parquet(BytesIO(obj['Body'].read()))
        else:
            raise f'suffix {file_suffix} not supported'
        dfs.append(file)

    final = pd.concat(dfs, ignore_index=True)
    return final


def s3_key(tenant_id, run_date=None, model_id=None):
    key = f'opportunity_scoring/{tenant_id}/{model_id}/'
    if run_date:
        key += f'{run_date}/'
    return key


def load_event_type_names(bucket, tenant_id, model_id, run_date):
    key = s3_key(tenant_id, None, model_id)
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key+'training/preprocess/event_names.csv')
    file = pd.read_csv(BytesIO(obj['Body'].read()))
    return file


def get_batch_idxs(bucket, tenant_id, model_id, run_date):
    key = s3_key(tenant_id, run_date, model_id) + 'scoring/prep/'
    s3_keys = [item.key for item in boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=key)
               if item.key.endswith('.csv')]
    return range(len(s3_keys))


def load_numpy_data(bucket, tenant_id, run_date, model_id, dataset='train', batch_idx=None):
    # Load event seqs
    if dataset in ['train', 'test']:
        key = s3_key(tenant_id, None, model_id)
        filename = f'training/preprocess/03_{dataset}_arrays.np'
    else:
        key = s3_key(tenant_id, run_date, model_id)
        filename = f'scoring/preprocess/03_{dataset}_arrays_{batch_idx:04}.np'
    print(f'Loading s3://{bucket}/{key}{filename}')
    with BytesIO() as obj:
        boto3.resource('s3').Bucket(bucket).download_fileobj(key+filename, obj)
        obj.seek(0)
        event_seqs = np.load(obj, allow_pickle=True)
        print('event_seqs', len(event_seqs), event_seqs[0].shape[1])

    # Load account ids
    if dataset in ['train', 'test']:
        filename = f'training/preprocess/02_{dataset}_transformed.csv'
    else:
        filename = f'scoring/preprocess/02_{dataset}_transformed_{batch_idx:04}.csv'
    print(f'Loading s3://{bucket}/{key}{filename}')
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key+filename)
    account_ids = pd.read_csv(BytesIO(obj['Body'].read()), sep='\t', escapechar='\\', encoding='utf-8',
                              doublequote=True)
    account_ids = account_ids[['tenant_id', 'account_id']].drop_duplicates()\
        .sort_values('account_id')\
        .reset_index(drop=True)
    account_ids['tenant_id'] = account_ids['tenant_id'].astype(int)
    account_ids['account_id'] = account_ids['account_id'].astype(int)
    print('account_ids', account_ids.shape)

    return {
        'event_seqs': event_seqs,
        'account_ids': account_ids,
    }


def save_pytorch_dataset(dataset, bucket, tenant_id, run_date, model_id, name):
    key = s3_key(tenant_id, run_date, model_id)
    s3 = boto3.client('s3')
    buffer = BytesIO()
    torch.save(dataset.cpu().numpy(), buffer, pickle_protocol=4)
    s3.put_object(Bucket=bucket, Key=key+name+'.pt', Body=buffer.getvalue())


def save_pytorch_model(model, bucket, tenant_id, run_date, model_id=1):
    key = s3_key(tenant_id, None, model_id)
    s3 = boto3.client('s3')
    buffer = BytesIO()
    torch.save(model, buffer)
    s3.put_object(Bucket=bucket, Key=key+'model/rnn.pt', Body=buffer.getvalue())


def load_pytorch_model(bucket, tenant_id, run_date, model_id):
    key = s3_key(tenant_id, None, model_id)
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(key+'model/rnn.pt', data)
        data.seek(0)
        obj = torch.load(data)
    return obj


def load_pytorch_data(bucket, key):
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(key, data)
        data.seek(0)
        obj = torch.load(data)
    return obj


def save_attributions(df, bucket, tenant_id, run_date, model_id=1):
    key = s3_key(tenant_id, run_date, model_id)
    s3_resource = boto3.resource('s3')
    buffer = StringIO()
    df.to_csv(buffer, index=False, escapechar='\\')
    s3_resource.Object(bucket, key+'attributions.csv').put(Body=buffer.getvalue())

