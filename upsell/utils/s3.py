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
        if file_suffix == '.csv':
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


def s3_key(tenant_id, run_date, model_id=1):
    return f'upsell/{tenant_id}/{model_id}/{run_date}/'


def load_event_type_names(bucket, tenant_id, model_id, run_date):
    key = s3_key(tenant_id, run_date, model_id)
    return pd_read_s3_multiple_files(bucket, key+'event_type_names/', '.parquet')


def load_numpy_data(bucket, tenant_id, run_date, model_id, dataset='train'):
    key = s3_key(tenant_id, run_date, model_id)
    filename = f'{dataset}_model_data.npz'
    print(f'Loading s3://{bucket}/{key}{filename}')
    with BytesIO() as obj:
        boto3.resource('s3').Bucket(bucket).download_fileobj(key+filename, obj)
        obj.seek(0)
        data = np.load(obj, allow_pickle=True)

        event_seqs = data['event_seqs']
        print('event_seqs', len(event_seqs), event_seqs[0].shape[1])
        account_ids = data['account_ids']
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
    key = s3_key(tenant_id, run_date, model_id)
    s3 = boto3.client('s3')
    buffer = BytesIO()
    torch.save(model, buffer)
    s3.put_object(Bucket=bucket, Key=key+'model.pt', Body=buffer.getvalue())


def load_pytorch_object(bucket, tenant_id, run_date, model_id, name):
    key = s3_key(tenant_id, run_date, model_id)
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(key+name+'.pt', data)
        data.seek(0)
        obj = torch.load(data)
    return obj


def save_attributions(df, bucket, tenant_id, run_date, model_id=1):
    key = s3_key(tenant_id, run_date, model_id)
    s3_resource = boto3.resource('s3')
    buffer = StringIO()
    df.to_csv(buffer, index=False, escapechar='\\')
    s3_resource.Object(bucket, key+'attributions.csv').put(Body=buffer.getvalue())

