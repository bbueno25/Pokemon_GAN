import datetime as dt
import google.cloud.storage as storage
import google.oauth2.service_account as service_account
import googleapiclient.discovery as discovery
import googleapiclient.errors as errors
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

def create_bucket(bucket_name, storage_client):
    logging.info('bucket:creating')
    if storage_client.lookup_bucket(bucket_name):
        logging.info('bucket:already exists')
    else:
        storage_client.create_bucket(bucket_name)
        logging.info('bucket:created')
    return storage_client.bucket(bucket_name)

def create_job(bucket_name, json_path, package_name, project_id):
    logging.info('job:creating')
    credentials = service_account.Credentials.from_service_account_file(json_path)
    ml = discovery.build('ml', 'v1', credentials=credentials)
    project_path = 'projects/{}'.format(project_id)
    training_input = {'jobDir': 'gs://{}/model'.format(bucket_name),
                      'packageUris': ['gs://{}/{}'.format(bucket_name, package_name)],
                      'pythonModule': 'trainer.task',
                      'region': 'us-central1',
                      'runtimeVersion': '1.6',
                      'scaleTier': 'BASIC'}
    now = dt.datetime.now().strftime("%y%m%d_%H%M%S")
    job_specs = {'jobId': 'pokegan_' + now, 'trainingInput': training_input}
    request = ml.projects().jobs().create(parent=project_path, body=job_specs)    # pylint:disable=E1101
    try:
        request.execute()
        logging.info('job:created')
    except errors.HttpError as err:
        logging.error('job:{}'.format(err._get_reason()))

def delete_bucket(json_path, project_id):
    logging.info('bucket:deleting')
    credentials = service_account.Credentials.from_service_account_file(json_path)
    storage_client = storage.Client(project_id, credentials)
    for bucket in storage_client.list_buckets():
        storage_client.delete(bucket, force=True)
    logging.info('bucket:deleted')

def upload_data(bucket_name, json_path, project_id, src_dir):
    credentials = service_account.Credentials.from_service_account_file(json_path)
    storage_client = storage.Client(project_id, credentials)
    bucket = create_bucket(bucket_name, storage_client)
    logging.info('training data:uploading')
    for filename in os.listdir(src_dir):
        logging.info('training data:uploading:{}'.format(filename))
        blob = storage.Blob('data/' + filename, bucket)
        blob.upload_from_filename(os.path.join(src_dir, filename))
    logging.info('training data:uploaded')

def upload_package(bucket_name, json_path, package_name, project_id):
    credentials = service_account.Credentials.from_service_account_file(json_path)
    storage_client = storage.Client(project_id, credentials)
    bucket = create_bucket(bucket_name, storage_client)
    logging.info('package:uploading')
    blob = storage.Blob(package_name, bucket)
    blob.upload_from_filename(os.path.join('dist', package_name))
    logging.info('package:uploaded')
