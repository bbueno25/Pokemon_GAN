"""

Upload distribution package to storage bucket on Google Cloud.
The distribution package is used to create a job on Google's ML Engine.

"""
import gcloud_utils as utils

BUCKET_NAME = 'pokegan-data'
JSON_PATH = 'My Project 58923-14efe3d8f5f7.json'
PACKAGE_NAME = 'pokegan-0.0.0.tar.gz'
PROJECT_ID = '657623122962'

if __name__ == '__main__':
    utils.upload_package(BUCKET_NAME, JSON_PATH, PACKAGE_NAME, PROJECT_ID)
