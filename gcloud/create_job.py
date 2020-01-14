"""

Create a Google ML Engine Job using Python API.

"""
import gcloud_utils as utils

BUCKET_NAME = 'pokegan-data'
JSON_PATH = 'My Project 58923-14efe3d8f5f7.json'
PACKAGE_NAME = 'pokegan-0.0.0.tar.gz'
PROJECT_ID = 'ivory-hallway-204216'

if __name__ == '__main__':
    utils.create_job(BUCKET_NAME, JSON_PATH, PACKAGE_NAME, PROJECT_ID)
