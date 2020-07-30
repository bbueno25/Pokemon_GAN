"""

Upload training data to Google Cloud.

"""
import gcloud_utils as utils

BUCKET_NAME = 'pokegan-data'
JSON_PATH = 'My Project 58923-14efe3d8f5f7.json'
PROJECT_ID = '657623122962'
SRC_DIR = 'data/rgb_images'

if __name__ == '__main__':
    utils.upload_data(BUCKET_NAME, JSON_PATH, PROJECT_ID, SRC_DIR)
