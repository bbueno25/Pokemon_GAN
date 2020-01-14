import gcloud_utils as utils

JSON_PATH = 'My Project 58923-14efe3d8f5f7.json'
PROJECT_ID = '657623122962'

if __name__ == '__main__':
    utils.delete_bucket(JSON_PATH, PROJECT_ID)
