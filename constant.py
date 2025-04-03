import os

import firebase_admin
from firebase_admin import credentials, db

BASE_DIR = os.getcwd()

DB_LOGS = "/tbl_logs"
DB_RECORDS = "/tbl_logging"
DB_SUBSCRIBER = "/tbl_subcribers/subscribers"
DB_PREDICTIONS = "/tbl_predictions"

# firebase certificates
FIREBASE_CERTIFICATE = os.path.sep.join([BASE_DIR, 'air_q.json'])
FIREBASE_DB_URL = 'https://air-quality-monitoring-8-gases-default-rtdb.asia-southeast1.firebasedatabase.app/'


# Initialize Firebase app
cred = credentials.Certificate(FIREBASE_CERTIFICATE)
firebase_app = firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
