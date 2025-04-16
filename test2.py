import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("air_q.json")

# IMPORTANT: Reuse app or check if already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://air-quality-monitoring-8-gases-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

ref = db.reference("/")  # or "tbl_logging"
print(ref.get())
