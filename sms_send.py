import requests
import json

url = 'https://sms.iprogtech.com/api/v1/sms_messages/send_bulk'

def send_sms(numbers:list, message:str):
    numbers = ','.join(numbers)

    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        'api_token': '5c44caff2b6bc10f34b394002a8475ade5e01cc4',
        'phone_number': numbers,
        'message': message
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
