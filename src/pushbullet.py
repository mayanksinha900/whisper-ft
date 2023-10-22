import json
import requests

def pushbullet_notifier(title, body, api_key):
    data = {"type": "note", "title": title, "body": body}
    response = requests.post(
        "https://api.pushbullet.com/v2/pushes",
        data=json.dumps(data),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )

    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print("Failed to send notification.")
        print(response.status_code)
