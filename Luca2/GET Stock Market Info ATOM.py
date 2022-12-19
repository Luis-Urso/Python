## From: https://docs.atom.finance/reference/news-1
## API KEY: 3a26d706-883e-4af1-8399-8fece12a13ab

## by Luis Urso - 8-Set-2022


import requests

url = "https://platform.atom.finance/api/2.0/news?api_key=3a26d706-883e-4af1-8399-8fece12a13ab"

payload = {
    "markets": ["BRA"],
    "languages": ["en-US"],
    "page": 0
}
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
