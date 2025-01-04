import requests

bear_url = "https://cdn.britannica.com/41/156441-050-A4424AEC/Grizzly-bear-Jasper-National-Park-Canada-Alberta.jpg"  # noqa
resp = requests.post("http://localhost:8000/classify", json={"image_url": bear_url, "should_translate": True})

print(resp.text)
# 'brown bear, bruin, Ursus arctos'

text = "Hello, the weather is quite fine today!"
resp = requests.post("http://localhost:8000/translate", json={"text": text})

print(resp.text)
# 'Hallo, das Wetter ist heute ziemlich gut!'