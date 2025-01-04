# https://docs.ray.io/en/latest/serve/multi-app.html

import starlette

from transformers import pipeline

from ray import serve


@serve.deployment(route_prefix='/translate')
class Translator:
    def __init__(self):
        self.model = pipeline("translation_en_to_de", model="t5-small")

    def translate(self, text: str) -> str:
        return self.model(text)[0]["translation_text"]

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return self.translate(req["text"])
    


app = Translator.bind()