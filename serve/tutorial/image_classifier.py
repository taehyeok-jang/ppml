# https://docs.ray.io/en/latest/serve/multi-app.html

import requests
import starlette

from transformers import pipeline
from io import BytesIO
from PIL import Image

from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
def downloader(image_url: str):
    image_bytes = requests.get(image_url).content
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


@serve.deployment(route_prefix='/classify')
class ImageClassifier:
    def __init__(self, downloader: DeploymentHandle):
        self.downloader = downloader
        self.model = pipeline(
            "image-classification", model="google/vit-base-patch16-224"
        )

    async def classify(self, image_url: str) -> str:
        image = await self.downloader.remote(image_url)
        results = self.model(image)
        return results[0]["label"]

    ### default
    # async def __call__(self, req: starlette.requests.Request):
    #     req = await req.json()
    #     return await self.classify(req["image_url"])
    ### call between applications by using Ray Serve API
    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        print(req)

        result = await self.classify(req["image_url"])

        if req.get("should_translate") is True:
            handle: DeploymentHandle = serve.get_app_handle("app2")
            return await handle.translate.remote(result)

        return result


app = ImageClassifier.bind(downloader.bind())