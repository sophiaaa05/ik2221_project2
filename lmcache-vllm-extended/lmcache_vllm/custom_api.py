import vllm.entrypoints.openai.api_server as base_api
from vllm.entrypoints.openai.protocol import *
from fastapi import APIRouter, Request

# You should use the following file to implement all APIs you may require in the project.
# Note that the two important ones are already implemented here simply by calling the default v1 implementation in VLLM.
# You may need to modify these functions to enable pre-processing of requests, before running the inference.

extended_router = APIRouter()

@extended_router.get("/models")
async def show_available_models(request: Request):
    print("v2 models is called!")
    return await base_api.show_available_models(request)


@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    print("v2 completion is called")
    return await base_api.create_chat_completion(request, raw_request)