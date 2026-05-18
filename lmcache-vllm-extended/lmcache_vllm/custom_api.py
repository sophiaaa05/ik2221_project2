import time

import vllm.entrypoints.openai.api_server as base_api
from vllm.entrypoints.openai.protocol import *
from fastapi import APIRouter, Request
import json

# You should use the following file to implement all APIs you may require in the project.
# Note that the two important ones are already implemented here simply by calling the default v1 implementation in VLLM.
# You may need to modify these functions to enable pre-processing of requests, before running the inference.

extended_router = APIRouter()
STREAM_END = "data: [DONE]"

class BatchedRequest(ChatCompletionRequest):
    """Used for a single request in the batch so it's possible to sort them"""
    request_id: int = -1
    context_id: str = ""

@extended_router.get("/models")
async def show_available_models(request: Request):
    print("v2 models is called!")
    return await base_api.show_available_models(request)

@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    print("v2 completion is called")
    return await base_api.create_chat_completion(request, raw_request)

@extended_router.post("/chat/batched-completions")
async def create_chat_batched_completions(requests: List[BatchedRequest], raw_request: Request):
    # Sorting the requests based on the context_id
    requests.sort(key=lambda req: req.context_id)

    print("create_chat_completion - Batch order after sorting:")
    print([req.context_id for req in requests])

    # Collecting the results together 
    results = []
    for req in requests:
        print(f"create_chat_completion - Processing request with context_id: {req.context_id}")
        start = time.perf_counter()
        req.stream = True
        stream = await base_api.create_chat_completion(req, raw_request)

        chunks: list[str] = []
        ttft = None

        # Using stream since we need to measure the TTFT 
        async for raw_stream_chunk in stream.body_iterator:
            # Skip empty stream chunks and stop reading if end of strem
            if not raw_stream_chunk.strip():
                continue            
            if STREAM_END in raw_stream_chunk:
                break
      
            data_str = raw_stream_chunk.removeprefix("data: ").strip()
            try:
                chunk_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not chunk_data.get("choices"):
                continue

            # Get the token and update the ttft if it's the first token
            content = chunk_data["choices"][0]["delta"].get("content", "")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks.append(content)

        full_response = "".join(chunks)
        latency = time.perf_counter() - start

        results.append({"request_id": req.request_id, "response": full_response, "latency": latency, "ttft": ttft}) 

    return results