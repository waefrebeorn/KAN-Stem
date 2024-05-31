import asyncio
from fastapi.responses import JSONResponse
import gr

async def queue_join_helper(body, request, username):
    # Log the request details
    log_request_details(body)
    if not 0 <= body.fn_index < len(blocks.block_fns):  # Check if function index is valid
        return JSONResponse({"msg": "error", "details": "Invalid function index"}, status_code=400)
    
    success, event_id = await blocks._queue.push(
        body.fn_index,
        username,
        body.session_hash,
        body.data,
        request.client.host,
        request.headers,
    )
    if success:
        return JSONResponse({"event_id": event_id, "msg": "success"})
    return JSONResponse({"msg": "error"}, status_code=500)