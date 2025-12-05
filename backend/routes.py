"""API route handlers for live video tracking."""

import io
import os
import base64
import tempfile
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from schemas import DetectionResponse, HealthResponse
from detector import detector
from video_processor import video_processor

# YouTube download support
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("WARNING: yt-dlp not installed. YouTube download disabled.")

# FFmpeg support (auto-installed via imageio-ffmpeg)
FFMPEG_PATH = None
try:
    import imageio_ffmpeg
    try:
        FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"FFmpeg found: {FFMPEG_PATH}")
    except Exception as e:
        print(f"WARNING: ffmpeg not found via imageio-ffmpeg ({e}). Continuing without bundled ffmpeg.")
except ImportError:
    print("WARNING: imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")


class YouTubeRequest(BaseModel):
    url: str

class ModelSelectRequest(BaseModel):
    model_id: str


router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        message="Sentry Detection API"
    )


@router.get("/classes")
async def get_classes():
    """Get available model classes."""
    try:
        classes = detector.get_classes()
        return {"classes": classes}
    except Exception as e:
        return {"error": str(e)}


@router.get("/models")
async def get_models():
    """Get list of available models."""
    return {"models": video_processor.get_available_models()}


@router.get("/device")
async def get_device():
    """Get current processing device info."""
    return video_processor.get_device_info()


@router.post("/models/select")
async def select_model(request: ModelSelectRequest):
    """Switch to a different model."""
    success = video_processor.load_model(request.model_id)
    if success:
        return {
            "status": "ok",
            "model_id": request.model_id,
            "models": video_processor.get_available_models()
        }
    else:
        return {"error": f"Failed to load model: {request.model_id}"}


@router.post("/detect", response_model=DetectionResponse)
async def detect(frame: UploadFile = File(...)):
    """Detect objects in a single image."""
    try:
        contents = await frame.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        detections = detector.detect(image)
        return DetectionResponse(detections=detections)
    except Exception as e:
        print(f"ERROR: {e}")
        return DetectionResponse(detections=[])


@router.post("/detect-annotated")
async def detect_annotated(frame: UploadFile = File(...)):
    """Detect objects and return annotated image."""
    try:
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process with tracking
        annotated_frame, detections = video_processor.process_frame(frame_bgr)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "detections": detections,
            "annotated_frame": frame_base64
        }
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"detections": [], "annotated_frame": None}


@router.post("/track/reset")
async def reset_tracker():
    """Reset the tracker state for a new video."""
    video_processor.reset_tracker()
    return {"status": "ok", "message": "Tracker reset"}


@router.websocket("/ws/track")
async def websocket_track(websocket: WebSocket):
    """
    WebSocket endpoint for live video tracking.
    """
    await websocket.accept()
    print("=" * 50)
    print("Live tracking WebSocket connected")
    print("=" * 50)
    
    # Reset tracker for new connection
    video_processor.reset_tracker()
    
    frame_count = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            frame_count += 1
            print(f"\n--- Frame {frame_count} received ({len(data)} bytes) ---")
            
            try:
                # Process frame with tracking
                annotated_base64, detections = video_processor.process_frame_base64(data)
                
                if annotated_base64:
                    print(f"Sending annotated frame with {len(detections)} detections")
                    await websocket.send_json({
                        "annotated_frame": annotated_base64,
                        "detections": detections
                    })
                else:
                    print("ERROR: No annotated frame returned")
                    await websocket.send_json({"error": "Invalid frame"})
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": str(e)})
    
    except WebSocketDisconnect:
        print("Live tracking WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


# Store downloaded YouTube videos and processed videos
youtube_videos = {}
processed_videos = {}
processing_progress = {}

@router.post("/youtube/prepare")
async def prepare_youtube(request: YouTubeRequest):
    """
    Download YouTube video and serve locally (bypasses CORS).
    """
    if not YT_DLP_AVAILABLE:
        return {"error": "yt-dlp not installed. Run: pip install yt-dlp"}
    
    url = request.url
    print(f"Downloading YouTube video: {url}")
    
    try:
        # Create temp file for download
        temp_dir = tempfile.mkdtemp(prefix="yt_")
        output_template = os.path.join(temp_dir, "video.mp4")
        
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': False,
        }
        
        # Use bundled ffmpeg if available
        if FFMPEG_PATH:
            ydl_opts['ffmpeg_location'] = FFMPEG_PATH
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
            ydl_opts['prefer_ffmpeg'] = True
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'video')
            video_id = info.get('id', str(hash(url)))
        
        # Find the downloaded file
        video_path = output_template
        if not os.path.exists(video_path):
            # Try to find any mp4 in temp dir
            for f in os.listdir(temp_dir):
                if f.endswith('.mp4'):
                    video_path = os.path.join(temp_dir, f)
                    break
        
        if os.path.exists(video_path):
            youtube_videos[video_id] = {
                'path': video_path,
                'title': title,
                'temp_dir': temp_dir
            }
            
            print(f"Downloaded: {title} -> {video_path}")
            
            return {
                "status": "ok",
                "title": title,
                "video_id": video_id,
                "proxy_url": f"/youtube/video/{video_id}"
            }
        else:
            return {"error": "Download failed"}
    
    except Exception as e:
        print(f"YouTube download error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@router.get("/youtube/video/{video_id}")
async def serve_youtube_video(video_id: str):
    """
    Serve downloaded YouTube video with CORS headers for canvas capture.
    """
    if video_id not in youtube_videos:
        return {"error": "Video not found"}
    
    video_info = youtube_videos[video_id]
    video_path = video_info['path']
    
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_info['title']}.mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


def run_video_analysis(proc_id: str, input_path: str, output_path: str, title: str):
    """Background task to process video."""
    def progress_cb(done: int, total: int, stage: str = "analyzing"):
        processing_progress[proc_id]["processed"] = done
        processing_progress[proc_id]["stage"] = stage
        if total:
            processing_progress[proc_id]["total"] = total
    
    try:
        print(f"Analyzing video: {title}")
        processing_progress[proc_id]["stage"] = "analyzing"
        meta = video_processor.process_video_file(input_path, output_path, progress_cb=progress_cb)
        processed_videos[proc_id] = {
            "path": output_path,
            "title": f"{title}_analyzed",
            "meta": meta,
        }
        processing_progress[proc_id]["status"] = "done"
        processing_progress[proc_id]["processed"] = meta.get("frame_count", 0)
        processing_progress[proc_id]["total"] = meta.get("total_frames", 0)
        processing_progress[proc_id]["stage"] = "complete"
        processing_progress[proc_id]["detection_counts"] = meta.get("detection_counts", {})
        print(f"Video analysis complete: {meta['frame_count']} frames")
        print(f"Detection counts: {meta.get('detection_counts', {})}")
    except Exception as e:
        print(f"Video analysis error: {e}")
        import traceback
        traceback.print_exc()
        processing_progress[proc_id]["status"] = "error"
        processing_progress[proc_id]["error"] = str(e)
        processing_progress[proc_id]["stage"] = "error"


@router.post("/video/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    video_id: str = Form(None)
):
    """
    Analyze a full video and return a processed video URL with bounding boxes.
    - If video_id is provided, use a previously downloaded YouTube video.
    - Otherwise, accept an uploaded file.
    Returns immediately and processes in background.
    """
    input_path = None
    title = "video"
    
    # Use existing YouTube download
    if video_id:
        if video_id not in youtube_videos:
            return {"error": "Video not found"}
        info = youtube_videos[video_id]
        input_path = info['path']
        title = info.get('title', 'video')
    elif file is not None:
        # Save uploaded file to temp path
        tmp_dir = tempfile.mkdtemp(prefix="upload_")
        input_path = os.path.join(tmp_dir, file.filename or "video.mp4")
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
        title = file.filename or "video"
    else:
        return {"error": "No video provided"}
    
    if not os.path.exists(input_path):
        return {"error": "Input video not found"}
    
    # Prepare output path
    proc_id = str(uuid.uuid4())
    out_dir = tempfile.mkdtemp(prefix="analyzed_")
    output_path = os.path.join(out_dir, "analyzed.mp4")
    
    # Initialize progress
    processing_progress[proc_id] = {
        "status": "processing",
        "processed": 0,
        "total": None,
        "error": None,
    }
    
    # Start background processing
    background_tasks.add_task(run_video_analysis, proc_id, input_path, output_path, title)
    
    # Return immediately with the process ID
    return {
        "status": "processing",
        "processed_id": proc_id,
        "processed_url": f"/video/analyzed/{proc_id}",
    }


@router.get("/video/analyzed/{proc_id}")
async def serve_analyzed_video(proc_id: str):
    """Serve analyzed video with CORS headers."""
    if proc_id not in processed_videos:
        return {"error": "Analyzed video not found"}
    
    info = processed_videos[proc_id]
    path = info['path']
    if not os.path.exists(path):
        return {"error": "Analyzed video file missing"}
    
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{info.get('title', 'analyzed')}.mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@router.get("/video/analyze/status/{proc_id}")
async def analyze_video_status(proc_id: str):
    """Get analysis progress for a video."""
    if proc_id not in processing_progress:
        return {"error": "Not found"}
    return processing_progress[proc_id]

