"""
EagleView AI - Detection Backend

Main entry point for the FastAPI application.
"""

import atexit
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT, CORS_ORIGINS
from routes import router
from detector import detector
from video_processor import video_processor


def cleanup_gpu():
    """Clear GPU memory allocation on shutdown."""
    # Cleanup model instances first
    try:
        video_processor.cleanup()
    except Exception as e:
        print(f"[WARNING] VideoProcessor cleanup error: {e}")
    
    try:
        detector.cleanup()
    except Exception as e:
        print(f"[WARNING] Detector cleanup error: {e}")
    
    # Final GPU cache clear
    try:
        import torch
        if torch.cuda.is_available():
            print("[*] Final GPU cache clear...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Print memory stats
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"[OK] GPU memory - allocated: {allocated:.1f} MB, reserved: {reserved:.1f} MB")
    except Exception as e:
        print(f"[WARNING] GPU cleanup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    yield
    # Shutdown
    print("\n" + "=" * 50)
    print("  Shutting down...")
    print("=" * 50)
    cleanup_gpu()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="EagleView AI",
        description="Aerial surveillance detection API for soldier/civilian classification",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    # Also register atexit handler for non-graceful shutdowns
    atexit.register(cleanup_gpu)
    
    return app


# Create the app instance
app = create_app()


def main():
    """Run the server."""
    import uvicorn
    
    print("=" * 50)
    print("  EagleView AI - Detection Backend")
    print("=" * 50)
    print()
    
    # Pre-load the model to verify it works
    if not detector.load():
        print("\nERROR: Failed to load model")
        print("\nPlease place your model file in one of these locations:")
        print("  - sentry/src/assets/model/yolo11s.pt")
        print("  - sentry/backend/yolo11s.pt")
        import sys
        sys.exit(1)
    
    print()
    print(f"Starting server at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
