"""
EagleView AI - Detection Backend

Main entry point for the FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT, CORS_ORIGINS
from routes import router
from detector import detector


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="EagleView AI",
        description="Aerial surveillance detection API for soldier/civilian classification",
        version="1.0.0",
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
        print("  - eyepatch/src/assets/model/yolo11s.pt")
        print("  - eyepatch/backend/yolo11s.pt")
        import sys
        sys.exit(1)
    
    print()
    print(f"Starting server at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
