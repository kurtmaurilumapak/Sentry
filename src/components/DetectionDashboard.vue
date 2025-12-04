<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';

const BACKEND_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/track';

const file = ref(null);
const previewUrl = ref('');
const isProcessing = ref(false);
const loadingMessage = ref('');
const detections = ref([]);
const errorMsg = ref('');

const backendStatus = ref('checking');

const videoRef = ref(null);

// Tracking state
const isTracking = ref(false);
const annotatedFrameUrl = ref('');
let websocket = null;
let trackingInterval = null;

const isVideo = computed(() => file.value?.type?.startsWith('video/') || isYouTubeVideo.value);
const isImage = computed(() => file.value?.type?.startsWith('image/'));
const hasMedia = computed(() => !!previewUrl.value);

// YouTube support
const youtubeUrl = ref('');
const isYouTubeVideo = ref(false);
const isDownloading = ref(false);

// Model selection
const availableModels = ref([]);
const selectedModelId = ref('');
const isChangingModel = ref(false);

// Device info
const deviceInfo = ref({ name: 'Detecting...', has_gpu: false });

async function loadDeviceInfo() {
  try {
    const response = await fetch(`${BACKEND_BASE}/device`);
    const data = await response.json();
    deviceInfo.value = data;
  } catch (e) {
    console.error('Failed to load device info:', e);
  }
}

async function loadAvailableModels() {
  try {
    const response = await fetch(`${BACKEND_BASE}/models`);
    const data = await response.json();
    if (data.models) {
      availableModels.value = data.models;
      const active = data.models.find(m => m.active);
      if (active) selectedModelId.value = active.id;
    }
  } catch (e) {
    console.error('Failed to load models:', e);
  }
}

async function changeModel(modelId) {
  if (modelId === selectedModelId.value) return;
  if (isChangingModel.value) return;
  
  isChangingModel.value = true;
  loadingMessage.value = 'Switching model...';
  
  try {
    const response = await fetch(`${BACKEND_BASE}/models/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId })
    });
    
    const data = await response.json();
    if (data.status === 'ok') {
      selectedModelId.value = modelId;
      availableModels.value = data.models;
      // Reset tracker for new model
      await resetTracker();
      isWarmedUp = false;
    } else {
      errorMsg.value = data.error || 'Failed to switch model';
    }
  } catch (e) {
    errorMsg.value = 'Failed to switch model';
  } finally {
    isChangingModel.value = false;
    loadingMessage.value = '';
  }
}

async function checkBackendStatus() {
  backendStatus.value = 'checking';
  try {
    const response = await fetch(`${BACKEND_BASE}/`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    if (response.ok) {
      backendStatus.value = 'online';
      // Load device info and models when backend comes online
      loadDeviceInfo();
      loadAvailableModels();
    } else {
      backendStatus.value = 'offline';
    }
  } catch (err) {
    backendStatus.value = 'offline';
  }
}

function clearMedia() {
  stopTracking(true); // Close WebSocket for new video
  isWarmedUp = false;
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  file.value = null;
  previewUrl.value = '';
  detections.value = [];
  errorMsg.value = '';
  annotatedFrameUrl.value = '';
}

async function resetTracker() {
  try {
    await fetch(`${BACKEND_BASE}/track/reset`, { method: 'POST' });
    console.log('Tracker reset');
  } catch (e) {
    console.error('Reset failed:', e);
  }
}

// WebSocket connection
function connectWebSocket() {
  if (websocket?.readyState === WebSocket.OPEN) return;
  
  console.log('Connecting WebSocket...');
  websocket = new WebSocket(WS_URL);
  
  websocket.onopen = () => {
    console.log('WebSocket connected');
  };
  
  websocket.onmessage = (event) => {
    pendingFrames = Math.max(0, pendingFrames - 1);
    try {
      const data = JSON.parse(event.data);
      if (data.annotated_frame) {
        annotatedFrameUrl.value = `data:image/jpeg;base64,${data.annotated_frame}`;
      }
      if (data.detections) {
        detections.value = data.detections;
      }
    } catch (e) {
      console.error('Message parse error:', e);
    }
  };
  
  websocket.onerror = (e) => {
    console.error('WebSocket error:', e);
  };
  
  websocket.onclose = () => {
    console.log('WebSocket closed');
    websocket = null;
  };
}

// Reusable canvas for performance
let captureCanvas = null;
let captureCtx = null;
let pendingFrames = 0;
const MAX_PENDING = 3; // Allow frame buffering for 60 FPS smoothness

// Capture frame and send to backend - FULL RESOLUTION
function captureAndSend() {
  const video = videoRef.value;
  if (!video || video.paused || video.ended) return;
  if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
  if (video.videoWidth === 0 || video.videoHeight === 0) return;
  
  // Skip if too many pending frames (prevents backlog)
  if (pendingFrames >= MAX_PENDING) return;
  
  try {
    const width = video.videoWidth;
    const height = video.videoHeight;
    
    // Reuse canvas at full resolution
    if (!captureCanvas || captureCanvas.width !== width || captureCanvas.height !== height) {
      captureCanvas = document.createElement('canvas');
      captureCanvas.width = width;
      captureCanvas.height = height;
      captureCtx = captureCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    }
    
    captureCtx.drawImage(video, 0, 0, width, height);
    
    // High quality JPEG for full resolution
    const base64 = captureCanvas.toDataURL('image/jpeg', 0.85).split(',')[1];
    if (base64 && base64.length > 100) {
      pendingFrames++;
      websocket.send(base64);
    }
  } catch (e) {
    console.error('Frame capture error:', e.message);
  }
}

// Start tracking
function startTracking() {
  if (isTracking.value) return;
  
  console.log('Starting tracking...');
  isTracking.value = true;
  annotatedFrameUrl.value = '';
  
  connectWebSocket();
  
  // Wait for WebSocket then start sending (16ms = ~60 FPS target, throttled by pending limit)
  setTimeout(() => {
    if (websocket?.readyState === WebSocket.OPEN) {
      console.log('Starting frame capture at 60 FPS...');
      trackingInterval = setInterval(captureAndSend, 16);
    } else {
      console.log('WebSocket not ready, retrying...');
      setTimeout(() => {
        trackingInterval = setInterval(captureAndSend, 16);
      }, 500);
    }
  }, 300);
}

// Stop tracking (keep WebSocket open for quick resume)
function stopTracking(closeSocket = false) {
  console.log('Stopping tracking...');
  isTracking.value = false;
  pendingFrames = 0;
  
  if (trackingInterval) {
    clearInterval(trackingInterval);
    trackingInterval = null;
  }
  
  // Only close WebSocket when explicitly requested (e.g., new video)
  if (closeSocket && websocket) {
    websocket.close();
    websocket = null;
    isWarmedUp = false;
  }
}

// Warm-up: pre-connect and send first frame before play
let isWarmedUp = false;

async function warmupModel() {
  const video = videoRef.value;
  if (!video || isWarmedUp) return;
  if (video.videoWidth === 0 || video.videoHeight === 0) return;
  
  console.log('Warming up model...');
  loadingMessage.value = 'Preparing detection...';
  
  // Connect WebSocket early
  connectWebSocket();
  
  // Wait for connection
  await new Promise(r => setTimeout(r, 300));
  
  if (websocket?.readyState === WebSocket.OPEN) {
    try {
      // Capture and send first frame to warm up
      const width = video.videoWidth;
      const height = video.videoHeight;
      
      if (!captureCanvas || captureCanvas.width !== width) {
        captureCanvas = document.createElement('canvas');
        captureCanvas.width = width;
        captureCanvas.height = height;
        captureCtx = captureCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
      }
      
      captureCtx.drawImage(video, 0, 0, width, height);
      const base64 = captureCanvas.toDataURL('image/jpeg', 0.85).split(',')[1];
      
      if (base64 && base64.length > 100) {
        pendingFrames++;
        websocket.send(base64);
        isWarmedUp = true;
        console.log('Model warmed up!');
      }
    } catch (e) {
      console.error('Warmup error:', e);
    }
  }
  
  loadingMessage.value = '';
}

// Video events
function onVideoLoadedData() {
  console.log('Video loaded, warming up...');
  // Small delay to ensure video is ready
  setTimeout(warmupModel, 200);
}

function onVideoPlay() {
  console.log('Video play');
  startTracking();
}

function onVideoPause() {
  console.log('Video pause');
  stopTracking();
}

function onVideoEnded() {
  console.log('Video ended');
  stopTracking();
}

function onVideoSeeked() {
  console.log('Video seeked');
  resetTracker();
  // Re-warmup after seek
  isWarmedUp = false;
  setTimeout(warmupModel, 100);
}

function onVideoError(e) {
  console.error('Video error:', e);
  if (isYouTubeVideo.value) {
    errorMsg.value = 'Cannot play YouTube video directly. Some videos have restrictions.';
  }
}

// Analyze image
async function analyzeImage() {
  if (!file.value) return;
  
  loadingMessage.value = 'Analyzing...';
  
  const canvas = document.createElement('canvas');
  const img = new Image();
  img.src = previewUrl.value;
  
  await new Promise(r => img.onload = r);
  
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  
  const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
  const formData = new FormData();
  formData.append('frame', blob, 'frame.jpg');
  
  const response = await fetch(`${BACKEND_BASE}/detect-annotated`, {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  
  if (data.annotated_frame) {
    annotatedFrameUrl.value = `data:image/jpeg;base64,${data.annotated_frame}`;
  }
  if (data.detections) {
    detections.value = data.detections;
  }
  
  loadingMessage.value = '';
}

async function loadYouTube() {
  if (!youtubeUrl.value) return;
  
  if (backendStatus.value !== 'online') {
    errorMsg.value = 'Backend offline';
    return;
  }
  
  isDownloading.value = true;
  loadingMessage.value = 'Downloading YouTube video...';
  errorMsg.value = '';
  
  try {
    const response = await fetch(`${BACKEND_BASE}/youtube/prepare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: youtubeUrl.value })
    });
    
    const data = await response.json();
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    if (data.proxy_url) {
      // Use local video URL (downloaded to server)
      file.value = { type: 'video/mp4', name: data.title || 'youtube_video.mp4' };
      isYouTubeVideo.value = true;
      previewUrl.value = `${BACKEND_BASE}${data.proxy_url}`;
      youtubeUrl.value = '';
      
      await resetTracker();
      loadingMessage.value = '';
    }
  } catch (err) {
    errorMsg.value = `YouTube error: ${err.message}`;
    loadingMessage.value = '';
  } finally {
    isDownloading.value = false;
  }
}

async function handleFile(selectedFile) {
  isYouTubeVideo.value = false;
  if (!selectedFile) return;
  
  if (backendStatus.value !== 'online') {
    await checkBackendStatus();
    if (backendStatus.value !== 'online') {
      errorMsg.value = 'Backend offline';
      return;
    }
  }
  
  stopTracking();
  await resetTracker();
  
  file.value = selectedFile;
  previewUrl.value = URL.createObjectURL(selectedFile);
  annotatedFrameUrl.value = '';
  detections.value = [];
  errorMsg.value = '';
  
  if (isImage.value) {
    isProcessing.value = true;
    try {
      await new Promise(r => setTimeout(r, 100));
      await analyzeImage();
    } catch (err) {
      errorMsg.value = err.message;
    } finally {
      isProcessing.value = false;
    }
  }
}

function onDrop(event) {
  handleFile(event.dataTransfer?.files?.[0]);
}

const fileInputRef = ref(null);

function onFileInputChange(event) {
  handleFile(event.target.files?.[0]);
  event.target.value = '';
}

onMounted(() => {
  checkBackendStatus();
});
onBeforeUnmount(() => {
  stopTracking(true);
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
});
</script>

<template>
  <div class="page">
    <header class="navbar">
      <div class="navbar-brand">
        <img src="@/assets/logo.png" alt="Sentry" class="brand-logo" />
        <span class="brand-text">Sentry</span>
      </div>
      
      <div class="navbar-right">
        <!-- Device Info -->
        <div class="device-badge" :class="{ gpu: deviceInfo.has_gpu }">
          <span class="device-icon">{{ deviceInfo.has_gpu ? '🚀' : '💻' }}</span>
          <span class="device-name">{{ deviceInfo.name }}</span>
        </div>
        
        <!-- Model Selector -->
        <div class="model-selector" v-if="availableModels.length > 0">
          <label>Model:</label>
          <select 
            :value="selectedModelId" 
            @change="changeModel($event.target.value)"
            :disabled="isChangingModel || isTracking"
          >
            <option 
              v-for="model in availableModels" 
              :key="model.id" 
              :value="model.id"
              :disabled="!model.available"
            >
              {{ model.name }}{{ !model.available ? ' (not found)' : '' }}
            </option>
          </select>
        </div>
        
        <div class="status-badge" :class="backendStatus">
          <span class="dot"></span>
          {{ backendStatus === 'online' ? 'Ready' : backendStatus === 'checking' ? 'Connecting...' : 'Offline' }}
          <button v-if="backendStatus === 'offline'" class="retry" @click="checkBackendStatus">Retry</button>
        </div>
      </div>
    </header>

    <main class="main-content">
      <div class="content-card">
        <!-- Upload -->
        <div v-if="!hasMedia" class="upload-section">
          <div class="drop-zone" @dragover.prevent @drop.prevent="onDrop">
            <input ref="fileInputRef" type="file" accept="image/*,video/*" @change="onFileInputChange" />
            <div class="drop-content">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
              </svg>
              <h2>Upload Video or Image</h2>
              <p>Drag & drop or click • Live YOLO tracking</p>
            </div>
          </div>
          
          <div class="divider">
            <span>OR</span>
          </div>
          
          <div class="youtube-input">
            <div class="yt-icon">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
              </svg>
            </div>
            <input 
              v-model="youtubeUrl" 
              type="text" 
              placeholder="Paste YouTube URL here..."
              @keyup.enter="loadYouTube"
              :disabled="isDownloading"
            />
            <button @click="loadYouTube" :disabled="!youtubeUrl || isDownloading">
              {{ isDownloading ? 'Downloading...' : 'Load' }}
            </button>
          </div>
          
          <div v-if="isDownloading" class="download-progress">
            <div class="spinner small"></div>
            <span>{{ loadingMessage }}</span>
          </div>
          
          <div v-if="errorMsg" class="error-inline">{{ errorMsg }}</div>
        </div>

        <!-- Results -->
        <div v-else class="results">
          <div class="toolbar">
            <button class="back-btn" @click="clearMedia">← New Upload</button>
            <div class="stats">
              <span v-if="isTracking" class="live"><span class="pulse"></span> LIVE</span>
              <span class="soldier-count">
                <span class="dot red"></span>
                soldier: {{ detections.filter(d => d.class === 'soldier').length }}
              </span>
              <span class="civilian-count">
                <span class="dot blue"></span>
                civilian: {{ detections.filter(d => d.class === 'civilian').length }}
              </span>
            </div>
          </div>

          <div class="display">
            <!-- Video element - always present for capture -->
            <video
              v-if="isVideo"
              ref="videoRef"
              :src="previewUrl"
              controls
              crossorigin="anonymous"
              class="video-element"
              :class="{ hidden: isTracking && annotatedFrameUrl }"
              @loadeddata="onVideoLoadedData"
              @play="onVideoPlay"
              @pause="onVideoPause"
              @ended="onVideoEnded"
              @seeked="onVideoSeeked"
              @error="onVideoError"
            />
            
            <!-- Annotated frame display -->
            <img
              v-if="isTracking && annotatedFrameUrl"
              :src="annotatedFrameUrl"
              class="annotated-frame"
              alt="YOLO Tracking"
            />
            
            <!-- Image -->
            <img
              v-if="isImage && !annotatedFrameUrl"
              :src="previewUrl"
              class="media"
              alt="Image"
            />
            <img
              v-if="isImage && annotatedFrameUrl"
              :src="annotatedFrameUrl"
              class="media"
              alt="Detected"
            />

            <div v-if="isProcessing" class="loading">
              <div class="spinner"></div>
              <p>{{ loadingMessage }}</p>
            </div>
            
            <div v-if="isVideo && !isTracking && !annotatedFrameUrl" class="play-hint">
              ▶ Press play to start tracking
            </div>
          </div>

          <div v-if="errorMsg" class="error">{{ errorMsg }}</div>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
* { box-sizing: border-box; }

.page {
  min-height: 100vh;
  background: #0a0a0a;
  color: #fff;
  font-family: system-ui, sans-serif;
}

.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 32px;
  border-bottom: 1px solid #222;
}

.navbar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-logo { 
  width: 32px; 
  height: 32px; 
  object-fit: contain;
  border-radius: 6px;
}
.brand-text { font-size: 20px; font-weight: 700; }

.status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 13px;
  background: #1a1a1a;
}

.status-badge .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-badge.online { color: #22c55e; }
.status-badge.online .dot { background: #22c55e; }
.status-badge.offline { color: #ef4444; }
.status-badge.offline .dot { background: #ef4444; }
.status-badge.checking { color: #f59e0b; }
.status-badge.checking .dot { background: #f59e0b; }

.retry {
  margin-left: 8px;
  padding: 2px 8px;
  font-size: 11px;
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid #ef4444;
  border-radius: 4px;
  color: #ef4444;
  cursor: pointer;
}

.navbar-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.device-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  font-size: 12px;
  color: #888;
}

.device-badge.gpu {
  border-color: #22c55e;
  color: #22c55e;
  background: rgba(34, 197, 94, 0.1);
}

.device-icon {
  font-size: 14px;
}

.device-name {
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-selector {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #888;
}

.model-selector label {
  color: #666;
}

.model-selector select {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #fff;
  padding: 6px 12px;
  font-size: 13px;
  cursor: pointer;
  outline: none;
  transition: border-color 0.2s;
}

.model-selector select:hover {
  border-color: #3b82f6;
}

.model-selector select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.main-content {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 70px);
  padding: 24px;
}

.content-card {
  width: 100%;
  max-width: 1200px;
  background: #111;
  border-radius: 12px;
  border: 1px solid #222;
  overflow: hidden;
}

.upload-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  min-height: 600px;
}

.drop-zone {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  max-width: 550px;
  min-height: 300px;
  border: 2px dashed #333;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.drop-zone:hover { 
  background: rgba(59, 130, 246, 0.05); 
  border-color: #3b82f6;
}

.drop-zone input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.drop-content {
  text-align: center;
  color: #888;
}

.drop-content svg {
  width: 48px;
  height: 48px;
  margin-bottom: 12px;
  color: #3b82f6;
}

.drop-content h2 {
  color: #fff;
  margin-bottom: 8px;
  font-size: 18px;
}

.divider {
  display: flex;
  align-items: center;
  width: 100%;
  max-width: 500px;
  margin: 24px 0;
  color: #555;
}

.divider::before,
.divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #333;
}

.divider span {
  padding: 0 16px;
  font-size: 13px;
}

.youtube-input {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  max-width: 500px;
  padding: 12px 16px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 8px;
}

.yt-icon {
  color: #ff0000;
  flex-shrink: 0;
}

.yt-icon svg {
  width: 24px;
  height: 24px;
}

.youtube-input input {
  flex: 1;
  background: transparent;
  border: none;
  color: #fff;
  font-size: 14px;
  outline: none;
}

.youtube-input input::placeholder {
  color: #666;
}

.youtube-input button {
  padding: 8px 20px;
  background: #3b82f6;
  border: none;
  border-radius: 6px;
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.youtube-input button:hover:not(:disabled) {
  background: #2563eb;
}

.youtube-input button:disabled {
  background: #333;
  color: #666;
  cursor: not-allowed;
}

.download-progress {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
  color: #888;
  font-size: 14px;
}

.spinner.small {
  width: 20px;
  height: 20px;
  border-width: 2px;
}

.error-inline {
  margin-top: 16px;
  padding: 10px 16px;
  background: rgba(239, 68, 68, 0.1);
  border-radius: 6px;
  color: #ef4444;
  font-size: 13px;
}

.results {
  display: flex;
  flex-direction: column;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #222;
}

.back-btn {
  padding: 8px 16px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
}

.back-btn:hover { background: #222; }

.stats {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 14px;
  color: #fff;
}

.soldier-count, .civilian-count {
  display: flex;
  align-items: center;
  gap: 6px;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.dot.red { background: #ef4444; }
.dot.blue { background: #3b82f6; }

.live {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #ef4444;
  font-weight: 600;
}

.pulse {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.display {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 450px;
  background: #000;
}

.video-element {
  max-width: 100%;
  max-height: 70vh;
}

.video-element.hidden {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  z-index: 1;
}

.annotated-frame {
  max-width: 100%;
  max-height: 70vh;
}

.media {
  max-width: 100%;
  max-height: 70vh;
}

.loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.8);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #333;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 12px;
}

@keyframes spin { to { transform: rotate(360deg); } }

.play-hint {
  position: absolute;
  color: #666;
  font-size: 16px;
}

.error {
  padding: 12px 16px;
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  font-size: 14px;
}
</style>
