from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import torch
import numpy as np
import io
import wave
import asyncio
import json
from datetime import datetime
from typing import Dict, List
import warnings
import re

warnings.filterwarnings("ignore")

app = FastAPI(title="Live Call Center Transcription")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU Check and Model Loading
print("=" * 70)
print("🚀 LIVE CALL CENTER TRANSCRIPTION SYSTEM - WHISPER LARGE-V3")
print("=" * 70)

if torch.cuda.is_available():
    print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = "cuda"
    compute_type = "float16"
else:
    print("⚠️ No GPU detected, using CPU (will be slower)")
    device = "cpu"
    compute_type = "int8"

print("\n📥 Loading Whisper Large-V3 model...")
print("   This may take a moment on first run (downloads ~3GB)")

try:
    whisper_model = WhisperModel(
        "large-v3",
        device=device,
        compute_type=compute_type,
        num_workers=16 if device == "cuda" else 4,
        cpu_threads=16,
        download_root=None
    )
    print("✅ Whisper Large-V3 loaded successfully!")
    print(f"   Device: {device.upper()}")
    print(f"   Expected latency: {'0.5-2 seconds' if device == 'cuda' else '5-15 seconds'}")
except Exception as e:
    print(f"❌ Error loading Large-V3: {e}")
    print("   Falling back to base model...")
    whisper_model = WhisperModel("base", device=device, compute_type=compute_type)

print("=" * 70)

# Active WebSocket connections for real-time updates
active_connections: Dict[str, List[WebSocket]] = {}

class TranscriptionManager:
    """Manages live transcription sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
    
    def create_session(self, session_id: str, language: str):
        """Create a new transcription session"""
        self.sessions[session_id] = {
            "language": language,
            "conversation": [],
            "buffer": b"",
            "created_at": datetime.now().isoformat(),
            "last_transcription": ""
        }
        active_connections[session_id] = []
    
    def add_connection(self, session_id: str, websocket: WebSocket):
        """Add WebSocket connection to session"""
        if session_id not in active_connections:
            active_connections[session_id] = []
        active_connections[session_id].append(websocket)
    
    def remove_connection(self, session_id: str, websocket: WebSocket):
        """Remove WebSocket connection from session"""
        if session_id in active_connections:
            active_connections[session_id].remove(websocket)
    
    async def broadcast(self, session_id: str, message: dict):
        """Broadcast message to all connections in session"""
        if session_id in active_connections:
            disconnected = []
            for connection in active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                active_connections[session_id].remove(conn)
    
    async def transcribe_audio(self, session_id: str, audio_data: bytes, speaker: str):
        """Transcribe audio chunk and broadcast result"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        language = session["language"]
        
        try:
            # Save audio to temporary WAV format
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            
            audio_io.seek(0)
            
            # Language mapping
            language_map = {
                "english": "en",
                "hindi": "hi",
                "marathi": "mr"
            }
            selected_lang = language_map.get(language.lower(), "en")
            
            # Language-specific prompts for better accuracy
            prompts = {
                "hi": "यह कॉल सेंटर बातचीत है। ग्राहक और एजेंट बात कर रहे हैं।",
                "mr": "हे कॉल सेंटर संभाषण आहे. ग्राहक आणि एजेंट बोलत आहेत.",
                "en": "This is a call center conversation between customer and agent."
            }
            
            print(f"🎤 Transcribing {speaker} in {language}...")
            
            # Fast transcription with optimized settings
            segments, info = whisper_model.transcribe(
                audio_io,
                language=selected_lang,
                task="transcribe",
                beam_size=5,
                best_of=3,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.4 if selected_lang in ['hi', 'mr'] else 2.0,
                log_prob_threshold=-0.8,
                no_speech_threshold=0.4,
                condition_on_previous_text=True,
                initial_prompt=prompts[selected_lang],
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            # Collect transcription
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            transcription = transcription.strip()
            
            if len(transcription) > 1:
                # Add to conversation history
                entry = {
                    "speaker": speaker,
                    "text": transcription,
                    "timestamp": datetime.now().isoformat(),
                    "language": language,
                    "confidence": info.language_probability
                }
                
                session["conversation"].append(entry)
                session["last_transcription"] = transcription
                
                print(f"✅ {speaker}: {transcription[:50]}...")
                
                # Broadcast to all connected clients
                await self.broadcast(session_id, {
                    "type": "transcription",
                    "data": entry
                })
                
                return transcription
            else:
                print("⚠️ No speech detected")
                return None
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            await self.broadcast(session_id, {
                "type": "error",
                "message": str(e)
            })
            return None

# Initialize manager
transcription_manager = TranscriptionManager()

# ==================== API ENDPOINTS ====================

@app.post("/api/create-session")
async def create_session(data: dict):
    """Create a new transcription session"""
    session_id = data.get("session_id", f"session_{datetime.now().timestamp()}")
    language = data.get("language", "english")
    
    transcription_manager.create_session(session_id, language)
    
    return JSONResponse({
        "success": True,
        "session_id": session_id,
        "language": language,
        "message": "Session created successfully"
    })

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session conversation history"""
    if session_id in transcription_manager.sessions:
        return JSONResponse({
            "success": True,
            "session": transcription_manager.sessions[session_id]
        })
    else:
        return JSONResponse({
            "success": False,
            "message": "Session not found"
        }, status_code=404)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time transcription"""
    await websocket.accept()
    
    # Add connection to manager
    transcription_manager.add_connection(session_id, websocket)
    
    print(f"🔌 WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive audio data and metadata
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                # Decode base64 audio
                import base64
                audio_bytes = base64.b64decode(data.get("audio"))
                speaker = data.get("speaker", "Citizen")
                
                # Transcribe in background
                asyncio.create_task(
                    transcription_manager.transcribe_audio(
                        session_id, 
                        audio_bytes, 
                        speaker
                    )
                )
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
        transcription_manager.remove_connection(session_id, websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        transcription_manager.remove_connection(session_id, websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "model": "large-v3",
        "device": device,
        "active_sessions": len(transcription_manager.sessions)
    })

@app.get("/")
async def root():
    """Redirect to live transcription interface"""
    return {"message": "Live Call Center Transcription API - Use /live for interface"}

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🎤 LIVE CALL CENTER TRANSCRIPTION SERVER")
    print("="*70)
    print("Model: Whisper Large-V3 (Best Accuracy)")
    print(f"Device: {device.upper()}")
    print(f"Expected Latency: {'0.5-2 seconds' if device == 'cuda' else '5-15 seconds'}")
    print("\nSupported Languages:")
    print("  • English")
    print("  • हिंदी (Hindi)")
    print("  • मराठी (Marathi)")
    print("\nFeatures:")
    print("  ✅ Real-time live transcription")
    print("  ✅ Dual speaker tracking (Citizen + Agent)")
    print("  ✅ WebSocket-based streaming")
    print("  ✅ GPU acceleration enabled")
    print("  ✅ Conversation history")
    print("\n🌐 Server starting at http://localhost:8000")
    print("📱 Open live.html in your browser for the interface")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")