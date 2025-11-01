import base64
import json
import wave
import numpy as np
import websocket
import time
import io
import requests

# generate 1s test tone (not speech, but tests the pipeline)
sr = 16000
t = np.linspace(0, 1, int(sr), False)
tone = 0.1 * np.sin(2 * np.pi * 220 * t)  # 220Hz tone

# save to WAV bytes
buf = io.BytesIO()
with wave.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes((tone * 32767).astype('int16').tobytes())
wav_bytes = buf.getvalue()

session_id = 'test_session_automated'
ws_url = f'ws://localhost:8000/ws/{session_id}'

# create session via HTTP
print('Creating session', session_id)
resp = requests.post('http://localhost:8000/api/create-session', json={'session_id': session_id, 'language': 'english'})
print('Create session response:', resp.status_code, resp.text)

print('Connecting to websocket', ws_url)
ws = websocket.create_connection(ws_url, timeout=10)

msg = {
    'type': 'audio',
    'audio': base64.b64encode(wav_bytes).decode('ascii'),
    'speaker': 'Citizen'
}

print('Sending audio chunk (base64, %d bytes)' % len(wav_bytes))
ws.send(json.dumps(msg))

print('Sent. Waiting for messages (10s)...')
deadline = time.time() + 10
try:
    while time.time() < deadline:
        try:
            r = ws.recv()
            print('Received:', r)
        except Exception as e:
            # no message available yet
            time.sleep(0.5)
            continue
finally:
    ws.close()
    print('WebSocket closed')
