import os
import asyncio
import json
import base64
import numpy as np
import websockets
import sounddevice as sd
from dotenv import load_dotenv

# Load environment variables (Make sure to set ELEVENLABS_API_KEY in .env)
load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Configuration
MODEL_ID = "scribe_v2_realtime"
SAMPLE_RATE = 16000  # ElevenLabs expects 16kHz audio
CHANNELS = 1
# ElevenLabs WebSocket URL for Realtime Speech-to-Text
WS_URL = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id={MODEL_ID}&commit_strategy=vad"

# Set up an async queue to pass microphone audio to the WebSocket sender
audio_queue = asyncio.Queue()

def mic_callback(indata, frames, time, status):
    """
    This callback is triggered by sounddevice whenever new audio is captured from the mic.
    It runs on a separate C-thread, so we must safely feed the data into our asyncio queue.
    """
    if status:
        print(status, flush=True)
    # Convert audio frames (NumPy array) to raw PCM 16-bit bytes
    pcm_data = (indata * 32767).astype(np.int16).tobytes()
    # Put the raw bytes into the queue loop-safely
    loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_data)
async def send_audio(websocket):
    print("🎤 Mic is live! Start speaking...")

    try:
        while True:
            pcm_chunk = await audio_queue.get()

            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm_chunk).decode("utf-8"),
                "sample_rate": 16000,
            }

            await websocket.send(json.dumps(message))
            audio_queue.task_done()
    except asyncio.CancelledError:
        pass

async def receive_transcript(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("message_type")

            if msg_type == "partial_transcript":
                text = data.get("text", "")
                # Rewrite the current line
                print(f"\r🎤 {text}", end="", flush=True)
                if "fruitsnack" in text or "fruit snack" in text:
                    snack = "fruitsnack"
                    print(f"\n🍎 Detected keyword: {snack}!")

            elif msg_type == "committed_transcript":
                text = data.get("text", "")
                # Finish the current line
                print(f"\r✅ {text}")
                if "fruitsnack" in text or "fruit snack" in text:
                    snack = "fruitsnack"
                    print(f"\n🍎 Detected keyword: {snack}!")


            elif msg_type == "session_started":
                print("✅ Session started.")

            elif msg_type == "error":
                print(f"\n❌ {data}")


    except Exception as e:
        print(f"\nError receiving transcript: {e}")
async def main():
    if not API_KEY:
        print("❌ Error: ELEVENLABS_API_KEY environment variable not found.")
        return

    # Connection headers required by ElevenLabs
    headers = {
        "xi-api-key": API_KEY
    }

    print("Connecting to ElevenLabs Scribe WebSocket...")
    async with websockets.connect(WS_URL, additional_headers=headers) as websocket:
        print("Connected! Initializing microphone...")
        
        # Start the sounddevice input stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=1600,      # 100 ms at 16 kHz
            callback=mic_callback,
        )
        
        with stream:
            # Run the sending and receiving tasks concurrently
            send_task = asyncio.create_task(send_audio(websocket))
            receive_task = asyncio.create_task(receive_transcript(websocket))
            
            # Keep running until Ctrl+C is pressed
            await asyncio.gather(send_task, receive_task)

if __name__ == "__main__":
    # Get the active event loop to share with the threadsafe mic callback
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
        
    except KeyboardInterrupt:
        print("\nStopping transcription. Goodbye!")