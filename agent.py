import asyncio
import base64
import json
import os
import ssl
import sounddevice as sd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
from openai import OpenAI
from playwright.async_api import async_playwright

# Global browser state
playwright = None
browser = None
page = None

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 24000  # OpenAI Realtime API uses 24kHz
CHANNELS = 1
PLAYBACK_SPEED = 1.25  # Speed up voice output (1.0 = normal, 1.25 = 25% faster)
# Allow overriding the synthesized voice via .env (OPENAI_REALTIME_VOICE or REALTIME_VOICE), default to coral
VOICE_NAME = (
    os.environ.get("OPENAI_REALTIME_VOICE")
    or os.environ.get("REALTIME_VOICE")
    or "coral"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERSATIONS_DIR = os.path.join(SCRIPT_DIR, "conversations")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
MEMORY_FILE = os.path.join(SCRIPT_DIR, "memory.json")

# Ensure directories exist
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize conversation log
conversation_log = []
session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load memory
def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"facts": []}

def save_memory(memory: dict):
    """Saves memory to disk (blocking)."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

async def remember_fact(fact: str) -> str:
    """Remember a fact about the user."""
    print(f"🧠 Remembering: {fact}")
    memory["facts"].append(fact)
    await asyncio.to_thread(save_memory, memory)
    return f"I'll remember that: {fact}"

async def log_conversation(role: str, content: str):
    """Log a conversation entry asynchronously."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content
    }
    conversation_log.append(entry)
    
    def _save_log():
        log_file = os.path.join(CONVERSATIONS_DIR, f"conversation_{session_start}.json")
        with open(log_file, "w") as f:
            json.dump(conversation_log, f, indent=2)
            
    await asyncio.to_thread(_save_log)

async def remember_fact(fact: str) -> str:
    """Remember a fact about the user."""
    print(f"🧠 Remembering: {fact}")
    memory["facts"].append(fact)
    
    def _save_memory():
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
            
    await asyncio.to_thread(_save_memory)
    return f"I'll remember that: {fact}

async def web_search(query: str) -> str:
    """Perform a web search using SerpAPI (Google)."""
    print(f"🔍 Searching Google: {query}")

    def _search():
        try:
            serpapi_key = os.environ.get("SERPAPI_KEY")
            if not serpapi_key:
                return "Error: SERPAPI_KEY not set"

            search = GoogleSearch({
                "q": query,
                "api_key": serpapi_key,
            })
            results = search.get_dict()

            if "error" in results:
                return f"Search error: {results['error']}"

            output = []

            if "answer_box" in results:
                answer_box = results.get("answer_box", {})
                title = answer_box.get("title", "Answer")
                answer = answer_box.get("answer")
                snippet = answer_box.get("snippet")
                if answer:
                    output.append(f"{title}: {answer}")
                elif snippet:
                    output.append(f"{title}: {snippet}")


            if "knowledge_graph" in results:
                kg = results.get("knowledge_graph", {})
                title = kg.get("title", "")
                description = kg.get("description")
                if description:
                    output.append(f"{title}: {description}")
                # Extracting attributes from the knowledge graph
                for key, value in kg.items():
                    if isinstance(value, str) and key not in ["title", "description"]:
                        output.append(f"  {key.replace('_', ' ').title()}: {value}")


            if "organic_results" in results:
                output.append("Web Results:")
                for result in results["organic_results"][:5]:
                    title = result.get("title", "No Title")
                    snippet = result.get("snippet", "No Snippet")
                    link = result.get("link", "#")
                    output.append(f"- {title}: {snippet} ({link})")

            if "related_questions" in results:
                output.append("\nRelated Questions:")
                for question in results["related_questions"][:4]:
                    output.append(f"- {question.get('question')}")

            if "local_results" in results:
                output.append("\nLocal Results:")
                for result in results["local_results"][:3]:
                    title = result.get("title", "No Title")
                    address = result.get("address", "")
                    output.append(f"- {title} ({address})")

            if not output:
                return "No results found."

            return "\n".join(output)

        except Exception as e:
            return f"An error occurred during web search: {e}"

    # Run blocking search in a thread
    return await asyncio.to_thread(_search)

def get_current_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

def remember_fact(fact: str) -> str:
    """Remember a fact about the user."""
    print(f"🧠 Remembering: {fact}")
    memory["facts"].append(fact)
    save_memory(memory)
    return f"I'll remember that: {fact}"

def recall_facts() -> str:
    """Recall all remembered facts."""
    if not memory["facts"]:
        return "I don't have any memories stored yet."
    return "Here's what I remember:\n" + "\n".join(f"- {fact}" for fact in memory["facts"])

async def generate_image(prompt: str) -> str:
    """Generate an image using DALL-E."""
    print(f"🎨 Generating image: {prompt}")
    
    def _generate():
        try:
            import httpx
            import subprocess
            
            # Create client
            http_client = httpx.Client()
            client = OpenAI(http_client=http_client)
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            filepath = os.path.join(IMAGES_DIR, filename)
            
            img_response = httpx.get(image_url)
            with open(filepath, "wb") as f:
                f.write(img_response.content)
            
            # Auto-open the image (macOS)
            print(f"🖼️ Opening image: {filepath}")
            subprocess.run(["open", filepath], check=False)
            
            return f"I've generated and saved the image. It should be opening now."
        except Exception as e:
            return f"Image generation error: {str(e)}"

    # Run blocking generation in a thread
    return await asyncio.to_thread(_generate)

async def browser_read() -> str:
    """Read page content."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print("👀 Reading page content...")
    try:
        # Get visible text
        text = await page.evaluate("document.body.innerText")
        return text[:2000] + "..." if len(text) > 2000 else text
    except Exception as e:
        return f"Read error: {str(e)}"

async def browser_screenshot() -> str:
    """Take a screenshot."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print("📸 Taking screenshot...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        await page.screenshot(path=filepath)
        
        # Auto-open
        import subprocess
        subprocess.run(["open", filepath], check=False)
        
        return f"Screenshot saved to {filepath}"
    except Exception as e:
        return f"Screenshot error: {str(e)}"

async def browser_click(text: str) -> str:
    """Click an element by text."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print(f"👆 Clicking: {text}")
    try:
        # Try to find element by text
        await page.get_by_text(text, exact=False).first.click()
        return f"Clicked element containing '{text}'"
    except Exception as e:
        return f"Click error: {str(e)}"

async def browser_type(text: str) -> str:
    """Type text."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print(f"⌨️ Typing: {text}")
    try:
        await page.keyboard.type(text)
        return f"Typed: {text}"
    except Exception as e:
        return f"Type error: {str(e)}"

async def browser_press(key: str) -> str:
    """Press a key."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print(f"🎹 Pressing: {key}")
    try:
        await page.keyboard.press(key)
        return f"Pressed: {key}"
    except Exception as e:
        return f"Press error: {str(e)}"

async def browser_scroll(direction: str) -> str:
    """Scroll page."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print(f"📜 Scrolling: {direction}")
    try:
        if direction == "down":
            await page.evaluate("window.scrollBy(0, 500)")
        else:
            await page.evaluate("window.scrollBy(0, -500)")
        return f"Scrolled {direction}"
    except Exception as e:
        return f"Scroll error: {str(e)}"

async def browser_back() -> str:
    """Go back."""
    await ensure_browser()
    global page
    if not page:
        return "Error: Browser not initialized."
    print("🔙 Going back...")
    try:
        await page.go_back()
        return "Went back to previous page"
    except Exception as e:
        return f"Back error: {str(e)}"

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if name == "web_search":
        return await web_search(arguments.get("query", ""))
    elif name == "get_current_time":
        return get_current_time()
    elif name == "remember":
        return remember_fact(arguments.get("fact", ""))
    elif name == "recall":
        return recall_facts()
    elif name == "generate_image":
        return await generate_image(arguments.get("prompt", ""))
    elif name == "browser_navigate":
        return await browser_navigate(arguments.get("url", ""))
    elif name == "browser_read":
        return await browser_read()
    elif name == "browser_screenshot":
        return await browser_screenshot()
    elif name == "browser_click":
        return await browser_click(arguments.get("text", ""))
    elif name == "browser_type":
        return await browser_type(arguments.get("text", ""))
    elif name == "browser_press":
        return await browser_press(arguments.get("key", ""))
    elif name == "browser_scroll":
        return await browser_scroll(arguments.get("direction", ""))
    elif name == "browser_back":
        return await browser_back()
    else:
        return f"Unknown tool: {name}"

def get_memory_context() -> str:
    """Get memory context for the system prompt."""
    if not memory["facts"]:
        return ""
    return "\n\nYou remember the following about the user:\n" + "\n".join(f"- {fact}" for fact in memory["facts"])

async def ensure_browser():
    """Ensure the browser is initialized."""
    global playwright, browser, page
    if not page:
        print("🌐 Launching browser...")
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()

async def main():
    import websockets
    global playwright, browser, page
    
    # Playwright initialized lazily on first use
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    ssl_context = ssl.create_default_context()

    while True: # Reconnection loop
        try:
            print("=" * 50)
            print("🤖 OpenAI Realtime Voice Agent")
            print("   Tools: web_search, remember, recall, generate_image, browser_*")
            print("   Memory: " + (f"{len(memory['facts'])} facts loaded" if memory['facts'] else "Empty"))
            print("   Connecting...")
            print("=" * 50)

            async with websockets.connect(
                url,
                additional_headers=headers,
                ssl=ssl_context,
                ping_interval=20, # Proactively keep the connection alive
                ping_timeout=60   # Increased timeout for stability
            ) as ws:
                print("✅ Connected!")
                
                system_prompt = """You are a helpful voice assistant with access to tools.
- Use web_search for real-time information (weather, news, etc.)
- Use remember to store facts the user shares about themselves
- Use recall to retrieve stored memories
- Use generate_image when asked to create/draw/generate images
Keep responses concise and conversational.""" + get_memory_context()
                
                session_config = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": system_prompt,
                        "voice": VOICE_NAME,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500
                        },
                        "tools": TOOLS,
                        "tool_choice": "auto"
                    }
                }
                await ws.send(json.dumps(session_config))
                print("📝 Session configured")
                
                # Trigger startup greeting
                await ws.send(json.dumps({
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "instructions": "Say 'Hey' to the user to let them know you are ready."
                    }
                }))
                
                print("🎤 Speak now! (Press Ctrl+C to exit)\n")

                audio_buffer = []
                is_playing = False
                current_ai_response = ""

                async def stream_microphone():
                    loop = asyncio.get_event_loop()
                    
                    def audio_callback(indata, frames, time, status):
                        if status:
                            print(f"Audio status: {status}")
                        if is_playing:
                            return
                        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        asyncio.run_coroutine_threadsafe(
                            ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64
                            })),
                            loop
                        )

                    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                                       callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.1)):
                        while True:
                            await asyncio.sleep(0.1)

                mic_task = asyncio.create_task(stream_microphone())

                try:
                    async for message in ws:
                        event = json.loads(message)
                        event_type = event.get("type", "")

                        if event_type == "session.created":
                            print("🎉 Session created!")

                        elif event_type == "input_audio_buffer.speech_started":
                            print("🎙️ Speech detected...")

                        elif event_type == "input_audio_buffer.speech_stopped":
                            print("⏹️ Speech ended, processing...")

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "")
                            print(f"💬 You said: {transcript}")
                            await log_conversation("user", transcript)

                        elif event_type == "response.audio_transcript.delta":
                            delta = event.get("delta", "")
                            current_ai_response += delta
                            print(delta, end="", flush=True)

                        elif event_type == "response.audio.delta":
                            audio_b64 = event.get("delta", "")
                            audio_bytes = base64.b64decode(audio_b64)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
                            audio_buffer.extend(audio_array)

                        elif event_type == "response.audio.done":
                            if audio_buffer:
                                print("\n🔊 Playing response...")
                                audio_data = np.array(audio_buffer)
                                audio_buffer.clear()

                                async def play_and_update_flag():
                                    nonlocal is_playing
                                    is_playing = True
                                    await asyncio.to_thread(sd.play, audio_data, int(SAMPLE_RATE * PLAYBACK_SPEED))
                                    await asyncio.to_thread(sd.wait)
                                    is_playing = False
                                    print()
                                asyncio.create_task(play_and_update_flag())

                            if current_ai_response:
                                await log_conversation("assistant", current_ai_response)
                                current_ai_response = ""

                        elif event_type == "response.function_call_arguments.done":
                            call_id = event.get("call_id")
                            name = event.get("name")
                            arguments = json.loads(event.get("arguments", "{}"))
                            
                            print(f"\n🔧 Tool call: {name}({arguments})")
                            await log_conversation("tool_call", f"{name}: {arguments}")
                            result = await execute_tool(name, arguments)
                            print(f"📋 Result: {result[:200]}...")
                            await log_conversation("tool_result", result)
                            
                            await ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result
                                }
                            }))
                            await ws.send(json.dumps({"type": "response.create"}))

                        elif event_type == "response.done":
                            print("\n✅ Response complete.\n")

                        elif event_type == "error":
                            error = event.get("error", {})
                            print(f"❌ Error: {error.get('message', 'Unknown error')}")

                except asyncio.CancelledError:
                    pass
                finally:
                    mic_task.cancel()

        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ Connection closed ({e.code} {e.reason}). Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
