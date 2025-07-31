(* current fastagi_recharge.py

sudo systemctl start asterisk


ls /var/run/asterisk/asterisk.ctl

sudo asterisk -rvvvvv


sudo vi /opt/ai/fastagi_recharge.py

sudo chmod +x /opt/ai/fastagi_recharge.py

sudo chown asterisk:asterisk /opt/ai/fastagi_recharge.py

nohup python3 /opt/ai/fastagi_recharge.py > /opt/ai/fastagi_recharge.log 2>&1 &

tail -f /opt/ai/fastagi_recharge.log

tail -f /opt/ai/fastagi_rag.log

ps aux | grep fastagi_recharge.py *)


import socketserver
import subprocess
import os
import whisper
from TTS.api import TTS
import requests
import logging
import time
import torch
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re

LOG_PATH = "/opt/ai/fastagi_rag.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL_NAME = "llama3.2"

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

whisper_model = whisper.load_model("medium", device=device)
logging.info("✅ Whisper model loaded.")

try:
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=(device == "cuda"))
    logging.info("✅ XTTS v2 model loaded.")
except Exception as e:
    logging.error(f"Failed to load TTS model: {e}")
    raise

try:
    with open("/opt/ai/vector_index.pkl", "rb") as f:
        vector_index, vector_docs, embed_model_name = pickle.load(f)
    embed_model = SentenceTransformer(embed_model_name)
    logging.info(f"✅ Loaded embedding model: {embed_model_name}")
except Exception as e:
    logging.error(f"Failed to load vector index or embedding model: {e}")
    vector_index, vector_docs, embed_model = None, None, None

session_histories = {}

def clean_response_for_tts(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r"[.!?]$", line):
            cleaned_lines.append(line + " ")
        else:
            cleaned_lines.append(line + ". ")
    cleaned_text = "".join(cleaned_lines)
    cleaned_text = re.sub(r"^[\*\-\•]\s*", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+\.\s+", ". ", cleaned_text)
    return cleaned_text.strip()

def speak(text, filename="/var/lib/asterisk/sounds/asterisk_response1.wav"):
    cleaned_text = clean_response_for_tts(text)
    if not cleaned_text:
        logging.warning("No text to speak after cleaning.")
        return
    logging.info(f"TTS text (len={len(cleaned_text)}): {cleaned_text[:100]}{'...' if len(cleaned_text) > 100 else ''}")
    try:
        tts_model.tts_to_file(
            text=cleaned_text,
            file_path="/opt/ai/audio/tmp.wav",
            speaker_wav="/opt/ai/audio/aditya.wav",
            language="en"
        )
    except Exception as e:
        logging.error(f"TTS failed: {e}")
        try:
            fallback_text = "Sorry, I am unable to generate speech."
            tts_model.tts_to_file(text=fallback_text, file_path="/opt/ai/audio/tmp.wav")
        except Exception as e2:
            logging.error(f"Fallback TTS failed: {e2}")
            return
    subprocess.run([
        "ffmpeg", "-y", "-i", "/opt/ai/audio/tmp.wav",
        "-ar", "8000", "-ac", "1", "-sample_fmt", "s16",
        "-f", "wav", "-map_metadata", "-1",
        filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists("/opt/ai/audio/tmp.wav"):
        os.remove("/opt/ai/audio/tmp.wav")

def summarize_conversation(history):
    summary_prompt = [
        {"role": "system", "content": "Summarize this conversation in 2 lines."}
    ] + history[-6:]
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL_NAME,
                "messages": summary_prompt,
                "stream": False
            },
            timeout=15
        )
        summary = response.json().get("message", {}).get("content", "").strip()
        logging.info(f"LLM summary: {summary}")
        return summary if summary else "I can't summarize right now."
    except Exception as e:
        logging.error(f"LLM summarization failed: {e}")
        return "I'm having trouble summarizing right now."

class FastAGIHandler(socketserver.StreamRequestHandler):

    def get_balance_by_phone(self, phone_number):
        try:
            response = requests.get(f"http://127.0.0.1:8000/balance/{phone_number}", timeout=5)
            if response.status_code == 200:
                return response.json().get("balance")
            else:
                logging.warning(f"Balance API returned status {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch balance: {e}")
            return None

    def send_recharge_email(self, phone_number, plan_type):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/send_recharge_email",
                json={"phone_number": phone_number, "plan_type": plan_type},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Recharge email API returned status {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Failed to send recharge email: {e}")
            return None

    def confirm_recharge_payment(self, phone_number, plan_type):
        try:
            response = requests.get(
                "http://127.0.0.1:8000/confirm",
                params={"phone_number": phone_number, "plan_type": plan_type},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Confirm recharge API returned status {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Failed to confirm recharge payment: {e}")
            return None

    def get_current_plan(self, phone_number):
        try:
            response = requests.get(f"http://localhost:8000/plan/{phone_number}", timeout=5)
            if response.status_code == 200:
                return response.json().get("current_plan")
            else:
                logging.warning(f"Current plan API returned status {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch current plan: {e}")
            return None

    def get_all_plans(self):
        try:
            response = requests.get("http://localhost:8000/plan/all", timeout=5)
            if response.status_code == 200:
                plan_objs = response.json()
                return [plan.get("name") for plan in plan_objs if "name" in plan]
            else:
                logging.warning(f"All plans API returned status {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Failed to fetch all plans: {e}")
            return []

    def handle(self):
        agi_env = {}
        awaiting_resolution_confirmation = False

        while True:
            line = self.rfile.readline().decode().strip()
            if not line:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                agi_env[key.strip()] = value.strip()

        session_id = agi_env.get("agi_uniqueid", str(time.time()))
        session_histories[session_id] = []
        logging.info(f"📞 Incoming AGI call - session_id: {session_id}")

        def send_cmd(cmd):
            self.wfile.write((cmd + "\n").encode())
            self.wfile.flush()
            response = self.rfile.readline().decode().strip()
            logging.info(f"AGI command: {cmd} -> {response}")
            if "hangup" in response.lower() or "result=-1" in response:
                raise ConnectionError("Call hangup detected")
            return response

        def get_relevant_chunks(query):
            if vector_index is None or embed_model is None:
                return []
            query_vec = embed_model.encode([query])
            D, I = vector_index.search(np.array(query_vec), k=3)
            return [vector_docs[i] for i in I[0] if i < len(vector_docs)]

        def play(text):
            speak(text, filename="/var/lib/asterisk/sounds/asterisk_response1.wav")
            send_cmd("STREAM FILE asterisk_response1 \"*\"")

        try:
            play("Hello I am your Telcovas AI assistant. How can I help you?")

            while True:
                for f in ["/opt/ai/audio/user.wav", "/opt/ai/audio/user_converted.wav"]:
                    if os.path.exists(f):
                        os.remove(f)

                send_cmd("RECORD FILE /opt/ai/audio/user wav \"#\" 10000 2")
                if not os.path.exists("/opt/ai/audio/user.wav"):
                    raise RuntimeError("Recording failed")

                subprocess.run([
                    "ffmpeg", "-y", "-i", "/opt/ai/audio/user.wav",
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                    "/opt/ai/audio/user_converted.wav"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                result = whisper_model.transcribe("/opt/ai/audio/user_converted.wav")
                user_text = result["text"].strip()
                user_text_lower = user_text.lower()
                logging.info(f"🗣️ User said: {user_text}")
                caller_id = agi_env.get("agi_callerid", "")

                # --- Recharge email ---
                if any(word in user_text_lower for word in ["recharge", "top up", "add money"]) and any(plan in user_text_lower for plan in ["daily", "weekly", "monthly", "yearly"]):
                    plan_type = next((plan for plan in ["daily", "weekly", "monthly", "yearly"] if plan in user_text_lower), None)
                    if plan_type:
                        result = self.send_recharge_email(caller_id, plan_type)
                        if result:
                            play(f"An email with the payment link has been sent to your registered email address {result.get('email', '')}. Please check your email to complete the recharge.")
                        else:
                            play("Sorry, I was unable to send the recharge email. Please try again later.")
                    else:
                        play("Please specify the recharge plan you want: daily, weekly, monthly, or yearly.")
                    continue

                # --- Payment confirmation ---
                if any(phrase in user_text_lower for phrase in ["i have paid", "payment done", "payment completed", "recharge done"]):
                    plan_type = next((plan for plan in ["daily", "weekly", "monthly", "yearly"] if plan in user_text_lower), "monthly")
                    result = self.confirm_recharge_payment(caller_id, plan_type)
                    if result:
                        play(f"{result.get('message', 'Your recharge has been confirmed successfully.')}")
                    else:
                        play("Sorry, I couldn't confirm your recharge payment. Please try again later or contact support.")
                    continue

                # --- Balance check ---
                if "balance" in user_text_lower and any(kw in user_text_lower for kw in ["my", "account", "plan", "how much", "check"]):
                    balance = self.get_balance_by_phone(caller_id)
                    if balance is not None:
                        play(f"Your current balance is {balance} rupees.")
                    else:
                        play("Sorry, I could not fetch your balance at the moment.")
                    continue

                # --- Current plan check ---
                if "current plan" in user_text_lower or "which plan" in user_text_lower:
                    current_plan = self.get_current_plan(caller_id)
                    if current_plan:
                        play(f"Your current active plan is the {current_plan} plan.")
                    else:
                        play("Sorry, I couldn't retrieve your current plan.")
                    continue

                # --- All plans listing ---
                if any(word in user_text_lower for word in ["available plans", "all plans", "show plans", "plan options"]):
                    plans = self.get_all_plans()
                    if plans:
                        plan_text = "We offer the following plans: " + ", ".join(plans) + "."
                        play(plan_text)
                    else:
                        play("Sorry, I couldn't retrieve the list of available plans.")
                    continue

                if any(kw in user_text_lower for kw in ["bye", "thank you", "thanks"]):
                    play("Thank you for calling. Have a nice day.")
                    send_cmd("HANGUP")
                    break

                if any(kw in user_text_lower for kw in ["what was my last", "summarize", "recap", "what did i ask"]):
                    summary = summarize_conversation(session_histories[session_id])
                    play(summary)
                    continue

                context_chunks = get_relevant_chunks(user_text)
                context = "\n".join(context_chunks) if context_chunks else ""

                session_histories[session_id].append({"role": "user", "content": user_text})

                SYSTEM_PROMPT = """ You are Telcovas' AI support assistant. The user is currently on a call and may ask multiple related questions. Your job is to:
- Understand and solve the user’s issue step-by-step.
- Ask clarifying questions if needed.
- Use the conversation history to keep context.
- After providing a possible solution, ask: “Has this resolved your issue?”
- If user says yes, reply politely and say goodbye.
- If user says no or asks something else, keep helping.
Keep responses friendly, short, and clear.
                """.strip()

                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session_histories[session_id][-6:]

                try:
                    response = requests.post(
                        OLLAMA_API_URL,
                        json={
                            "model": OLLAMA_MODEL_NAME,
                            "messages": messages,
                            "stream": False
                        },
                        timeout=60
                    )
                    raw_reply = response.json().get("message", {}).get("content", "").strip()
                except Exception as e:
                    raw_reply = "Sorry, I'm unable to respond right now."
                    logging.error(f"LLM error: {e}")

                session_histories[session_id].append({"role": "assistant", "content": raw_reply})
                logging.info(f"🤖 LLM reply (short): {raw_reply}")

                awaiting_resolution_confirmation = "has this resolved your issue" in raw_reply.lower()
                play(raw_reply)

        except ConnectionError:
            logging.info("📴 Call hangup detected, ending session.")
        except Exception as ex:
            logging.error(f"❌ AGI error: {ex}")
        finally:
            logging.info("🛑 AGI handler stopped")

if __name__ == "__main__":
    logging.info("🚀 Starting multilingual FastAGI server on port 4573...")
    server = socketserver.TCPServer(("0.0.0.0", 4573), FastAGIHandler)
    try:
        server.serve_forever()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
