import os
import urllib.request
import zipfile
import cv2
import numpy as np
import gradio as gr
import datetime
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- AUTOMATED REFERENCE DB DOWNLOADER (GITHUB) ---
REFERENCE_DIR = "reference_db/"

if not os.path.exists(REFERENCE_DIR) or not os.listdir(REFERENCE_DIR):
    print("📥 Downloading reference database from GitHub...")
    zip_url = "https://github.com/BhimanapalliAkshaya/conveyor-reference-db/raw/main/reference_db.zip"
    urllib.request.urlretrieve(zip_url, "ref.zip")
    with zipfile.ZipFile("ref.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ Reference database downloaded and extracted successfully!")
else:
    print("✅ Reference database already exists locally.")

# --- LOCAL CSV LOGGING SETUP ---
LOG_FILE = "conveyor_maintenance_logs.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Health_Index,Primary_Fault,Status\n")

def log_to_csv(health, fault, status):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f'"{timestamp}",{health},"{fault}","{status}"\n')

# --- CORE EVALUATION ENGINE ---
def get_prescriptive_solution(fault_name, health_index):
    if health_index > 80:
        return "✅ System is healthy. Continue standard scheduled operations."

    actions = []
    fault_lower = fault_name.lower()

    if "rupture" in fault_lower or "tear" in fault_lower or "seperation" in fault_lower or "separation" in fault_lower or "failure" in fault_lower:
        actions.append("🛑 [CRITICAL EMERGENCY] Structural failure/rupture detected! Shut down conveyor immediately.")
        actions.append("🛠️ [Repair] Isolate section, inspect mechanical fasteners/welds, and replace damaged components.")
    elif "misalignment" in fault_lower:
        actions.append("⚠️ [Warning] Belt tracking is off-center.")
        actions.append("🛠️ [Adjustment] Adjust tracking idler rollers and clear material buildup.")
    elif "normal" not in fault_lower and fault_name != "Unknown":
        actions.append(f"🔍 [Inspection] Investigate surface anomaly classified as: {fault_name.upper()}")

    return " \n ".join(actions)

def evaluate_conveyor_system(test_img):
    if test_img is None:
        return "⚠️ Error: Please upload an inspection image.", 0.0, "N/A", "No image provided.", {}

    img_test = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

    best_match_name = "Unknown"
    highest_score = -1.0
    all_scores = {}

    if os.path.exists(REFERENCE_DIR):
        for filename in os.listdir(REFERENCE_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                ref_path = os.path.join(REFERENCE_DIR, filename)
                img_ref = cv2.imread(ref_path)
                if img_ref is None:
                    continue

                h, w = img_ref.shape[:2]
                img_test_resized = cv2.resize(img_test, (w, h))

                hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
                hsv_test = cv2.cvtColor(img_test_resized, cv2.COLOR_BGR2HSV)

                hist_ref = cv2.calcHist([hsv_ref], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist_ref, hist_ref, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                hist_test = cv2.calcHist([hsv_test], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                score = cv2.compareHist(hist_ref, hist_test, cv2.HISTCMP_CORREL)

                fault_category = os.path.splitext(filename)[0]
                score_pct = round(max(0.0, float(score)) * 100, 2)
                all_scores[fault_category] = score_pct

                if score > highest_score:
                    highest_score = score
                    best_match_name = fault_category

    visual_confidence = round(max(0.0, float(highest_score)) * 100, 2)
    fault_lower = best_match_name.lower()

    if visual_confidence < 28.0 and "normal" not in fault_lower:
        health_index = 95.0
        status = "🟢 HEALTHY: System operating normally."
        best_match_name = "Normal Operation"
    elif "rupture" in fault_lower or "tear" in fault_lower or "seperation" in fault_lower or "separation" in fault_lower or "failure" in fault_lower:
        health_index = 15.0
        status = f"🔴 CRITICAL ALERT: Structural failure ({best_match_name.upper()}) detected!"
    elif "normal" in fault_lower:
        health_index = 95.0
        status = "🟢 HEALTHY: System operating normally."
    else:
        raw_health = round(max(0.0, 70.0 - visual_confidence), 2)
        health_index = min(raw_health, 65.0)

        if health_index > 40:
            status = "🟡 WARNING: Degradation or anomaly detected."
        else:
            status = "🔴 CRITICAL ALERT: Immediate inspection required!"

    prescriptive_guidance = get_prescriptive_solution(best_match_name, health_index)
    log_to_csv(health_index, best_match_name.upper(), status)

    return status, health_index, best_match_name.upper(), prescriptive_guidance, all_scores

def load_recent_logs():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            return df.tail(10)
        except Exception:
            return pd.DataFrame(columns=["Timestamp", "Health_Index", "Primary_Fault", "Status"])
    return pd.DataFrame(columns=["Timestamp", "Health_Index", "Primary_Fault", "Status"])

# --- COLORFUL VIBRANT CUSTOM CSS & THEME ---
vibrant_css = """
body {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%) !important;
}
.gradio-container {
    background: rgba(15, 23, 42, 0.90) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}
.gr-box, .gr-block, .tabitem {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%) !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
.gr-button-primary {
    background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 50%, #6366f1 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
    transition: all 0.3s ease;
}
.gr-button-primary:hover {
    background: linear-gradient(90deg, #0891b2 0%, #2563eb 50%, #4f46e5 100%) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
}
.heading-telemetry {
    background: linear-gradient(90deg, rgba(6, 182, 212, 0.25), transparent);
    border-left: 4px solid #06b6d4;
    padding: 8px 14px;
    border-radius: 6px;
    color: #38bdf8;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 15px;
}
.heading-logs {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), transparent);
    border-left: 4px solid #6366f1;
    padding: 8px 14px;
    border-radius: 6px;
    color: #818cf8;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 15px;
}
.heading-internal {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.2), transparent);
    border-left: 4px solid #f59e0b;
    padding: 6px 12px;
    border-radius: 4px;
    color: #fbbf24;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    margin-top: 15px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(css=vibrant_css, title="ESP32 - Conveyor Monitoring System") as demo:
    with gr.Row():
        gr.Markdown(
            """
            <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%); padding: 22px; border-radius: 14px; display: flex; justify-content: space-between; align-items: center; border: 2px solid #06b6d4; box-shadow: 0 0 20px rgba(6, 182, 212, 0.25);">
                <div>
                    <span style="font-size: 26px;">⚡</span>
                    <span style="color: #ffffff; font-size: 22px; font-weight: 800; margin-left: 10px; letter-spacing: 0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">ESP32 - CONVEYOR MONITORING SYSTEM</span>
                    <p style="color: #38bdf8; font-size: 13px; margin: 4px 0 0 38px; font-weight: 600;">Live Sensor Telemetry & AI Visual Diagnostics Hub</p>
                </div>
                <div style="text-align: right; background: rgba(6, 182, 212, 0.15); padding: 10px 18px; border-radius: 10px; border: 1px solid #06b6d4;">
                    <span style="color: #34d399; font-size: 13px; font-weight: bold;">● SYSTEM ONLINE</span>
                    <p style="color: #94a3b8; font-size: 11px; margin: 0; font-weight: 500;">IP: 192.168.1.45 | COM5</p>
                </div>
            </div>
            """
        )

    with gr.Tabs():
        with gr.TabItem("📊 Live Telemetry Dashboard"):
            gr.Markdown("### 📈 REAL-TIME HEALTH METRICS & STATUS", elem_classes=["heading-telemetry"])
            
            with gr.Row():
                out_health = gr.Number(label="💓 System Health Index (%)", interactive=False)
                out_match = gr.Textbox(label="🔍 Primary Fault Match", interactive=False)
                out_status = gr.Textbox(label="⚡ Operational Status", interactive=False)

            with gr.Row():
                out_action = gr.Textbox(label="⚠️ Prescriptive Action Plan & Safety Protocols", lines=3, interactive=False)

            gr.Markdown("#### 📷 AI Vision Analysis & Snapshot Input", elem_classes=["heading-internal"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="numpy", label="Upload Conveyor Inspection Snapshot")
                    with gr.Row():
                        clear_btn = gr.ClearButton([img_input], value="Clear")
                        submit_btn = gr.Button("🚀 Run AI Diagnostic Analysis", variant="primary")
                
                with gr.Column(scale=1):
                    out_json = gr.JSON(label="📊 Visual Confidence Breakdown (%)", value={})

        with gr.TabItem("📋 Serial / CSV Audit Logs"):
            gr.Markdown("### 🖥️ HARDWARE SERIAL LOG BUFFER & HISTORY", elem_classes=["heading-logs"])
            log_table = gr.DataFrame(value=load_recent_logs(), interactive=False, wrap=True)
            refresh_btn = gr.Button("🔄 Refresh Log Table")
            refresh_btn.click(fn=load_recent_logs, outputs=[log_table])

    submit_btn.click(
        fn=evaluate_conveyor_system,
        inputs=[img_input],
        outputs=[out_status, out_health, out_match, out_action, out_json]
    ).then(
        fn=load_recent_logs,
        outputs=[log_table]
    )

# --- FASTAPI BACKEND & WEBHOOK INTEGRATION ---
app = FastAPI()

class TelemetryData(BaseModel):
    vibration: float
    load: float
    current: float

@app.post("/api/telemetry")
async def receive_esp32_telemetry(data: TelemetryData):
    status = "🟢 HEALTHY: System operating normally."
    if data.vibration > 15.0 or data.load > 5.0 or data.current > 2.5:
        status = "🔴 WARNING: ESP32 sensor threshold exceeded!"
    
    log_to_csv(health=85.0, fault=f"Vib:{data.vibration} | Load:{data.load}", status=status)
    return {"status": "success", "message": "Telemetry received successfully"}

# Mount Gradio app onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# --- CLOUD LAUNCH CONFIGURATION ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)