import os
import gradio as gr
import cv2
import numpy as np
import pandas as pd

# Mock function for AI Diagnostic Analysis (replace or link with your actual processing pipeline)
def run_ai_diagnostic(image):
    if image is None:
        return "Error", "Error", "Error", "Error"
    
    # Example processing logic
    # In your actual app, this processes the frame using your reference database
    health_index = "94.5%"
    primary_fault = "Normal Wear (Minor Scuff)"
    operational_status = "Optimal"
    action_plan = "Continue routine monitoring. Check lubrication in 48 hours."
    
    return health_index, primary_fault, operational_status, action_plan

# Build the Gradio UI interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ESP32 CONVEYOR MONITORING SYSTEM")
    gr.Markdown("Live Sensor Telemetry & AI Visual Diagnostics Hub")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 REAL-TIME HEALTH METRICS & STATUS")
            health_out = gr.Textbox(label="System Health Index (%)", value="Error")
            fault_out = gr.Textbox(label="Primary Fault Match", value="Error")
            status_out = gr.Textbox(label="Operational Status", value="Error")
            action_out = gr.Textbox(label="Prescriptive Action Plan & Safety Protocols", value="Error")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 AI Vision Analysis & Snapshot Input")
            image_input = gr.Image(type="numpy", label="Upload Conveyor Inspection Snapshot")
            analyze_btn = gr.Button("🚀 Run AI Diagnostic Analysis", variant="primary")

    # Connect button to function
    analyze_btn.click(
        fn=run_ai_diagnostic,
        inputs=image_input,
        outputs=[health_out, fault_out, status_out, action_out]
    )

if __name__ == "__main__":
    # Dynamically grab Render's assigned port, default to 10000 locally
    port = int(os.environ.get("PORT", 10000))
    
    # Bind to 0.0.0.0 so external mobile devices and networks can connect safely
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)