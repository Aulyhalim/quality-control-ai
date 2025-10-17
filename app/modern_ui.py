# app/modern_ui.py

import gradio as gr
from PIL import Image
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2

# Import untuk Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- PERBAIKAN: Import tanpa prefix 'app.' karena dijalankan dari folder app ---
from model_loader import model_for_pred, model_for_cam, transform, class_names

# ============================================================================
# MODERN CSS STYLING
# ============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
:root { --primary: #00f5ff; --secondary: #ff006e; --success: #00ff9f; --warning: #ffd60a; --info: #7209b7; --bg-dark: #0a0e27; --bg-card: #12172e; --bg-card-hover: #1a2038; --text-primary: #ffffff; --text-secondary: #8b92b8; --border: rgba(0, 245, 255, 0.1); --glow: rgba(0, 245, 255, 0.4); }
body { background: #0a0e27 !important; background-image: radial-gradient(at 0% 0%, rgba(0, 245, 255, 0.05) 0px, transparent 50%), radial-gradient(at 100% 0%, rgba(255, 0, 110, 0.05) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(0, 255, 159, 0.05) 0px, transparent 50%); background-attachment: fixed; }
.gradio-container { max-width: 1600px !important; }
.modern-header { position: relative; padding: 60px 40px; text-align: center; overflow: hidden; border-radius: 24px; background: linear-gradient(135deg, #12172e 0%, #1a2038 100%); border: 1px solid var(--border); margin-bottom: 40px; }
.modern-header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent 0%, var(--primary) 20%, var(--secondary) 50%, var(--success) 80%, transparent 100%); animation: shimmer 3s infinite; }
@keyframes shimmer { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
.header-badge { display: inline-block; padding: 8px 20px; background: rgba(0, 245, 255, 0.1); border: 1px solid var(--primary); border-radius: 50px; color: var(--primary); font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 20px; animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.05); opacity: 0.8; } }
.header-title { font-size: 3.5em; font-weight: 700; background: linear-gradient(135deg, #00f5ff 0%, #ff006e 50%, #00ff9f 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 20px 0; line-height: 1.2; animation: gradientShift 5s ease infinite; }
@keyframes gradientShift { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.header-subtitle { font-size: 1.2em; color: var(--text-secondary); margin-top: 15px; max-width: 700px; margin-left: auto; margin-right: auto; }
.glass-card { background: rgba(18, 23, 46, 0.7) !important; backdrop-filter: blur(20px) !important; border: 1px solid var(--border) !important; border-radius: 20px !important; padding: 30px !important; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important; position: relative !important; }
.glass-card:hover { transform: translateY(-5px); box-shadow: 0 20px 60px rgba(0, 245, 255, 0.1); }
.section-title { font-size: 1.4em; font-weight: 600; color: var(--text-primary); margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }
.section-title::before { content: ''; width: 4px; height: 24px; background: linear-gradient(180deg, var(--primary), var(--secondary)); border-radius: 2px; }
.upload-zone { border: 2px dashed var(--border) !important; border-radius: 16px !important; background: rgba(0, 245, 255, 0.02) !important; transition: all 0.3s ease !important; min-height: 300px; }
.upload-zone:hover { border-color: var(--primary) !important; background: rgba(0, 245, 255, 0.05) !important; box-shadow: 0 0 30px rgba(0, 245, 255, 0.1); }
.result-card-base { border-radius: 20px; padding: 35px; animation: slideIn 0.5s ease-out; }
.result-success { background: linear-gradient(135deg, rgba(0, 255, 159, 0.15), rgba(0, 245, 255, 0.1)); border: 1px solid var(--success); }
.result-danger { background: linear-gradient(135deg, rgba(255, 0, 110, 0.15), rgba(255, 70, 70, 0.1)); border: 1px solid var(--secondary); }
.result-warning { background: linear-gradient(135deg, rgba(255, 214, 10, 0.15), rgba(255, 120, 0, 0.1)); border: 1px solid var(--warning); }
@keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.result-header { display: flex; align-items: center; gap: 15px; margin-bottom: 25px; }
.result-icon { font-size: 3em; line-height: 1; animation: bounce 1s ease-in-out; }
@keyframes bounce { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
.result-title { font-size: 2em; font-weight: 700; color: var(--text-primary); margin: 0; }
.result-prediction { font-size: 1.5em; font-weight: 600; margin: 15px 0; padding: 15px 25px; background: rgba(0, 0, 0, 0.2); border-radius: 12px; display: inline-block; }
.confidence-container { margin: 25px 0; }
.confidence-label { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 1.1em; font-weight: 500; }
.confidence-bar-bg { width: 100%; height: 12px; background: rgba(0, 0, 0, 0.2); border-radius: 10px; overflow: hidden; position: relative; }
.confidence-bar-fill { height: 100%; border-radius: 10px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; }
.bar-success { background: linear-gradient(90deg, var(--primary), var(--success)); }
.bar-danger { background: linear-gradient(90deg, var(--secondary), #ff4646); }
.bar-warning { background: linear-gradient(90deg, var(--warning), #ff8c00); }
.confidence-bar-fill::after { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.3) 50%, transparent 100%); animation: shimmer-bar 2s infinite; }
@keyframes shimmer-bar { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
.analysis-card { background: rgba(18, 23, 46, 0.5); padding: 30px; border-radius: 16px; border: 1px solid var(--border); margin-top: 25px; }
.analysis-title { font-size: 1.2em; font-weight: 600; color: var(--text-primary); margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
.class-item { margin-bottom: 18px; padding: 15px; background: rgba(0, 0, 0, 0.2); border-radius: 12px; transition: all 0.3s ease; }
.class-item:hover { background: rgba(0, 0, 0, 0.3); transform: translateX(5px); }
.class-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.class-name { font-weight: 600; color: var(--text-primary); font-size: 1em; }
.class-confidence { font-weight: 700; color: var(--primary); font-family: 'JetBrains Mono', monospace; }
.class-bar-bg { width: 100%; height: 6px; background: rgba(0, 0, 0, 0.3); border-radius: 10px; overflow: hidden; }
.class-bar-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary)); border-radius: 10px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }
.modern-button { background: linear-gradient(135deg, var(--primary), var(--secondary)) !important; border: none !important; color: white !important; font-weight: 600 !important; padding: 16px 40px !important; border-radius: 12px !important; font-size: 1.1em !important; transition: all 0.3s ease !important; text-transform: uppercase; letter-spacing: 0.05em; position: relative; overflow: hidden; }
.modern-button:hover { transform: translateY(-2px); box-shadow: 0 15px 40px rgba(0, 245, 255, 0.3); }
.modern-footer { text-align: center; padding: 40px 20px; margin-top: 60px; border-top: 1px solid var(--border); color: var(--text-secondary); }
.waiting-state { text-align: center; padding: 60px 20px; color: var(--text-secondary); }
.waiting-icon { font-size: 4em; margin-bottom: 20px; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-20px); } }
"""

# ============================================================================
# KONFIGURASI DAN LOGIKA APLIKASI
# ============================================================================

target_layers = [model_for_cam.layer4[-1]]
cam = GradCAM(model=model_for_cam, target_layers=target_layers)

def get_example_paths():
    """Mencari example images dengan aman."""
    base_path = Path(__file__).parent.parent / "data" / "processed" / "val"
    examples = []
    try:
        defective_files = list((base_path / "defective").glob("*.jpeg"))
        if defective_files: examples.append(str(defective_files[0]))
        good_files = list((base_path / "good").glob("*.jpeg"))
        if good_files: examples.append(str(good_files[0]))
    except FileNotFoundError:
        print("Warning: Folder contoh tidak ditemukan.")
    return examples

def predict_and_visualize(image: Image.Image):
    """Prediksi modern dengan Grad-CAM dan peringatan confidence."""
    if image is None:
        waiting_html = """<div class="waiting-state"><div class="waiting-icon">🎯</div><div class="waiting-text">Upload gambar untuk memulai analisis...</div></div>"""
        return None, waiting_html, ""

    try:
        image_rgb = image.convert("RGB")
        image_np_float = np.array(image_rgb) / 255.0
        image_resized_np = cv2.resize(image_np_float, (224, 224))
        image_tensor = transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = model_for_pred(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        confidences = {name: float(prob) for name, prob in zip(class_names, probabilities.detach().cpu())}
        
        sorted_conf = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        prediction = sorted_conf[0][0]
        score = sorted_conf[0][1]

        target_category_index = class_names.index(prediction)
        targets = [ClassifierOutputTarget(target_category_index)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(image_resized_np, grayscale_cam, use_rgb=True)
        
        confidence_pct = score * 100
        if confidence_pct > 85:
            if prediction == 'good':
                status_icon, status_text, card_class, bar_class = "✅", "PASSED", "result-success", "bar-success"
            else:
                status_icon, status_text, card_class, bar_class = "❌", "DEFECT DETECTED", "result-danger", "bar-danger"
        else:
            status_icon, status_text, card_class, bar_class = "⚠️", "NEEDS HUMAN REVIEW", "result-warning", "bar-warning"
        
        result_html = f"""
        <div class="result-card-base {card_class}">
            <div class="result-header"><div class="result-icon">{status_icon}</div><div><div class="result-title">{status_text}</div></div></div>
            <div class="result-prediction">Classification: <strong>{prediction.upper()}</strong></div>
            <div class="confidence-container">
                <div class="confidence-label"><span>Confidence Level</span><span style="font-family: 'JetBrains Mono', monospace; font-weight: 700;">{confidence_pct:.2f}%</span></div>
                <div class="confidence-bar-bg"><div class="confidence-bar-fill {bar_class}" style="width: {min(confidence_pct, 100)}%"></div></div>
            </div>
        </div>"""

        detail_html = "<div class='analysis-card'><div class='analysis-title'><span>📊</span><span>Detailed Class Probabilities</span></div>"
        for i, (class_name, conf) in enumerate(sorted_conf, 1):
            conf_pct = conf * 100
            detail_html += f"""
            <div class="class-item">
                <div class="class-header">
                    <div class="class-name"><span style="color: var(--text-secondary); margin-right: 8px;">#{i}</span>{class_name.upper()}</div>
                    <div class="class-confidence">{conf_pct:.2f}%</div>
                </div>
                <div class="class-bar-bg"><div class="class-bar-fill" style="width: {conf_pct}%"></div></div>
            </div>"""
        detail_html += "</div>"

        return visualization, result_html, detail_html

    except Exception as e:
        error_html = f"""<div class="result-card-base result-danger"><div class="result-header"><div class="result-icon">🔥</div><div class="result-title">Analysis Error</div></div><div style="margin-top: 20px; font-size: 1em; opacity: 0.9; color: white; font-family: 'JetBrains Mono', monospace;">{str(e)}</div></div>"""
        return None, error_html, ""

# ============================================================================
# ANTARMUKA GRADIO
# ============================================================================

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(), title="AI Quality Control v2.0") as demo:
    gr.HTML("""<div class="modern-header"><div class="header-badge">🚀 AI-Powered System v2.0</div><h1 class="header-title">Quality Control Intelligence</h1><p class="header-subtitle">Advanced defect detection with XAI • Real-time analysis with Grad-CAM</p></div>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title">📤 Upload Image</div>')
            with gr.Group(elem_classes="glass-card"):
                image_input = gr.Image(type="pil", label=None, interactive=True, elem_classes="upload-zone")
                submit_btn = gr.Button("🔍 Start Analysis", variant="primary", elem_classes="modern-button", size="lg")
            with gr.Group(elem_classes="glass-card", elem_id="examples-group"):
                example_paths = get_example_paths()
                if example_paths:
                    gr.Examples(examples=[[path] for path in example_paths], inputs=image_input, cache_examples=False, label="Sample Images")
                else:
                    gr.HTML("""<div style="text-align: center; padding: 20px; color: var(--text-secondary);">No sample images found.</div>""")
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title">📊 Analysis Results</div>')
            with gr.Tabs():
                with gr.TabItem("📈 Main Result", elem_id="main-result-tab"):
                    output_result = gr.HTML()
                    output_details = gr.HTML()
                with gr.TabItem("🧠 Model Explanation (Grad-CAM)", elem_id="grad-cam-tab"):
                    output_heatmap = gr.Image(label=None, interactive=False, type="pil", elem_classes="glass-card")
    gr.HTML("""<div class="modern-footer"><div style="font-size: 1.1em; font-weight: 600; margin-bottom: 10px;">⚡ Powered by PyTorch & Gradio</div><div style="margin-top: 20px; font-size: 0.9em; opacity: 0.7;">© 2025 Advanced AI Systems • All Rights Reserved</div></div>""")
    submit_btn.click(fn=predict_and_visualize, inputs=image_input, outputs=[output_heatmap, output_result, output_details])
    image_input.change(fn=predict_and_visualize, inputs=image_input, outputs=[output_heatmap, output_result, output_details])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)