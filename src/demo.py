"""
CXR Sentinel — Phase 4 demo app.

Every number shown in this UI comes from an actual forward pass through your
trained checkpoint on the actual uploaded image. There is no hardcoded
findings dict here — if you upload two different images you get two
different outputs, because the model actually ran on each one.

Run in Colab:
    from src.demo import build_demo
    demo = build_demo(checkpoint_path="checkpoints/best_model.pt")
    demo.launch(share=True)

If no checkpoint exists yet, build_demo() will still launch using a freshly
initialized (untrained) model, purely so you can confirm the wiring works —
it will print a loud warning, and the findings will be near-random until you
actually train and pass a real checkpoint.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from PIL import Image

from src.calibrate import TemperatureScaler
from src.claim_verify import verify_claims
from src.data import DEFAULT_TARGETS, build_transforms
from src.gradcam import GradCAM
from src.model import CXRClassifier
from src.ood import ConvAutoencoder, reconstruction_error
from src.report_draft import claims_to_json, draft_report_templated
from src.temporal import classify_change, compute_heatmap_iou


def load_model(checkpoint_path: str | None, target_names: list[str], device: str):
    model = CXRClassifier(num_targets=len(target_names), pretrained=False).to(device)
    if checkpoint_path is not None:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            target_names = ckpt.get("target_names", target_names)
        except FileNotFoundError:
            warnings.warn(f"No checkpoint at {checkpoint_path} — using an UNTRAINED model. Findings will be near-random.")
    else:
        warnings.warn("No checkpoint provided — using an UNTRAINED model. Findings will be near-random.")
    model.eval()
    return model, target_names


def _preprocess(pil_image: Image.Image, image_size: int = 224) -> torch.Tensor:
    transform = build_transforms(image_size=image_size, train=False)
    return transform(pil_image.convert("RGB"))


def run_pipeline(
    current_pil: Image.Image,
    prior_pil: Image.Image | None,
    model,
    target_names: list[str],
    device: str,
    ood_model: ConvAutoencoder | None = None,
    ood_threshold: float | None = None,
    positive_threshold: float = 0.5,
):
    """The actual, real prediction function — no UI code here, so it's independently testable."""
    current_tensor = _preprocess(current_pil)
    gradcam = GradCAM(model)

    with torch.no_grad():
        logits = model(current_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    findings = []
    gradcam_maps = {}
    for i, name in enumerate(target_names):
        heatmap = gradcam(current_tensor.unsqueeze(0).to(device), target_index=i)
        gradcam_maps[name] = heatmap
        findings.append({"finding": name, "probability": float(probs[i])})

    # Phase 2 — only runs if a prior image was actually provided
    if prior_pil is not None:
        prior_tensor = _preprocess(prior_pil)
        with torch.no_grad():
            prior_logits = model(prior_tensor.unsqueeze(0).to(device))
            prior_probs = torch.sigmoid(prior_logits).squeeze(0).cpu().numpy()

        for i, name in enumerate(target_names):
            prior_heatmap = gradcam(prior_tensor.unsqueeze(0).to(device), target_index=i)
            iou = compute_heatmap_iou(gradcam_maps[name], prior_heatmap)
            status = classify_change(float(probs[i]), float(prior_probs[i]), iou, positive_threshold)
            findings[i]["status"] = status
            findings[i]["prior_probability"] = float(prior_probs[i])

    # Unsupervised OOD flag — only runs if an OOD model was actually passed in
    ood_flag = None
    if ood_model is not None and ood_threshold is not None:
        error = reconstruction_error(ood_model, current_tensor.unsqueeze(0), device)[0]
        ood_flag = {"reconstruction_error": float(error), "threshold": ood_threshold, "flagged_ood": bool(error > ood_threshold)}

    # Phase 3 — real templated drafting + real verification, not a mock
    claims = draft_report_templated(findings)
    verification = verify_claims(claims, findings)

    return {
        "findings": findings,
        "gradcam_maps": gradcam_maps,
        "ood": ood_flag,
        "report_json": claims_to_json(claims),
        "verification": [
            {"finding": v.finding, "claim_text": v.claim_text, "verdict": v.verdict, "reason": v.reason} for v in verification
        ],
    }


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Blends a Grad-CAM heatmap onto the original image for display."""
    import matplotlib.cm as cm

    img = np.array(pil_image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))) / 255.0
    colored = cm.jet(heatmap)[..., :3]
    blended = 0.6 * img + 0.4 * colored
    return (blended * 255).astype(np.uint8)


def build_demo(checkpoint_path: str | None = None, ood_checkpoint_path: str | None = None, device: str | None = None):
    import gradio as gr

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, target_names = load_model(checkpoint_path, DEFAULT_TARGETS, device)

    ood_model = None
    ood_threshold = None
    if ood_checkpoint_path is not None:
        ood_model = ConvAutoencoder().to(device)
        ckpt = torch.load(ood_checkpoint_path, map_location=device)
        ood_model.load_state_dict(ckpt["model_state_dict"])
        ood_threshold = ckpt["threshold"]
        ood_model.eval()

    def predict(current_img, prior_img):
        if current_img is None:
            return None, {}, "Upload an image first.", []

        result = run_pipeline(current_img, prior_img, model, target_names, device, ood_model, ood_threshold)

        primary_finding = max(result["findings"], key=lambda f: f["probability"])["finding"]
        overlay = overlay_heatmap(current_img, result["gradcam_maps"][primary_finding])

        return overlay, {"findings": result["findings"], "ood": result["ood"]}, result["report_json"], result["verification"]

    with gr.Blocks(title="CXR Sentinel") as demo:
        gr.Markdown("# CXR Sentinel — live pipeline, not a mock")
        gr.Markdown(
            "Every field below comes from a real forward pass through the loaded checkpoint. "
            "Upload a prior study too to get Phase 2 longitudinal comparison."
        )
        with gr.Row():
            with gr.Column():
                current_image = gr.Image(type="pil", label="Current study")
                prior_image = gr.Image(type="pil", label="Prior study (optional)")
                submit_btn = gr.Button("Analyze")
            with gr.Column():
                gradcam_output = gr.Image(label="Grad-CAM (highest-probability finding)")
                findings_output = gr.JSON(label="Structured findings + OOD flag (Phase 1/2/unsupervised)")
                report_output = gr.Code(language="json", label="Drafted report claims (Phase 3a)")
                verification_output = gr.JSON(label="Claim verification (Phase 3b)")

        submit_btn.click(
            fn=predict,
            inputs=[current_image, prior_image],
            outputs=[gradcam_output, findings_output, report_output, verification_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
