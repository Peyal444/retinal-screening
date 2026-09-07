"""Retinal Disease Screening — polished visitor web app.
Folder contents for deployment: app.py + requirements.txt + hier_swin_best.pth
Run locally:  python -m streamlit run app.py
Deploy: Streamlit Community Cloud (upload this folder to GitHub, set main file app.py).
"""
from pathlib import Path
import urllib.request
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm
import streamlit as st

st.set_page_config(page_title="Retinal Disease Screening", page_icon="👁️", layout="wide")

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
L2_NAMES = ["Normal", "DR", "AMD", "Glaucoma", "Other"]
L2_INFO = {
    "Normal": "No disease signs found. Keep up regular eye checkups.",
    "DR": "Diabetic retinopathy signs found — see severity grade below.",
    "AMD": "Macular degeneration signs found. Please see an ophthalmologist.",
    "Glaucoma": "Optic-nerve (glaucoma) signs found. Urgent referral recommended.",
    "Other": "Another condition found (e.g. cataract). Please see an ophthalmologist.",
}
L3_NAMES = ["Mild DR", "Moderate DR", "Severe DR", "Prolif DR", "Adv PDR", "VerySev NPDR"]
CKPT = Path(__file__).parent / "hier_swin_best.pth"
MODEL_URL = "https://huggingface.co/buckets/Peyal/retinal-swin-hier-bucket/resolve/hier_swin_best.pth?download=true"


def ensure_model():
    if CKPT.exists():
        return
    with st.spinner("Downloading model (110MB, first run only)..."):
        try:
            urllib.request.urlretrieve(MODEL_URL, CKPT)
        except Exception as e:
            st.error(f"Model download failed: {e}. Admin: check MODEL_URL.")
            st.stop()


ensure_model()

st.markdown(
    """
    <style>
    .hero { background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
            padding: 28px 30px; border-radius: 16px; color: white; margin-bottom: 18px; }
    .hero h1 { margin: 0; font-size: 30px; }
    .hero p { margin: 6px 0 0 0; opacity: 0.9; }
    .card { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px; color: #111827; }
    .result-ok { background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 14px; padding: 16px 18px; color: #111827; }
    .result-bad { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 14px; padding: 16px 18px; color: #111827; }
    .small { color: #4b5563; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


class HierarchicalSwin(nn.Module):
    def __init__(self, backbone="swin_tiny_patch4_window7_224", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        f = self.backbone.num_features
        self.head_l1 = nn.Linear(f, 2)
        self.head_l2 = nn.Linear(f, 5)
        self.head_l3 = nn.Linear(f, 6)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_l1(f), self.head_l2(f), self.head_l3(f)


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = HierarchicalSwin(pretrained=False).to(device)
    m.load_state_dict(torch.load(str(CKPT), map_location=device))
    m.eval()
    return m, device


def circle_mask(n=IMG_SIZE, r_frac=0.46):
    yy, xx = np.ogrid[:n, :n]
    c = (n - 1) / 2
    return ((yy - c) ** 2 + (xx - c) ** 2) <= (r_frac * n) ** 2


def gradcam(model, x, head="l2", stage=2):
    feats, grads = {}, {}

    def fwd(m, i, o):
        feats["v"] = o.detach()

    def bwd(m, gi, go):
        grads["v"] = go[0].detach()

    layer = model.backbone.layers[stage]
    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)
    model.zero_grad()
    o1, o2, o3 = model(x)
    out = {"l1": o1, "l2": o2, "l3": o3}[head]
    c = out.argmax(1).item()
    probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
    out[0, c].backward()
    h1.remove()
    h2.remove()
    F, G = feats["v"][0], grads["v"][0]
    if F.dim() == 2:
        s = int(F.shape[0] ** 0.5)
        F, G = F.view(s, s, -1), G.view(s, s, -1)
    w = G.mean(dim=(0, 1))
    cam = (F * w).sum(-1).relu().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return (cam * circle_mask()), c, probs


st.markdown(
    "<div class='hero'><h1>👁️ Retinal Disease Screening</h1>"
    "<p>Upload a retinal fundus photo — get an instant AI screening result with visual explanation.</p></div>",
    unsafe_allow_html=True,
)

if not CKPT.exists():
    st.error("Model file missing: put hier_swin_best.pth in this folder.")
    st.stop()

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("<div class='card'><b>📤 Step 1 — Upload your eye photo</b><br>"
                "<span class='small'>Clear color fundus image (JPG/PNG). Your photo never leaves this page.</span></div>",
                unsafe_allow_html=True)
    img_file = st.file_uploader("Fundus image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
                                label_visibility="collapsed")
    stage = 2
    st.markdown("<div class='card'><b>🔬 How the AI decides (3 steps, automatic)</b><br>"
                "<span class='small'>"
                "<b>Step 1 — 2 choices:</b> Healthy vs Sick<br>"
                "<b>Step 2 — 5 families:</b> Normal, DR (diabetic retinopathy), AMD, Glaucoma, Other "
                "(Cataract / Hypertensive / Myopia)<br>"
                "<b>Step 3 — 6 DR grades</b> (only if DR): Mild, Moderate, Severe, Prolif, Adv PDR, VerySevere"
                "</span></div>",
                unsafe_allow_html=True)

with right:
    if not img_file:
        st.markdown("<div class='card'>👈 <b>Your result will appear here.</b><br>"
                    "<span class='small'>Upload a photo on the left to begin.</span></div>", unsafe_allow_html=True)
    else:
        with st.spinner("🔬 Analyzing your photo…"):
            model, device = load_model()
            raw = Image.open(img_file).convert("RGB")
            tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)])
            x = tf(raw).unsqueeze(0).to(device)
            with torch.no_grad():
                o1, o2, o3 = model(x)
                p1 = torch.softmax(o1, 1)[0].cpu().numpy()
                p2 = torch.softmax(o2, 1)[0].cpu().numpy()
                p3 = torch.softmax(o3, 1)[0].cpu().numpy()
            l1, l2, l3 = int(p1.argmax()), int(p2.argmax()), int(p3.argmax())

        st.markdown("<b>✅ Step 2 — Screening result</b>", unsafe_allow_html=True)
        if l1 == 0:
            st.markdown(f"<div class='result-ok'><b style='font-size:20px'>✅ Healthy retina</b><br>"
                        f"Confidence: {p1.max():.0%}<br><span class='small'>{L2_INFO['Normal']}</span></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-bad'><b style='font-size:20px'>🔍 {L2_NAMES[l2]} detected</b><br>"
                        f"Confidence: {p2.max():.0%}<br><span class='small'>{L2_INFO[L2_NAMES[l2]]}</span></div>",
                        unsafe_allow_html=True)
            if l2 == 1:
                st.warning(f"DR severity: **{L3_NAMES[l3]}** ({p3.max():.0%} confident)")
        st.progress(float(p1.max()))

        st.markdown("<b>🔥 Step 3 — What the AI looked at</b> <span class='small'>(red = focused areas)</span>",
                    unsafe_allow_html=True)
        head = "l1" if l1 == 0 else ("l3" if l2 == 1 else "l2")
        cam, _, _ = gradcam(model, x, head=head, stage=stage)
        raw_rs = np.array(raw.resize((IMG_SIZE, IMG_SIZE)))
        heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        overlay = (0.55 * raw_rs + 0.45 * heat).astype(np.uint8)
        c1, c2, c3 = st.columns(3)
        c1.image(raw_rs, caption="📷 Your photo", use_column_width=True)
        c2.image((cam * 255).astype(np.uint8), caption="🌡️ AI attention", use_column_width=True)
        c3.image(overlay, caption="✨ Overlay (red = focus)", use_column_width=True)

        st.markdown("<b>📊 How sure is the AI? (all answers on page)</b>", unsafe_allow_html=True)
        st.write("**Step 1 — Healthy or Sick? (2 choices)**")
        for name, v in zip(["Healthy", "Sick"], p1):
            icon = "✅" if name == ["Healthy", "Sick"][l1] else "▫️"
            st.write(f"{icon} {name}: {v:.1%}")
            st.progress(float(v))
        st.write("**Step 2 — Which disease family? (5 choices)**")
        for name, v in zip(L2_NAMES, p2):
            icon = "👉" if name == L2_NAMES[l2] else "▫️"
            st.write(f"{icon} {name}: {v:.1%}")
            st.progress(float(v))
        if l1 == 1 and l2 == 1:
            st.write("**Step 3 — DR severity grade (6 grades)**")
            for name, v in zip(L3_NAMES, p3):
                icon = "👉" if name == L3_NAMES[l3] else "▫️"
                st.write(f"{icon} {name}: {v:.1%}")
                st.progress(float(v))

st.divider()
st.caption("⚠️ Screening aid only — not a medical diagnosis. Please consult an ophthalmologist. "
           "Built with PyTorch + Swin Transformer for thesis demo.")
