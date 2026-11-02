import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import io

st.set_page_config(page_title="Model Arena — Construction Site AI", page_icon="🎨", layout="wide")

st.title("🎨 Generative Safety Model Arena")
st.markdown("""
Interactively synthesize photorealistic construction worker crops across all 4 safety states. 
Models are dynamically loaded on the GPU in FP16 precision to strictly preserve the **6 GB VRAM budget**.
""")

st.divider()

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("Synthesis Controls")
    model_choice = st.selectbox(
        "Select Generative Architecture:",
        ["Conditional VAE (C-VAE)", "Conditional GAN (DCGAN)", "Conditional DDPM Diffusion", "VQ-VAE + ImageGPT"]
    )

    class_names = {
        0: "Class 0: Compliant (Hardhat & Vest)",
        1: "Class 1: Missing Helmet",
        2: "Class 2: Missing Vest",
        3: "Class 3: Both Missing"
    }
    selected_class = st.selectbox("Target Safety State:", options=list(class_names.keys()), format_func=lambda x: class_names[x])
    num_samples = st.slider("Number of Worker Crops:", min_value=1, max_value=16, value=4)
    seed = st.number_input("Random Seed:", min_value=0, max_value=99999, value=42)

    # Architecture-specific parameters
    if model_choice == "Conditional DDPM Diffusion":
        cfg_scale = st.slider("Guidance Scale (CFG w):", min_value=1.0, max_value=7.5, value=3.0, step=0.5)
        ddim_steps = st.select_slider("DDIM Sampling Steps:", options=[10, 20, 25, 50], value=20)
    elif model_choice == "VQ-VAE + ImageGPT":
        temp = st.slider("Sampling Temperature (T):", min_value=0.5, max_value=1.5, value=1.0, step=0.1)

    btn_generate = st.button("🚀 Generate Synthetic Crops", type="primary", use_container_width=True)

with col_view:
    st.subheader("Synthesized Output")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if btn_generate:
        torch.manual_seed(seed)
        np.random.seed(seed)

        with st.spinner(f"Generating {num_samples} crops with {model_choice}..."):
            try:
                if model_choice == "Conditional VAE (C-VAE)":
                    from src.models.vae import ConditionalVAE
                    ckpt_path = "runs/cvae_baseline/best.pt" if os.path.exists("runs/cvae_baseline/best.pt") else "runs/cvae_baseline/latest.pt"
                    if not os.path.exists(ckpt_path):
                        st.error("C-VAE checkpoint not found.")
                        st.stop()

                    model = ConditionalVAE(in_channels=3, latent_dim=128, num_classes=4, class_emb_dim=32).to(device)
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()

                    with torch.no_grad():
                        z = torch.randn(num_samples, 128, device=device)
                        y = torch.full((num_samples,), selected_class, dtype=torch.long, device=device)
                        samples = model.decode(z, y).clamp(0, 1)

                elif model_choice == "Conditional GAN (DCGAN)":
                    from src.models.gan import ConditionalGenerator
                    ckpt_path = "runs/cgan_baseline/best.pt" if os.path.exists("runs/cgan_baseline/best.pt") else "runs/cgan_baseline/latest.pt"
                    if not os.path.exists(ckpt_path):
                        st.error("Conditional GAN checkpoint not found. Training may still be in progress.")
                        st.stop()

                    model = ConditionalGenerator(latent_dim=128, num_classes=4).to(device)
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["netG_state_dict"])
                    model.eval()

                    with torch.no_grad():
                        z = torch.randn(num_samples, 128, device=device)
                        y = torch.full((num_samples,), selected_class, dtype=torch.long, device=device)
                        raw = model(z, y)
                        samples = ((raw + 1.0) * 0.5).clamp(0, 1)

                elif model_choice == "Conditional DDPM Diffusion":
                    from src.models.diffusion import get_diffusion_unet, GaussianDiffusion
                    ckpt_path = "runs/diffusion_baseline/best.pt" if os.path.exists("runs/diffusion_baseline/best.pt") else "runs/diffusion_baseline/latest.pt"
                    if not os.path.exists(ckpt_path):
                        st.error("Diffusion checkpoint not found. Training may still be in progress.")
                        st.stop()

                    unet = get_diffusion_unet(input_resolution=64, base_channels=64, num_classes=4).to(device)
                    diffusion = GaussianDiffusion(unet=unet, num_classes=4).to(device)
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    if "ema_unet_state_dict" in ckpt:
                        unet.load_state_dict(ckpt["ema_unet_state_dict"])
                    else:
                        unet.load_state_dict(ckpt["unet_state_dict"])
                    diffusion.eval()

                    with torch.no_grad():
                        raw, _ = diffusion.sample_ddim(
                            num_samples=num_samples,
                            class_id=selected_class,
                            steps=ddim_steps,
                            guidance_scale=cfg_scale,
                            device=device
                        )
                        samples = ((raw + 1.0) * 0.5).clamp(0, 1)

                elif model_choice == "VQ-VAE + ImageGPT":
                    from src.models.vqvae import VQVAE
                    from src.models.transformer import ImageTokenTransformer
                    ckpt_path = "runs/vqvae_transformer_baseline/best.pt" if os.path.exists("runs/vqvae_transformer_baseline/best.pt") else "runs/vqvae_transformer_baseline/latest.pt"
                    if not os.path.exists(ckpt_path):
                        st.error("VQ-VAE + ImageGPT checkpoint not found. Training may still be in progress.")
                        st.stop()

                    vqvae = VQVAE(in_channels=3, codebook_size=512, embedding_dim=64).to(device)
                    gpt = ImageTokenTransformer(vocab_size=516, seq_len=257, n_layer=6, n_head=8, n_embd=384).to(device)
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    vqvae.load_state_dict(ckpt["vqvae_state_dict"])
                    gpt.load_state_dict(ckpt["gpt_state_dict"])
                    vqvae.eval()
                    gpt.eval()

                    with torch.no_grad():
                        sample_tokens = gpt.generate(class_id=selected_class, num_samples=num_samples, temperature=temp, device=device)
                        samples = vqvae.decode_indices(sample_tokens).clamp(0, 1)

                # Render grid
                grid = vutils.make_grid(samples, nrow=4, padding=2)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                pil_img = Image.fromarray(ndarr)

                st.image(pil_img, use_container_width=True, caption=f"Generated {class_names[selected_class]} using {model_choice}")

                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Generated Grid (PNG)",
                    data=buf.getvalue(),
                    file_name=f"synthetic_{model_choice.split()[0].lower()}_class_{selected_class}.png",
                    mime="image/png"
                )

                # Free GPU memory
                if "model" in locals():
                    del model
                if "unet" in locals():
                    del unet
                if "vqvae" in locals():
                    del vqvae
                if "gpt" in locals():
                    del gpt
                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"Generation error: {e}")
    else:
        st.info("Select parameters and click **Generate Synthetic Crops** to synthesize workers.")
