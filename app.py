import os
import io
import zipfile
import tempfile
import streamlit as st
import fal_client

st.set_page_config(
    page_title="AI Influencer Studio",
    page_icon="✨",
    layout="wide"
)

st.title("✨ AI Influencer Studio (Flux LoRA)")
st.caption("Train a consistent AI influencer and generate realistic photos")

# Sidebar - API Key & Tips
with st.sidebar:
    st.header("⚙️ Settings")
    default_key = os.environ.get("FAL_KEY", "")
    fal_key = st.text_input("Fal.ai API Key", value=default_key, type="password", help="Enter your key from fal.ai/dashboard/keys")
    if fal_key:
        os.environ["FAL_KEY"] = fal_key

    st.markdown("---")
    st.markdown("### 💡 Tips for Best Consistency:")
    st.markdown("""
    - **15–25 Photos:** Use clear, high-res images.
    - **Variations:** Different angles, lightings, and expressions.
    - **Clean background:** Avoid group photos or other people's faces.
    """)

if not os.environ.get("FAL_KEY"):
    st.warning("⚠️ Please enter your Fal.ai API Key in the sidebar (or add FAL_KEY in Space Secrets) to get started.")
    st.stop()

if "lora_url" not in st.session_state:
    st.session_state.lora_url = ""
if "trigger_word" not in st.session_state:
    st.session_state.trigger_word = "TOK"

tab1, tab2 = st.tabs(["🚀 1. Train LoRA", "🎨 2. Generate Photos"])

# TAB 1: TRAIN LORA
with tab1:
    st.subheader("Step 1: Train Your Influencer Face")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload 15-25 Character Images", 
            type=["png", "jpg", "jpeg", "webp"], 
            accept_multiple_files=True
        )
        trigger_word_input = st.text_input(
            "Unique Trigger Word / Token", 
            value=st.session_state.trigger_word,
            help="Unique name for your influencer (e.g. MEGHNA, AVATAR_X)"
        )
        steps = st.slider("Training Steps", min_value=800, max_value=2500, value=1200, step=100)
        train_button = st.button("🚀 Start LoRA Training", type="primary")

    with col2:
        if uploaded_files:
            st.write(f"📸 **Selected {len(uploaded_files)} Images:**")
            preview_cols = st.columns(3)
            for idx, img_file in enumerate(uploaded_files[:6]):
                with preview_cols[idx % 3]:
                    st.image(img_file, use_container_width=True)
            if len(uploaded_files) > 6:
                st.caption(f"+ {len(uploaded_files) - 6} more images")

    if train_button:
        if not uploaded_files or len(uploaded_files) < 8:
            st.error("Please upload at least 8 to 15 high-quality images for good results.")
        else:
            status_box = st.status("Training in progress... This takes 2–4 minutes.", expanded=True)
            try:
                status_box.write("📦 Packaging images into zip...")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for uploaded_file in uploaded_files:
                        zip_file.writestr(uploaded_file.name, uploaded_file.getvalue())
                zip_buffer.seek(0)

                status_box.write("☁️ Uploading dataset to Fal.ai...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                    tmp_file.write(zip_buffer.read())
                    tmp_path = tmp_file.name

                images_data_url = fal_client.upload_file(tmp_path)
                os.remove(tmp_path)

                status_box.write("🔥 Training Flux LoRA on cloud GPU...")
                
                def on_queue_update(update):
                    if isinstance(update, fal_client.InProgress):
                        for log in update.logs:
                            status_box.write(f"📝 {log['message']}")

                result = fal_client.subscribe(
                    "fal-ai/flux-lora-fast-training",
                    arguments={
                        "images_data_url": images_data_url,
                        "trigger_word": trigger_word_input,
                        "steps": steps,
                        "is_style": False
                    },
                    with_logs=True,
                    on_queue_update=on_queue_update
                )

                lora_url = result.get("diffusers_lora_file", {}).get("url", "")
                if lora_url:
                    st.session_state.lora_url = lora_url
                    st.session_state.trigger_word = trigger_word_input
                    status_box.update(label="✅ LoRA Training Completed!", state="complete", expanded=False)
                    st.success("LoRA weights generated successfully!")
                    st.code(lora_url, language="text")
                    st.info("👉 Switch to the **'2. Generate Photos'** tab to start creating images.")
                else:
                    status_box.update(label="Failed to get LoRA file URL", state="error")
                    st.error("Training finished but LoRA URL was not found in response.")

            except Exception as e:
                status_box.update(label="❌ Training failed", state="error")
                st.error(f"Error: {str(e)}")

# TAB 2: GENERATE PHOTOS
with tab2:
    st.subheader("Step 2: Generate Consistent Photos")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        lora_path = st.text_input(
            "LoRA Model URL (auto-filled after training)", 
            value=st.session_state.lora_url,
            help="Direct .safetensors link from training"
        )
        active_trigger = st.text_input("Active Trigger Word", value=st.session_state.trigger_word)
        
        st.markdown("**Quick Preset Prompts:**")
        presets = {
            "Custom": "",
            "Coffee Shop Casual": f"Candid portrait photo of {active_trigger} woman sitting in a modern aesthetic cafe, holding a coffee cup, soft natural morning sunlight, 35mm lens, photorealistic, 8k",
            "Kerala Saree Portrait": f"High quality photo of {active_trigger} woman wearing an elegant traditional Kerala kasavu saree, festival temple background, warm golden hour lighting, smiling gently, highly detailed face",
            "Fitness / Gym": f"Action photo of {active_trigger} woman in athletic gym wear doing workout, modern gym background, cinematic athletic lighting, realistic sweat and skin texture",
            "Studio Fashion": f"Editorial fashion studio photoshoot of {active_trigger} woman, minimalist beige background, dramatic studio rim lighting, high fashion outfit, sharp focus"
        }
        selected_preset = st.selectbox("Choose a preset or write your own:", list(presets.keys()))
        default_prompt = presets[selected_preset] if selected_preset != "Custom" else f"Photo of {active_trigger} woman wearing casual clothes, outdoors, realistic lighting, highly detailed face"
        
        prompt = st.text_area("Prompt (Must include your trigger word):", value=default_prompt, height=120)
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            aspect_ratio = st.selectbox(
                "Image Ratio", 
                ["portrait_4_3", "landscape_4_3", "square_hd"], 
                index=0
            )
            lora_scale = st.slider("LoRA Strength (Scale)", min_value=0.5, max_value=1.3, value=0.95, step=0.05)
        with subcol2:
            num_steps = st.slider("Inference Steps", min_value=20, max_value=40, value=28)
            guidance = st.slider("Guidance Scale", min_value=2.0, max_value=5.0, value=3.5, step=0.1)

        generate_btn = st.button("✨ Generate Photo", type="primary")

    with c2:
        st.write("🖼️ **Generated Result:**")
        if generate_btn:
            if not lora_path:
                st.error("Please provide a LoRA Model URL first.")
            elif not prompt:
                st.error("Please enter a prompt.")
            else:
                with st.spinner("Generating consistent image with Flux..."):
                    try:
                        result = fal_client.subscribe(
                            "fal-ai/flux-lora",
                            arguments={
                                "prompt": prompt,
                                "loras": [{"path": lora_path, "scale": lora_scale}],
                                "image_size": aspect_ratio,
                                "num_inference_steps": num_steps,
                                "guidance_scale": guidance,
                                "enable_safety_checker": True
                            }
                        )
                        images = result.get("images", [])
                        if images:
                            img_url = images[0].get("url")
                            st.image(img_url, use_container_width=True)
                            st.markdown(f"[📥 Open Full Resolution Image]({img_url})")
                        else:
                            st.error("No image returned from Fal API.")
                    except Exception as e:
                        st.error(f"Generation error: {str(e)}")
