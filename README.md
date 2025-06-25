
# Stable Diffusion from Scratch 🧠🎨

This is an educational implementation of **Stable Diffusion**, a latent text-to-image diffusion model, built from scratch using PyTorch. It demonstrates the end-to-end process of converting text prompts into high-resolution images using a denoising diffusion probabilistic model (DDPM), a CLIP text encoder, and a VAE.

---

## 🧰 Project Structure

```

.
├── attention.py          # Self-attention modules used in the diffusion model
├── clip.py               # CLIP-based text encoder
├── ddpm.py               # DDPM sampler implementation
├── decoder.py            # VAE decoder (not shown, assumed to be present)
├── diffusion.py          # Core U-Net architecture for denoising
├── encoder.py            # VAE encoder
├── model\_loader.py       # Load and prepare pretrained model weights
├── model\_converter.py    # Convert Hugging Face / other checkpoint formats
├── pipeline.py           # Main generation pipeline
├── demo.ipynb            # Jupyter notebook demo to run text-to-image generation
├── data/
│   ├── vocab.json        # Vocabulary for tokenizer
│   ├── merges.txt        # Merge rules for BPE tokenizer
│   └── v1-5-pruned-emaonly.ckpt  # Pretrained Stable Diffusion checkpoint
├── images/
│   └── dog.jpg           # Sample input image for image-to-image generation
├── output/
│   ├── output/           # Folder containing generated output images
│   └── output2/
└── README.md             # You are here 📄

````

---

## 🚀 Features

- **Text-to-Image** generation using diffusion
- **Image-to-Image** transformation with noise control
- Uses **DDPM sampler**
- Classifier-Free Guidance support (CFG)
- Seed control for reproducibility
- CPU / CUDA / MPS support

---

## 📦 Requirements

```bash
pip install torch torchvision
pip install transformers
pip install pillow tqdm
````

Also install `pytorch_lightning` (required for `.ckpt` checkpoint loading):

```bash
pip install pytorch_lightning
```

---

## 📝 Usage

### Run from notebook

Use the Jupyter notebook to test both text-to-image and image-to-image generation:

```bash
jupyter notebook demo.ipynb
```

### Generate image from text prompt

In `demo.ipynb`, set:

```python
prompt = "A cat stretching on the floor, ultra sharp, 8k"
input_image = None
```

### Image-to-image transformation

Uncomment the following lines in the notebook:

```python
input_image = Image.open("../images/dog.jpg")
strength = 0.8  # Lower = closer to input
```

---

## 🖼 Sample Outputs

Outputs are saved in:

* `output/output/`
* `output/output2/`

Example images generated from prompts are stored here.

---

## 📌 Notes

* This project is for **learning and experimentation**.
* Model weights and tokenizer files are expected in the `data/` folder.
* The implementation is simplified but faithful to the core Stable Diffusion pipeline.

---

## 📚 References

* [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
* [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
* [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 👤 Author

**Rohan Shenoy** — [GitHub Profile](https://github.com/rohanshenoy30)

```

Let me know if you'd like badges (e.g., Python version, license) or deployment instructions (e.g., Gradio/Web UI) added!
```
