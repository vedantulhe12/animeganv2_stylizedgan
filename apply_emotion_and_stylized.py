import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
from model import Generator as AnimeGANGenerator
from models.psp import pSp
from utils.common import tensor2im

# -------------------------------
# Config
# -------------------------------
E4E_CHECKPOINT = r"C:\Users\Vedant\Desktop\encoder4editing\pretrained_models\e4e_ffhq_encode.pt"
ANIMEGAN_CHECKPOINT = r"weights/paprika.pt"
BOUNDARY_PATH = r"boundaries/boundary_happy.npy"  # change to match emotion
INPUT_IMAGE = r"C:\Users\Vedant\Desktop\animegan2-pytorch\inputs images\7acb0a06072b15d99b3989f7df009146_69999.png"  # path to input human face
OUTPUT_PATH = "results/output_happy.png"
INTENSITY = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load e4e model
# -------------------------------
from argparse import Namespace
opts = Namespace(
    checkpoint_path=E4E_CHECKPOINT,
    device=DEVICE,
    encoder_type='Encoder4Editing',
    start_from_latent_avg=True,
    input_nc=3,
    n_styles=18,
    stylegan_size=1024,
    is_train=False,
    learn_in_w=False,
    output_size=1024,
    id_lambda=0,
    lpips_lambda=0,
    l2_lambda=1,
    w_discriminator_lambda=0,
    use_w_pool=False,
    w_pool_size=50,
    use_ballholder_loss=False,
    optim_type='adam',
    batch_size=1,
    resize_outputs=False
)
encoder = pSp(opts).to(DEVICE).eval()

# -------------------------------
# Preprocess Image
# -------------------------------
image = Image.open(INPUT_IMAGE).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# -------------------------------
# Latent Encoding & Editing
# -------------------------------
with torch.no_grad():
    _, latent = encoder(img_tensor, return_latents=True)
    boundary = torch.from_numpy(np.load(BOUNDARY_PATH)).float().to(DEVICE)
    for i in range(4, 9):
        latent[:, i, :] += INTENSITY * boundary

    edited_tensor = encoder(latent, input_code=True, resize=True)
    edited_image = tensor2im(edited_tensor[0])  # to PIL

# -------------------------------
# Stylize with AnimeGANv2
# -------------------------------
animegan = AnimeGANGenerator().to(DEVICE)
animegan.load_state_dict(torch.load(ANIMEGAN_CHECKPOINT, map_location=DEVICE))
animegan.eval()

face_tensor = to_tensor(edited_image).unsqueeze(0).to(DEVICE) * 2 - 1
with torch.no_grad():
    output = animegan(face_tensor).cpu().squeeze(0).clamp(-1, 1)
    output = output * 0.5 + 0.5
    anime_pil = to_pil_image(output)

# -------------------------------
# Save Output
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
anime_pil.save(OUTPUT_PATH)
print(f"✅ Saved stylized image with emotion at: {OUTPUT_PATH}")
