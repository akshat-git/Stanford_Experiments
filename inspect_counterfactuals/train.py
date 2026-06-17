from form_dataset import InspectDataset
from setup_models import PlainAutoencoder, CondAutoencoder, CoLaDiffUNet
import torch, torch.optim as optim, tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Parameters
COND_DIM = 2   # Only 2 treatment options: anticoagulation and thrombolysis

# Build the dataset
train_set = InspectDataset(split="train")
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)

# --- Model A ---
ae = PlainAutoencoder().cuda()
opt_ae = optim.Adam(ae.parameters(), 1e-3)
for epoch in range(5):
    for imgs, _, _ in train_loader:
        imgs = imgs.cuda()
        opt_ae.zero_grad()
        loss = F.mse_loss(ae(imgs), imgs)
        loss.backward()
        opt_ae.step()

# --- Model B ---
cae = CondAutoencoder(cond_dim=COND_DIM).cuda()
opt_cae = optim.Adam(cae.parameters(), 1e-3)
for epoch in range(5):
    for imgs, cond, _ in train_loader:
        imgs, cond = imgs.cuda(), cond.cuda()
        opt_cae.zero_grad()
        loss = F.mse_loss(cae(imgs, cond), imgs)
        loss.backward()
        opt_cae.step()

# --- Model C (CoLa‑diff) ---
diff = CoLaDiffUNet(cond_dim=COND_DIM).cuda()
opt_diff = optim.Adam(diff.parameters(), 1e-4)
for epoch in range(5):
    for imgs, cond, _ in train_loader:
        imgs, cond = imgs.cuda(), cond.cuda()
        opt_diff.zero_grad()
        loss = diff(imgs, cond)
        loss.backward()
        opt_diff.step()
