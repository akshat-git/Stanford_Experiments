from setup_models import PlainAutoencoder, CondAutoencoder, CoLaDiffUNet
import torch, tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from form_dataset import InspectDataset

# Load test dataset
test_set = InspectDataset(split="test")
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# Function to evaluate performance
def evaluate_model(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, cond, lbl in tqdm.tqdm(loader, desc="Evaluating..."):
            imgs, cond = imgs.cuda(), cond.cuda()
            loss = model(imgs, cond)
            total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation on each model
ae = PlainAutoencoder().cuda()
cae = CondAutoencoder(cond_dim=2).cuda()
diff = CoLaDiffUNet(cond_dim=2).cuda()

# Load trained weights if needed
# ae.load_state_dict(torch.load('ae_model.pth'))
# cae.load_state_dict(torch.load('cae_model.pth'))
# diff.load_state_dict(torch.load('diff_model.pth'))

print("Evaluating Model A (PlainAutoencoder)...")
ae_loss = evaluate_model(ae, test_loader)
print(f"Model A Loss: {ae_loss}")

print("Evaluating Model B (CondAutoencoder)...")
cae_loss = evaluate_model(cae, test_loader)
print(f"Model B Loss: {cae_loss}")

print("Evaluating Model C (CoLaDiffUNet)...")
diff_loss = evaluate_model(diff, test_loader)
print(f"Model C Loss: {diff_loss}")
