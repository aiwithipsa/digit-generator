import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# Load model
device = torch.device("cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

# Web App UI
st.title("🧠 MNIST Digit Generator")
digit = st.selectbox("Select a digit (0–9):", list(range(10)))

if st.button("Generate"):
    noise = torch.randn(5, 100)
    labels = torch.tensor([digit]*5)
    with torch.no_grad():
        imgs = G(noise, labels)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(imgs[i][0], cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
