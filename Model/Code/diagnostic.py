import torch
import time

device = torch.device("cuda")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create dummy model and data
model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6
).to(device).train()

x = torch.randn(32, 512, 512, device=device)
y = torch.randint(0, 512, (32, 512), device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Warmup
for _ in range(10):
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, 512), y.view(-1))
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()

# Speed test
steps = 100
t0 = time.time()
for i in range(steps):
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, 512), y.view(-1))
    loss.backward()
    optimizer.step()
    if i % 20 == 0:
        print(f"Step {i}")
torch.cuda.synchronize()
t1 = time.time()

print(f"\n{steps} steps in {t1-t0:.1f}s = {steps/(t1-t0)*60:.0f} steps/min")
print(f"Expected: ~300-500 steps/min on RTX 4060")