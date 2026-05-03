import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),  # 9 state + 1 action
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.net(x)


df = pd.read_csv("logs/transition_log.csv")

state_cols = [c for c in df.columns if c.startswith("s")]
next_cols = [c for c in df.columns if c.startswith("next_s")]

X = df[state_cols].values
actions = df["action"].values.reshape(-1, 1)
X = np.concatenate([X, actions], axis=1)

y = df[next_cols].values

model = Surrogate()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X).float()
y_tensor = torch.tensor(y).float()

for epoch in range(80):
    pred = model(X_tensor)
    loss = loss_fn(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "models/surrogate.pth")
print("Surrogate trained & saved.")