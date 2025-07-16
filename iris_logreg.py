#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main(args):
    # 1. Load & prep data
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long,   device=device)
    X_val   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val,   dtype=torch.long,   device=device)

    # 3. Model, loss, optimizer
    model = nn.Linear(4, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_val).argmax(dim=1) == y_val).float().mean()
            print(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Loss: {loss.item():.4f} "
                f"Val Acc: {val_acc:.2%}"
            )

    # 5. Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Logistic Regression on GPU")
    parser.add_argument("--epochs",       type=int,   default=100,      help="Number of training epochs")
    parser.add_argument("--lr",           type=float, default=0.1,      help="Learning rate")
    parser.add_argument("--test_size",    type=float, default=0.2,      help="Fraction for validation")
    parser.add_argument("--log_interval", type=int,   default=20,       help="Epochs between logs")
    parser.add_argument("--output",       type=str,   default="model.pt", help="Where to save model")
    args = parser.parse_args()
    main(args)
