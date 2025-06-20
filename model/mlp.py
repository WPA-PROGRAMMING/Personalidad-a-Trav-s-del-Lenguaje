import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # para mejor matriz de confusión

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.2):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def entrenar_mlp(X, y, output_path="output/resultados/", epochs=50, test_size=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    model = MLPModel(input_dim=X.shape[1], output_dim=len(set(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_train = []
    loss_val = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        loss_train.append(loss.item())
        loss_val.append(val_loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

    # Evaluación final
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1).cpu().numpy()
        y_true = y_val.cpu().numpy()
        acc = accuracy_score(y_true, val_preds)
        print(f"Validation Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_true, val_preds)

    # Guardar modelo
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

    # Guardar curva de pérdidas
    plt.figure()
    plt.plot(loss_train, label="Entrenamiento")
    plt.plot(loss_val, label="Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Curva de Pérdida del MLP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mlp_loss_curve.png"))
    plt.close()

    # Guardar matriz de confusión (heatmap)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.title("Matriz de Confusión en Validación")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mlp_confusion_matrix.png"))
    plt.close()

    return model
