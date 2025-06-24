import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # para mejor matriz de confusión
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)



def entrenar_mlp(X, y, output_path="output/resultados/", epochs=150, patience=5):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # 1. Primera división: entrenamiento (60%) y temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # 2. Segunda división: validación (20%) y prueba (20%) de la parte temporal
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convertir a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    model = MLPModel(input_dim=X.shape[1], output_dim=len(set(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_train = []
    loss_val = []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

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

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping activado.")
            break

    # Restaurar el mejor modelo
    model.load_state_dict(best_model_state)

    # Evaluación en conjunto de validación
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1).cpu().numpy()
        y_true_val = y_val.cpu().numpy()
        acc_val = accuracy_score(y_true_val, val_preds)
        cm_val = confusion_matrix(y_true_val, val_preds)
        print(f"Validation Accuracy final (mejor modelo): {acc_val:.4f}")

    # Guardar resultados
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

    # Gráfica de pérdida
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

    # Matriz de confusión
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.title("Matriz de Confusión en Validación")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mlp_confusion_matrix.png"))
    plt.close()

    return model