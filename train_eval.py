from data import train_loader, test_loader, vocab
from model import RNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json

def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01):
    # Khởi tạo loss function và optimizer SGD (không dùng Adam)
    # [Dùng CrossEntropyLoss và optim.SGD]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Vòng lặp huấn luyện
    for epoch in range(epochs):
        model.train()
        for text, labels in train_loader:
            # [Dùng forward, tính loss, backward, cập nhật trọng số bằng SGD]
            optimizer.zero_grad()
            outputs = model(text)       # (batch_size, output_dim)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Đánh giá mô hình
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, labels in test_loader:
            # [Dự đoán và thu thập kết quả]
            outputs = model(text)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

#Thử nghiệm Pretrained vs Scratch
results = {}
for pretrained in [True, False]:
    model = RNNModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=pretrained)
    key = f"RNN_Pretrained={pretrained}"
    acc, f1 = train_and_evaluate(model, train_loader, test_loader)
    results[key] = {"Accuracy": acc, "F1-score": f1}
    print(f"{key} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
