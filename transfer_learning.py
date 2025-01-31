import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import kagglehub
from torch.utils.data import DataLoader

# Baixar o dataset do Kaggle
path = kagglehub.dataset_download("vencerlanz09/sea-animals-image-dataste")
print("Path to dataset files:", path)

# Transformações para pré-processamento
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregar dataset
dataset = datasets.ImageFolder(root=path, transform=data_transforms)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Função para treinar o modelo com Early Stopping
def train_model_with_early_stopping(model, criterion, optimizer, num_epochs=20, patience=5):
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Treinamento
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Perda no conjunto de teste
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, Test Loss: {test_loss:.4f}")
        
        # Verificar se houve melhora na perda
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
            # Salvar o modelo com melhor desempenho
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1
        
        # Parar se não houver melhora por 'patience' épocas consecutivas
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Test Loss: {best_loss:.4f}")
            break

    return model

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar AlexNet pré-treinado
alexnet = models.alexnet(pretrained=True)
alexnet.to(device)

# Feature Extractor
alexnet_feature_extractor = models.alexnet(pretrained=True)
for param in alexnet_feature_extractor.features.parameters():
    param.requires_grad = False
alexnet_feature_extractor.classifier[6] = nn.Linear(4096, len(dataset.classes))
alexnet_feature_extractor.to(device)

# Fine-Tuning
alexnet_finetuning = models.alexnet(pretrained=True)
alexnet_finetuning.classifier[6] = nn.Linear(4096, len(dataset.classes))
alexnet_finetuning.to(device)

# Configuração dos otimizadores e função de perda
criterion = nn.CrossEntropyLoss()
optimizer_feature_extractor = optim.Adam(alexnet_feature_extractor.classifier.parameters(), lr=0.0001)
optimizer_finetuning = optim.Adam(alexnet_finetuning.parameters(), lr=0.0001)

# Treinamento das duas abordagens com Early Stopping
print("Treinando Feature Extractor...")
alexnet_feature_extractor = train_model_with_early_stopping(alexnet_feature_extractor, criterion, optimizer_feature_extractor)

print("Treinando Fine-Tuning...")
alexnet_finetuning = train_model_with_early_stopping(alexnet_finetuning, criterion, optimizer_finetuning)

# Salvar os modelos finais
torch.save(alexnet_feature_extractor.state_dict(), "feature_extractor_3.pth")
torch.save(alexnet_finetuning.state_dict(), "fine_tuning_3.pth")

print("Modelos treinados e salvos com sucesso!")
