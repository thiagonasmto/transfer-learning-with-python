import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo treinado
@st.cache_resource
def load_model(model_path):
    model = models.alexnet(pretrained=False)
    num_classes = len(torch.load(model_path, map_location=device)['classifier.6.weight'])  # Obtém o número de classes salvo
    model.classifier[6] = nn.Linear(4096, num_classes)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = "fine_tuning_3.pth"  # Alterar para "fine_tuning.pth" se desejar
model = load_model(model_path)

# Definir transformações para pré-processamento
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lista das classes de animais marinhos
class_names = [
    "Seahorse", "Nudibranchs", "Sea Urchins", "Octopus", "Puffers", 
    "Rays", "Whales", "Eels", "Crabs", "Squid", "Corals", "Dolphins", 
    "Seal", "Penguin", "Starfish", "Lobster", "Jelly Fish", "Sea Otter", 
    "Fish", "Shrimp", "Clams"
]

# Interface Streamlit
st.title("Classificação de Animais Marinhos")
st.write("Carregue uma imagem para verificar se é um animal marinho.")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem Carregada", use_container_width=True)
    
    # Pré-processar a imagem
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    # Fazer previsão
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Obter o nome da classe predita
    predicted_class = class_names[predicted.item()]
    
    # Exibir resultado
    st.write(f"**Classe Predita Index:** {predicted.item()}")
    st.write(f"**Classe Predita:** {predicted_class}")
    st.write("Probabilidades:", torch.nn.functional.softmax(outputs, dim=1))
