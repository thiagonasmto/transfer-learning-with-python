# 🚀 Transfer Learning com AlexNet: Feature Extractor vs Fine-Tuning

Este repositório contém o código utilizado no artigo do Medium sobre **Transfer Learning**, onde comparamos as abordagens **Feature Extractor** e **Fine-Tuning** usando a arquitetura **AlexNet** para classificar imagens de animais marinhos. Além disso, incluímos uma aplicação **Streamlit** para testar os modelos treinados.

## 📌 Conteúdo do Repositório

- Código para treinamento do **Feature Extractor**
- Código para treinamento do **Fine-Tuning**
- Script para testar os modelos com **Streamlit**
- Dataset utilizado: [Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)

## 🛠️ Tecnologias Utilizadas

- **Python**
- **PyTorch**
- **Torchvision**
- **Streamlit**
- **Kaggle API**

## 📥 Como Executar o Projeto

### 1️⃣ Clone o Repositório

```bash
git clone https://github.com/thiagonasmto/transfer-learning-with-python.git
cd transfer-learning-alexnet
```

### 2️⃣ Instale as Dependências

```bash
pip install -r requirements.txt
```

### 3️⃣ Baixe o Dataset do Kaggle

Antes de executar o código, certifique-se de configurar suas credenciais do Kaggle para baixar o dataset:

```python
import kagglehub
path = kagglehub.dataset_download("vencerlanz09/sea-animals-image-dataste")
print("Path to dataset files:", path)
```

### 4️⃣ Treine os Modelos

Execute o script de treinamento para Feature Extractor e Fine-Tuning:

```bash
python transfer_learning.py
```

### 5️⃣ Execute a Aplicação Streamlit

```bash
streamlit python -m streamlit run transfer_learning_aplication.py
```

## 🔗 Link para o Artigo no Medium

📖 Leia o artigo completo no Medium: **[Clique aqui!]**(https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)