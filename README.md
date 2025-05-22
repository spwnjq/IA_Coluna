# 🧠 Classificador de Problemas na Coluna Cervical com Deep Learning

Este projeto utiliza **redes neurais convolucionais (CNN)** para identificar **fraturas na coluna cervical** a partir de imagens médicas, como raios-X.

## 📦 Estrutura dos Dados

O modelo é treinado com imagens organizadas em pastas:

modelo.zip
└── modelo/
├── train/
│ ├── normal/
│ └── fracture/
└── val/
├── normal/
└── fracture/

- `normal/`: imagens sem problemas.
- `fracture/`: imagens com fraturas.


## ⚙️ Como Executar

### 1. Instale as dependências

pip install tensorflow matplotlib

Coloque o arquivo modelo.zip na mesma pasta do script. O arquvo modelo.zip é o obtido como database de imagens em sites como Kaggle;

### 2. Execute o script

python treino_coluna.py


## - Base de dados e Arquivo .h5

- Base = https://drive.google.com/file/d/12Wc8LuRLk2JPjHkBW__UH173WtsQ7uR1/view?usp=sharing
- Arquivo .h5 = https://drive.google.com/file/d/1NKYTj3BJuZ31s3MErCOZywDkDvq9q3bM/view?usp=sharing
