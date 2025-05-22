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

```bash
pip install tensorflow matplotlib
