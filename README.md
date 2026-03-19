# MKII_IA
Agente de Inteligência Artificial desenvolvido para jogar Mortal Kombat II (Genesis) utilizando a biblioteca stable-retro.

O projeto explora técnicas de Aprendizado por Reforço (Reinforcement Learning) aplicadas a jogos clássicos, permitindo treinar agentes capazes de tomar decisões com base no estado do jogo.

---

## Objetivo

O objetivo deste projeto é:

- Treinar um agente para jogar Mortal Kombat II
- Explorar diferentes representações de estado (RAM vs imagem)
- Avaliar desempenho do agente ao longo do tempo
- Servir como base para experimentos com RL em jogos retrô

---

## 🛠️ Tecnologias utilizadas

- Python 3
- Stable-Retro
- Stable-Baselines 3
- Gymnasium
- NumPy
<!-- Adicione aqui outras libs como PyTorch, se usar -->

---


## Configuração para Desenvolvimento

### Windows (WSL 2) e Linux

Para criar o ambiente de desenvolvimento primeiro instale as dependências necessárias:

```sh
sudo apt-get install freeglut3-dev
sudo apt-get update
sudo apt-get install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg
```
#### Para desenvolver com a interface do retro:

```sh
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro
pip3 install -e .
sudo apt-get install capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev zlib1g-dev
sudo apt  install cmake
cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY
make -j$(grep -c ^processor /proc/cpuinfo)
```

Para executar a interface do retro:

```sh
./gym-retro-integration
```

#### Caso não queira usar a interface:

```sh
pip3 install stable-retro
```

Clone o repositório:
```sh
git clone https://github.com/Antonio-Secchin/MKII_IA.git
```

Instale as bibliotecas necessárias:

```sh
pip install -r requirements.txt
```

Será necessário conseguir uma versão da ROM de Mortal Kombat II Genesis do Super Nintendo Entertainment System (SNES) (.bin), depois execute na pasta do arquivo
```sh
python3 -m retro.import .
```
