# MKII_IA
Um agente desenvolvido para o jogo Mortal Kombat II Genesis

## Configuração para Desenvolvimento

### Windows (WSL 2) e Linux

Para criar o ambiente de desenvolvimento primeiro instale as dependências necessárias:

```sh
sudo apt-get install freeglut3-dev
sudo apt-get update
sudo apt-get install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg
```
Para desenvolver com a interface do retro:

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

Caso não queira usar a interface:

```sh
pip3 install stable-retro
```

Clone o repositório:
```sh
git clone https://github.com/Antonio-Secchin/MKII_IA.git
```

Será necessário conseguir uma versão da ROM de Mortal Kombat II Genesis do Super Nintendo Entertainment System (SNES) (.bin), depois execute na pasta do arquivo
```sh
python3 -m retro.import .
```
