# MKII_AI

An Artificial Intelligence agent developed to play Mortal Kombat II (Genesis) using the Stable-Retro library.

This project explores Reinforcement Learning techniques applied to classic video games, enabling agents to learn decision-making strategies based on the game state.

---

## Project Goals

The main objectives of this project are:

* Train an agent to play Mortal Kombat II
* Explore different state representations (RAM vs. image-based observations)
* Evaluate agent performance throughout training
* Provide a foundation for Reinforcement Learning experiments in retro games

---

## Technologies Used

* Python 3
* Stable-Retro
* Stable-Baselines3
* Gymnasium
* NumPy

---

## Development Setup

### Linux / WSL 2

First, install the required system dependencies:

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg freeglut3-dev
```

### Using the Stable-Retro UI

If you want to use the Stable-Retro graphical interface:

```bash
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro

pip3 install -e .

sudo apt-get install capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev zlib1g-dev
sudo apt install cmake

cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY
make -j$(grep -c ^processor /proc/cpuinfo)
```

Run the interface with:

```bash
./gym-retro-integration
```

### Without the UI

If you only need the library:

```bash
pip3 install stable-retro
```

### Clone the Repository

```bash
git clone https://github.com/Antonio-Secchin/MKII_IA.git
cd MKII_IA
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Import the Game ROM

A legal copy of the Mortal Kombat II (Genesis) ROM is required.

Place the ROM file in a folder and run:

```bash
python3 -m retro.import .
```

---

## Research Topics

This project investigates several Reinforcement Learning challenges, including:

* Learning from RAM observations
* Learning from image observations
* Reward engineering
* Agent evaluation and benchmarking

---

## Disclaimer

This repository does not distribute copyrighted game ROMs. Users must provide their own legally obtained copy of Mortal Kombat II.
