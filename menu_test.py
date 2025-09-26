import os
import time
from model_train import *
from model_callback import EvalCallback, SimpleEvalCallback

#graphics = True

def clear():
    """Limpa a tela (Windows/Linux/Mac)."""
    os.system('cls' if os.name == 'nt' else 'clear')

def menu_principal(variables,
    env_space = make_test_env,
    timesteps = 10000,
    device = "cuda",
    ):
    env = env_space(variables)
    callback =  ["Models/", 100, 20, True]   #Diretório, frequência de avaliação, n° de partidas por avaliação, gerar gráfico
    while True:
        clear()
        print("=== MENU PRINCIPAL ===")
        print("1 - Iniciar treinamento")
        print("2 - Iniciar teste (em desenvolvimento)")
        print("3 - Configurações")
        print("4 - Sair")

        print("\n --- Configurações Atuais ---")
        info_model(env_space, timesteps, device)
        print("------------------------------\n")

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            tela_treino(callback, env, env_space, timesteps, device)
            break
        elif escolha == "2":
            tela_teste()
        elif escolha == "3":
            env.close()
            callback, env_space, timesteps, device = tela_config()
            env = env_space(variables)
        elif escolha == "4":
            print("Saindo...")
            time.sleep(1)
            break
        else:
            print("Opção inválida!")
            time.sleep(1)

def tela_treino(callbackconfig, env, env_space, timesteps, device):
    clear()
    print("=== Começando o treinamento ===")
    print(" --- Configurações Atuais ---")
    info_model(env_space, timesteps, device)
    print(" -----------------------------")
    print(" --- Configurações Callback ---")
    info_callback(*callbackconfig)
    print("------------------------------")
    time.sleep(3)
    clear()

    vec_env = DummyVecEnv([lambda:env])
    callback = EvalCallback(eval_env= vec_env, save_dir=callbackconfig[0], 
                    n_eval_episodes=callbackconfig[1], eval_freq=callbackconfig[2], generate_graphic=callbackconfig[3])
    if env_space == make_env_image:
        model = PPO("CnnPolicy", vec_env, verbose=0, device=device)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=0, device=device)
    model.learn(total_timesteps=timesteps, progress_bar=True, callback= callback)


def tela_teste():
    clear()
    print("=== Começando o teste ===")
    time.sleep(1)

def tela_config(env_space = make_test_env,
    timesteps = 10000,
    device = "cuda"):
    callback = ["Models/", 100, 20, True]   #Diretório, frequência de avaliação, n° de partidas por avaliação, gerar gráfico
    while True:
        clear()
        print("=== CONFIGURAÇÕES ===")
        print("1 - Alterar opção Environment Observation Space")
        print("2 - Alterar quantidade de Timesteps")
        print("3 - Configurações de Callback")
        print("4 - Alterar Device (cuda/cpu)")
        print("5 - Voltar")

        print("\n--- Configurações Atuais ---")
        info_model(env_space, timesteps, device)
        print("------------------------------")
        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            while True:
                clear()
                print("=== Environment Observation Space ===")
                print("1 - RAM")
                print("2 - RAM Reduzida + Ações")
                print("3 - Imagem")
                print("4 - Voltar")

                escolha = input("Escolha uma opção: ")

                if escolha == "1":
                    print("Environment RAM selecionado!")
                    env_space = make_env#Seria as funções de make_env()
                    time.sleep(1)
                    break
                elif escolha == "2":
                    print("Environment RAM Reduzida + Ações selecionado!")
                    env_space = make_test_env
                    time.sleep(1)
                    break
                elif escolha == "3":
                    print("Environment Imagem selecionado!")
                    env_space = make_env_image
                    time.sleep(1)
                    break
                elif escolha == "4":
                    break
            time.sleep(1)
        elif escolha == "2":
            while True:
                clear()
                print("=== Quantidade de Timesteps ===")

                escolha = input("Escolha um valor: ")
                if escolha.isdigit() and int(escolha) > 0:
                    timesteps = int(escolha)
                    print(f"Timesteps alterados para {escolha}!")
                    break
                else:
                    print("Por favor, insira um número válido.")
            time.sleep(1)
        elif escolha == "3":
            #Mudar para já mandar o env criado e fechar quando mudar para outro tipo
            callback = tela_config_callback(*callback)
            time.sleep(1)
        elif escolha == "4":
            if device == "cuda":
                device = "cpu"
            else:
                device = "cuda"
            print(f"Device alterado para {device}!")
            time.sleep(1)
        elif escolha == "5":
            return callback, env_space, timesteps, device
        else:
            print("Opção inválida!")
            time.sleep(1)


def tela_config_callback(save_dir:str = "Models", eval_freq:int = 100, n_eval_episodes:int=20, graphics:bool = True):
    while True:
        clear()
        print("=== CONFIGURAÇÕES DE CALLBACK ===")
        print("1 - Alterar diretório de salvamento")
        print("2 - Alterar frequencia de avalição")
        print("3 - N° de partidas para avaliar o modelo")
        print("4 - Gerar gráfico de recompensas (T/F)")
        print("5 - Voltar")
        
        print("\n--- Configurações Atuais ---")
        info_callback(save_dir, eval_freq, n_eval_episodes, graphics)
        print("------------------------------")

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            while True:
                clear()
                print("=== Diretório de Salvamento ===")
                escolha = input("Escolha o diretório de saída: ")

                if escolha != "":
                    save_dir = escolha
                    print(f"Diretório alterado para {escolha}!")
                    time.sleep(1)
                    break
            time.sleep(1)
        elif escolha == "2":
            while True:
                clear()
                print("=== Frequencia de avaliação ===")

                escolha = input("Escolha um valor: ")
                if escolha.isdigit() and int(escolha) > 0:
                    eval_freq = int(escolha)
                    print(f"Frequencia de avaliação alterado para {escolha}!")
                    break
                else:
                    print("Por favor, insira um número válido.")
            time.sleep(1)
        elif escolha == "3":
            while True:
                clear()
                print("=== N° de partidas para avaliar o modelo ===")

                escolha = input("Escolha um valor: ")
                if escolha.isdigit() and int(escolha) > 0:
                    n_eval_episodes = int(escolha)
                    print(f"N° de partidas de avaliação alterados para {escolha}!")
                    break
                else:
                    print("Por favor, insira um número válido.")
            time.sleep(1)
        elif escolha == "4":
            graphics = not graphics
            print(f"Gerar gráfico de recompensas alterado para {graphics}!")
            time.sleep(1)
        elif escolha == "5":
            return [save_dir, eval_freq, n_eval_episodes, graphics]
        else:
            print("Opção inválida!")
            time.sleep(1)


def info_model(env_space, timesteps, device):
    model = "PPO"
    police = ""
    if env_space == make_env_image:
        env_space_name = "Image"
        police = "CnnPolicy"
    elif env_space == make_test_env:
        env_space_name = "Reduced RAM + Actions"
        police = "MlpPolicy" 
    else:
        env_space_name = "RAM"
        police = "MlpPolicy"
    print(f" Model: {model}\n Police: {police}\n Observation Space: {env_space_name}\n Timesteps: {timesteps}\n Device: {device}")

def info_callback(save_dir, frq_aval, n_parts, graphics):
    print(f" Save_dir = {save_dir} \n Frequencia de  avaliação: {frq_aval}\n N° de partidas por avaliação: {n_parts}\n Gerar Gráficos: {graphics}")

if __name__ == "__main__":
    #graphics = True
    variables = [
        "Tempo",
        "health",
        "enemy_health",
        "rounds_won",
        "enemy_rounds_won",
        "wins",
        "x_position",
        "y_position",
        "enemy_x_position",
        "enemy_y_position",
        "Block_aliado",
        "Block_inimigo",
        "Projectile_x",
        "Projectile_y",
        "Projectile_Position_Enemy_x",
        "Projectile_Position_Enemy_y",
    ]

    menu_principal(variables)
