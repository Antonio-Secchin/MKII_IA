import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ===== CONFIGURAÇÕES =====
csv_paths = [
    "Models/RAM_Scorpion_Long_att/RolloutModels/RewardsGraf.csv",
    "Models/RAM_Scorpion_Long_att_2/RolloutModels/RewardsGraf.csv",
    "Models/RAM_Scorpion_Long_att_3/RolloutModels/RewardsGraf.csv",
    "Models/RAM_Scorpion_Long_att_4/RolloutModels/RewardsGraf.csv",
    "Models/RAM_Scorpion_Long_att_5/RolloutModels/RewardsGraf.csv",
]

graph_path = "aaaaaa.png"
rollout_max = 7300  # valor final de cada arquivo

# ===== ACUMULADORES =====
x_total = []
y_total = []

# ===== LEITURA EM LOOP =====
for i, csv_path in enumerate(csv_paths):
    df = pd.read_csv(csv_path)

    # eixo X original (100 até 7300)
    x = df["rollout"].to_numpy()

    # offset para continuar o gráfico anterior
    x_offset = i * rollout_max
    x = x + x_offset

    # reward: remove colchetes e converte
    y = (
        df["reward"]
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .astype(float)
        .to_numpy()
    )

    x_total.append(x)
    y_total.append(y)

# concatena tudo como um treino único
x_total = np.concatenate(x_total)
y_total = np.concatenate(y_total)

# ===== PLOT =====
plt.figure(figsize=(10, 6))

if len(x_total) >= 4:
    x_smooth = np.linspace(x_total.min(), x_total.max(), 500)
    spline = make_interp_spline(x_total, y_total, k=3)
    y_smooth = spline(x_smooth)

    plt.plot(
        x_smooth,
        y_smooth,
        linewidth=2,
        label="Recompensa Média (Treino Contínuo)"
    )
else:
    plt.plot(
        x_total,
        y_total,
        marker="o",
        linestyle="-",
        label="Recompensa Média"
    )

plt.title("Evolução da Recompensa Média (Treino Contínuo)")
plt.xlabel("Rollouts Realizados")
plt.ylabel("Recompensa Média")
plt.grid(True)
plt.legend()
plt.savefig(graph_path)
plt.close()


# ===== SALVAR CSV CONSOLIDADO =====
output_csv_path = "RewardsGraf_RAM.csv"

df_out = pd.DataFrame({
    "rollout": x_total.astype(int),
    # volta exatamente para o formato "[valor]"
    "reward": [f"[{v}]" for v in y_total]
})

df_out.to_csv(output_csv_path, index=False)