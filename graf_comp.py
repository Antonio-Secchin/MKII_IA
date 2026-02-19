import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline

# =========================
# CONFIGURAÇÕES
# =========================

#CSV_DIR = Path("Models/results/Jax")

CSV_DIR = Path("")

# MODELS = {
#     "RAM": "Jax_RAM.csv",
#     "RAM com golpes especiais": "Jax_RAM_Action.csv",
#     "RAM reduzida": "Jax_Info.csv",
#     "RAM reduzida com golpes especiais": "Jax_Red_Old.csv",
#     "Imagem": "Jax_IMG.csv"
#     # até 8 modelos
# }

MODELS = {
    "RAM": "RewardsGraf_RAM.csv",
    "RAM reduzida com golpes especiais": "RewardsGraf_RAM_RED.csv",
    # até 8 modelos
}

COLORS = [
    "#1f77b4",  # azul
    "#ff7f0e",  # laranja
    "#2ca02c",  # verde
    "#d62728",  # vermelho
    "#9467bd",  # roxo
    "#8c564b",  # marrom
    "#e377c2",  # rosa
    "#7f7f7f",  # cinza
]

OUTPUT_IMAGE = "comparacao_modelos_teste.png"

# =========================
# FUNÇÕES AUXILIARES
# =========================

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    # "[-825.2]" -> -825.2
    df["reward"] = df["reward"].apply(
        lambda x: ast.literal_eval(x)[0]
        if isinstance(x, str)
        else float(x)
    )

    return df


def spline_curve(x, y, n_points=1000):
    """
    Aplica spline cúbica se houver pontos suficientes.
    Caso contrário, retorna os dados originais.
    """
    if len(x) < 4:
        return x, y

    x = np.asarray(x)
    y = np.asarray(y)

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)

    return x_smooth, y_smooth


# =========================
# PLOT
# =========================

plt.figure(figsize=(12, 6))

for idx, (model_name, csv_file) in enumerate(MODELS.items()):
    csv_path = CSV_DIR / csv_file
    df = load_csv(csv_path)

    x = df["rollout"].values
    y = df["reward"].values

    x_plot, y_plot = spline_curve(x, y)

    plt.plot(
        x_plot,
        y_plot,
        label=model_name,
        color=COLORS[idx],
        linewidth=2
    )

# =========================
# ESTÉTICA DO GRÁFICO
# =========================

plt.title("Evolução da Recompensa Média por Modelo", fontsize=14)
plt.xlabel("Rollouts Realizados", fontsize=12)
plt.ylabel("Recompensa Média", fontsize=12)

plt.grid(True, alpha=0.3)
plt.legend(title="Modelos", fontsize=10)

plt.tight_layout()

# =========================
# SALVAR IMAGEM
# =========================

plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.close()

print(f"Gráfico salvo em: {OUTPUT_IMAGE}")
