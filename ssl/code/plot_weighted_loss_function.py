import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import font_manager

plt.rcParams["font.family"] = "Rekha"

if __name__ == '__main__':
    # print("List of all fonts currently available in the matplotlib:")
    # print(*font_manager.findSystemFonts(fontpaths=None, fontext='ttf'), sep="\n")

    # Parámetro para la sigmoide
    a = 500 

    # Umbrales
    th = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    # Puntos en el eje x
    points = np.linspace(-0.02, 0.12, 100)

    # Definir fuente Times New Roman
    font = {}

    # Usar un colormap para generar colores automáticamente
    colors = cm.jet(np.linspace(0, 1, len(th)))

    # Crear figura
    plt.figure(figsize=(8, 5))

    # Graficar cada curva con colores del colormap
    for th_i, color in zip(th, colors): 
        loss_plt = 1 / (1 + np.exp(-a * (points - th_i)))
        plt.plot(points, loss_plt, label=f'b = {th_i}', color=color)

    # Etiquetas y título con fuente Times New Roman
    plt.xlabel('Weights', **font)
    plt.ylabel('Sigmoid Weights', **font)
    plt.title('Sigmoid Weighted Loss Curve', **font)

    # Agregar leyenda y grid
    plt.legend()
    plt.grid(True)

    # Guardar imagen
    plt.savefig('sigmoid_weighted_loss_function_colormap.png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('sigmoid_weighted_loss_function_colormap.pdf', bbox_inches='tight', pad_inches=0.1)

    # Limpiar figura
    plt.clf()
