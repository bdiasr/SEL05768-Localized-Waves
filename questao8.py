import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from matplotlib import cm

# Parâmetros
k = 2*np.pi       # Número de onda (em termos de 1/lambda)
Ns = 9            # Número de fontes
Nr = 9            # Número de receptores
h = 4             # Comprimento vertical (em termos de lambda)
delta_z = 1/2     # Espaçamento vertical (em termos de lambda)
L = 5             # Comprimento horizontal (em termos de lambda)
g_SR = -4*np.pi*L # Fator g_SR

# Cálculo das coordenadas y e z das fontes e dos receptores
S_y = np.arange(0.5, 4.5+0.1, delta_z)
S_z = 0
R_y = S_y.copy()
R_z = L

# Matriz G
G = np.zeros((Nr, Ns), dtype=complex)

# Cálculo dos coeficientes da matriz G
for i in range(Nr):
    for j in range(Ns):
        r = np.sqrt((R_y[i]-S_y[j])**2 + (R_z-S_z)**2)
        G[i,j] = -1/(4*np.pi)*np.exp(1j*k*r)/r

G_h = np.conj(G.T)  # Matriz adjunta hermitiana de G

# Cálculo dos autovalores e autovetores
M = g_SR**2 * G_h @ G
autovalores_S, autovetores_S = np.linalg.eig(M)

# Ordenar em ordem decrescente
indices = np.argsort(-autovalores_S)
autovalores_S = autovalores_S[indices]
autovetores_S = autovetores_S[:, indices]

modes_to_plot = [0, 1, 2]  # Python é zero-based

# Definindo o domínio
z = np.linspace(0.1, 5, 200)
y = np.linspace(0, 5, 200)
Z, Y = np.meshgrid(z, y)

fig = plt.figure(figsize=(10,8))
fig.patch.set_facecolor('white')

# Definindo o GridSpec: 3 linhas (modos) x 3 colunas (amplitude, fase, onda)
gs = gridspec.GridSpec(len(modes_to_plot), 3, width_ratios=[1, 1, 2], wspace=0.1, hspace=0.2)


for idx, mode_index in enumerate(modes_to_plot):
    h_j = autovetores_S[:, mode_index]

    # Ajuste de fase global: fonte central com fase zero
    s_center = (Ns)//2  # índice da fonte central (zero-based)
    phase_center = np.angle(h_j[s_center])
    h_j = h_j * np.exp(-1j*phase_center)

    # Cálculo do campo phi
    phi = np.zeros_like(Z, dtype=complex)
    for q in range(Ns):
        R = np.sqrt((Y - S_y[q])**2 + (Z - S_z)**2)
        phi += -1/(4*np.pi)*np.exp(1j*k*R)*h_j[q]/R
    phi = phi.T

    # Multiplica por sqrt(|Y|)
    phi = phi * np.sqrt(np.abs(Y))

    # Cálculo do campo nas fontes
    fontes = np.zeros(Ns, dtype=complex)
    for s in range(Ns):
        for q in range(Ns):
            if q != s:
                R = np.sqrt((S_y[s] - S_y[q])**2 + (S_z - S_z)**2)
                fontes[s] += -1/(4*np.pi)*np.exp(1j*k*R)*h_j[q]/R

    # Vetores de amplitude e fase das fontes
    amplitude = fontes.real
    fase = -np.angle(fontes)
    fase -= fase[4]

    # Configurar colormap para amplitudes (jet)
    cmap = cm.jet
    norm = Normalize(vmin=min(amplitude), vmax=max(amplitude))
    colors = cmap(norm(amplitude))

    # Adicionar subtítulos "a)", "b)" e "c)"
    labels = [r'a) Modo 1', r'b) Modo 2', r'c) Modo 3']
    fig.text(0.02, 0.88 - idx * 0.3, labels[idx], fontsize=12, fontweight='bold')

    # ---- Plot das amplitudes ----
    ax_amp = plt.subplot(gs[idx, 0])
    n = len(amplitude)
    y_positions = np.arange(n)

    if idx != 1:
        ax_amp.scatter(amplitude, y_positions, c=colors, s=20)
    else:
        amplitude -= min(amplitude)
        ax_amp.scatter(amplitude, y_positions, c=colors, s=20)

    buffer = 5e-2  # Margem extra para evitar cortes
    ax_amp.set_xlim(min(amplitude) - buffer, max(amplitude) + buffer)
    ax_amp.axvline(0, color='k', linestyle='-')  # Linha central em x=0
    ax_amp.set_aspect(0.05)
    ax_amp.set_xticks([0])
    ax_amp.set_yticks([])
    ax_amp.invert_yaxis()
    ax_amp.xaxis.tick_top()
    ax_amp.tick_params(bottom=False)
    ax_amp.spines['right'].set_visible(False)  # Remover a borda direita
    ax_amp.spines['left'].set_visible(False)   # Remover a borda esquerda
    ax_amp.spines['bottom'].set_visible(False) # Remover a borda inferior
    if idx == 0:
        ax_amp.set_title(r'Amplitude da fonte', fontsize=12)


    # ---- Plot das fases ----
    ax_phase = plt.subplot(gs[idx, 1])

    if idx == 0:
        fase[:4] -= (min(fase[:4]) - 1*max(fase[:4]))
        fase[5:] += 0.5*min(fase[5:])
   
    ax_phase.scatter(fase, S_y, c=colors, s=20)
    ax_phase.axvline(0, color='k', linestyle='-')  # Linha central em x=0
    ax_phase.axvline(np.pi, color='k', linestyle='-', alpha=0) 
    ax_phase.axvline(2 * np.pi, color='k', linestyle='--', alpha=0)
    ax_phase.set_aspect(3)

    if idx != 2:
        ax_phase.set_xticks([0, np.pi, 2*np.pi])
        ax_phase.set_xticklabels(['0', 'π', '2π'])
    else:
        ax_phase.set_xlim([-np.pi, np.pi])
        ax_phase.set_xticks([-np.pi, 0, np.pi])
        ax_phase.set_xticklabels(['-π', '0', 'π'])
    
    ax_phase.xaxis.tick_top()
    ax_phase.tick_params(bottom=False)  # Remover ticks inferiores
    ax_phase.tick_params(left=False, labelleft=False)
    ax_phase.spines['right'].set_visible(False)  # Remover a borda direita
    ax_phase.spines['left'].set_visible(False)   # Remover a borda esquerda
    ax_phase.spines['bottom'].set_visible(False) # Remover a borda inferior
    if idx == 0:
        ax_phase.set_title(r'Source phase', fontsize=12)

    ax = plt.subplot(gs[idx, 2])
    im = ax.imshow(np.real(phi.T), extent=[z.min(), z.max(), y.min(), y.max()],
                   origin='lower', aspect='equal', cmap='jet')
    
    # Plotar posições das fontes e receptores
    z_positions = np.zeros_like(S_y) - 0.1  # 0.1 fora do domínio
    ax.plot(z_positions, S_y, 'ko', markerfacecolor='k', markersize=3)
    ax.plot((5-0.1)*np.ones_like(R_y), R_y, 'wo', markerfacecolor='w', markersize=3)
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(y.min(), y.max())

    # Removendo as bordas
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ajustar os limites para que as bolinhas apareçam:
    ax.set_xlim(-0.2, 5)
    ax.set_ylim(0.1, 5)

    ax.set_xticks([])
    ax.set_yticks([])


    if idx == 0:
        ax.set_title(r'Onda Resultante (Real($ \Phi $))', fontsize=12)
    


plt.suptitle(' ', fontsize=14)
#plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=1.0)  # Espaçamento otimizado
#plt.subplots_adjust(wspace=0.3, hspace=0.2)      # Ajuste fino dos espaços
plt.show()
