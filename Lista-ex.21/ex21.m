% Parâmetros iniciais
lambda = 0.632e-6;      % Comprimento de onda (m)
omega0 = 2.98e15;       % Frequência angular (rad/s)
c = 3e8;                % Velocidade da luz (m/s)
L = 0.5;                % Intervalo longitudinal (m)
Q = 0.9998*omega0/c;    % Parâmetro Q
N = 20;                 % Número de termos na soma

% Função F(z) definida no artigo
z = linspace(0, L, 1000);
F = zeros(size(z));
l1 = L/10; 
l2 = 3*L/10;
l3 = 4*L/10; 
l4 = 6*L/10;
l5 = 7*L/10; 
l6 = 9*L/10;

% Definindo os valores de F(z)
F(z >= l1 & z <= l2) = -4 * (z(z >= l1 & z <= l2) - l1) .* (z(z >= l1 & z <= l2) - l2) / (l2 - l1)^2;
F(z >= l3 & z <= l4) = 1;
F(z >= l5 & z <= l6) = -4 * (z(z >= l5 & z <= l6) - l5) .* (z(z >= l5 & z <= l6) - l6) / (l6 - l5)^2;

% Cálculo dos coeficientes An
An = zeros(1, 2*N+1);
for n = -N:N
    An(n + N + 1) = (1/L) * trapz(z, F .* exp(-1i * 2 * pi * n * z / L));
end

% Construindo a Frozen Wave no eixo longitudinal (ρ = 0)
FW = zeros(size(z));
for n = -N:N
    FW = FW + An(n + N + 1) * exp(1i * 2 * pi * n * z / L);
end
FW = abs(FW).^2;

% Figura 1(a): Comparação de F(z) e FW
figure;
plot(z, F, 'k-', 'LineWidth', 1.5); hold on;
plot(z, FW, 'r--', 'LineWidth', 1.5);
xlabel('z (m)'); ylabel('|Ψ|^2');
legend('F(z)', 'Frozen Wave');
title('Comparação entre as intensidades de F(z) e FW');
grid on;

% Figura 1(b): Representação 3D da intensidade
rho = linspace(-5e-4, 5e-4, 200);
[Rho, Z] = meshgrid(rho, z);
FW_3D = zeros(size(Rho));
for n = -N:N
    k_rho_n = sqrt((omega0 / c)^2 - (Q + 2 * pi * n / L)^2);
    FW_3D = FW_3D + An(n + N + 1) * besselj(0, k_rho_n * Rho) .* exp(1i * 2 * pi * n * Z / L);
end
FW_3D = abs(FW_3D).^2;

% Plot 3D
figure;
mesh(Z, Rho* 1e4, FW_3D);
custom_gray = linspace(0.4, 1, 500)' * [1, 1, 1];
colormap(custom_gray);  
xlabel('z (m)');
ylabel('ρ(m) \times 10^{-4}');
zlabel('|Ψ|^2');
title('Intensidade da Frozen Wave');
grid on;
