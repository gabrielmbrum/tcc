# Ablação da via visual (repeats=3, razão teste/treino=0.25, margem ±0.02)

## 1. F1 macro (média ± desvio entre folds externos)

| configuração | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| deit_tiny/frozen | 0.735 ± 0.127 | 0.956 ± 0.020 | 0.866 ± 0.040 | 0.934 ± 0.038 | 0.713 ± 0.089 |
| deit_tiny/ft | 0.777 ± 0.160 | 0.984 ± 0.032 | 0.953 ± 0.039 | 0.986 ± 0.018 | 0.827 ± 0.093 |
| efficientnet_b0/frozen | 0.903 ± 0.093 | 0.990 ± 0.018 | 0.930 ± 0.035 | 0.961 ± 0.029 | 0.747 ± 0.070 |
| efficientnet_b0/ft | 0.850 ± 0.116 | 0.982 ± 0.036 | 0.988 ± 0.012 | 0.994 ± 0.014 | 0.904 ± 0.058 |
| mobilevit_s/frozen | 0.739 ± 0.133 | 0.742 ± 0.075 | 0.852 ± 0.033 | 0.959 ± 0.034 | 0.734 ± 0.077 |
| mobilevit_s/ft | 0.863 ± 0.119 | 0.990 ± 0.018 | 0.994 ± 0.009 | 0.994 ± 0.011 | 0.901 ± 0.045 |
| topológica bilstm/cum | 0.683 ± 0.157 | 0.936 ± 0.042 | 0.992 ± 0.007 | 0.970 ± 0.023 | 0.905 ± 0.049 |
| topológica bilstm/diff | 0.676 ± 0.159 | 0.939 ± 0.034 | 0.993 ± 0.008 | 0.978 ± 0.017 | 0.902 ± 0.044 |
| topológica mlp/cum | 0.692 ± 0.144 | 0.941 ± 0.032 | 0.994 ± 0.009 | 0.979 ± 0.025 | 0.920 ± 0.043 |
| topológica mlp/diff | 0.758 ± 0.133 | 0.946 ± 0.062 | 0.991 ± 0.012 | 0.983 ± 0.023 | 0.940 ± 0.039 |

Ranking médio entre datasets (descritivo):

| configuração | rank_medio |
|---|---|
| mobilevit_s/ft | 2.20 |
| efficientnet_b0/ft | 3.80 |
| topológica mlp/diff | 4.20 |
| deit_tiny/ft | 4.80 |
| topológica mlp/cum | 4.80 |
| efficientnet_b0/frozen | 5.40 |
| topológica bilstm/diff | 6.40 |
| topológica bilstm/cum | 6.40 |
| deit_tiny/frozen | 8.20 |
| mobilevit_s/frozen | 8.80 |

## 2. Configurações visuais contra `mobilevit_s/ft`

Diferença = F1(configuração) − F1(referência). Holm por dataset.

| vs mobilevit_s/ft | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| deit_tiny/frozen | -0.127 [-0.340, +0.086] p=1.00 | -0.034 [-0.058, -0.009] p=0.04 | -0.128 [-0.176, -0.080] p=0.00 | -0.059 [-0.110, -0.008] p=0.13 | -0.188 [-0.299, -0.077] p=0.01 |
| deit_tiny/ft | -0.086 [-0.345, +0.174] p=1.00 | -0.005 [-0.051, +0.040] p=1.00 | -0.041 [-0.087, +0.005] p=0.16 | -0.008 [-0.032, +0.016] p=0.97 | -0.074 [-0.169, +0.022] p=0.24 |
| efficientnet_b0/frozen | +0.040 [-0.123, +0.203] p=1.00 | -0.000 [-0.031, +0.030] p=1.00 | -0.063 [-0.110, -0.017] p=0.03 | -0.032 [-0.071, +0.006] p=0.30 | -0.153 [-0.254, -0.053] p=0.02 |
| efficientnet_b0/ft | -0.013 [-0.190, +0.164] p=1.00 | -0.008 [-0.048, +0.032] p=1.00 | -0.005 [-0.019, +0.008] p=0.41 eq. | -0.000 [-0.011, +0.011] p=1.00 eq. | +0.003 [-0.091, +0.097] p=0.94 |
| mobilevit_s/frozen | -0.123 [-0.347, +0.100] p=1.00 | -0.248 [-0.353, -0.142] p=0.00 | -0.141 [-0.179, -0.104] p=0.00 | -0.035 [-0.074, +0.004] p=0.30 | -0.166 [-0.259, -0.073] p=0.01 |
Significativas após Holm (p<0.05): 8 de 25.

## 3. `mobilevit_s/ft` contra a via topológica (mesmos folds)

Diferença = F1(visual) − F1(topológica); positiva favorece a via visual. Holm por dataset.

| visual − topológica | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| topológica bilstm/cum | +0.180 [-0.075, +0.434] p=0.47 | +0.054 [-0.001, +0.109] p=0.12 | +0.001 [-0.012, +0.015] p=1.00 eq. | +0.024 [+0.002, +0.046] p=0.14 | -0.005 [-0.079, +0.070] p=1.00 |
| topológica bilstm/diff | +0.187 [-0.097, +0.471] p=0.47 | +0.051 [+0.003, +0.099] p=0.12 | +0.001 [-0.013, +0.015] p=1.00 eq. | +0.016 [-0.002, +0.034] p=0.22 | -0.001 [-0.084, +0.082] p=1.00 |
| topológica mlp/cum | +0.171 [-0.048, +0.390] p=0.47 | +0.049 [+0.008, +0.090] p=0.09 | +0.000 [-0.013, +0.013] p=1.00 eq. | +0.014 [-0.007, +0.035] p=0.34 | -0.019 [-0.100, +0.063] p=1.00 |
| topológica mlp/diff | +0.105 [-0.109, +0.319] p=0.47 | +0.044 [-0.032, +0.119] p=0.23 | +0.003 [-0.014, +0.020] p=1.00 | +0.011 [-0.007, +0.029] p=0.34 | -0.040 [-0.121, +0.042] p=1.00 |
Significativas após Holm (p<0.05): 0 de 20.

## 4. Ganho do ajuste (F1 selecionado − F1 padrão) e configurações selecionadas

| configuração | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| deit_tiny/frozen | 0.003 | -0.010 | 0.004 | -0.013 | -0.011 |
| deit_tiny/ft | -0.009 | -0.003 | -0.021 | -0.003 | -0.045 |
| efficientnet_b0/frozen | -0.008 | 0.005 | -0.004 | -0.010 | -0.007 |
| efficientnet_b0/ft | -0.030 | -0.003 | 0.002 | 0.010 | 0.043 |
| mobilevit_s/frozen | -0.013 | 0.015 | 0.007 | -0.009 | -0.012 |
| mobilevit_s/ft | 0.086 | 0.003 | 0.013 | 0.010 | 0.023 |

- **deit_tiny/frozen**: lr_hd: 0.0001=0.41, 0.0003=0.32, 0.001=0.27; wd: 0.0001=0.41, 0.01=0.59; dropout: 0.1=0.53, 0.3=0.27, 0.5=0.2; bs: 8=0.15, 16=0.43, 32=0.43; config padrão em 20% dos folds; épocas medianas=39; parâmetros=41730; tempo mediano por fold=5s
- **deit_tiny/ft**: lr_bb: 1e-05=0.29, 3e-05=0.15, 0.0001=0.56; lr_hd: 0.0001=0.25, 0.0003=0.56, 0.001=0.19; wd: 0.0001=0.25, 0.01=0.75; dropout: 0.1=0.45, 0.3=0.39, 0.5=0.16; bs: 8=0.08, 16=0.56, 32=0.36; config padrão em 16% dos folds; épocas medianas=16; parâmetros=5566146; tempo mediano por fold=60s
- **efficientnet_b0/frozen**: lr_hd: 0.0001=0.53, 0.0003=0.24, 0.001=0.23; wd: 0.0001=0.53, 0.01=0.47; dropout: 0.1=0.48, 0.3=0.21, 0.5=0.31; bs: 8=0.07, 16=0.63, 32=0.31; config padrão em 31% dos folds; épocas medianas=32; parâmetros=180994; tempo mediano por fold=4s
- **efficientnet_b0/ft**: lr_bb: 1e-05=0.03, 3e-05=0.13, 0.0001=0.84; lr_hd: 0.0001=0.03, 0.0003=0.84, 0.001=0.13; wd: 0.0001=0.03, 0.01=0.97; dropout: 0.1=0.56, 0.3=0.41, 0.5=0.03; bs: 8=0.12, 16=0.55, 32=0.33; config padrão em 3% dos folds; épocas medianas=14; parâmetros=4188542; tempo mediano por fold=83s
- **mobilevit_s/frozen**: lr_hd: 0.0001=0.31, 0.0003=0.35, 0.001=0.35; wd: 0.0001=0.31, 0.01=0.69; dropout: 0.1=0.65, 0.3=0.25, 0.5=0.09; bs: 8=0.11, 16=0.52, 32=0.37; config padrão em 9% dos folds; épocas medianas=34; parâmetros=99074; tempo mediano por fold=6s
- **mobilevit_s/ft**: lr_bb: 1e-05=0.03, 3e-05=0.03, 0.0001=0.95; lr_hd: 0.0003=0.95, 0.001=0.05; wd: 0.01=1.0; dropout: 0.1=0.68, 0.3=0.32; bs: 8=0.19, 16=0.51, 32=0.31; config padrão em 0% dos folds; épocas medianas=14; parâmetros=5036706; tempo mediano por fold=175s
