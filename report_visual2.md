# Ablação da via visual (repeats=3, configurações por tarefa=12, razão teste/treino=0.25, margem ±0.02)

## 1. F1 macro (média ± desvio entre folds externos)

| configuração | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| deit_tiny/frozen | 0.711 ± 0.142 | 0.956 ± 0.020 | 0.863 ± 0.046 | 0.937 ± 0.038 | 0.721 ± 0.081 |
| deit_tiny/ft_aug | 0.726 ± 0.185 | 0.985 ± 0.028 | 0.968 ± 0.028 | 0.990 ± 0.012 | 0.854 ± 0.040 |
| efficientnet_b0/frozen | 0.917 ± 0.091 | 0.987 ± 0.024 | 0.926 ± 0.028 | 0.973 ± 0.022 | 0.756 ± 0.061 |
| efficientnet_b0/ft_aug | 0.912 ± 0.083 | 0.984 ± 0.024 | 0.989 ± 0.010 | 0.989 ± 0.020 | 0.896 ± 0.068 |
| mobilevit_s/frozen | 0.743 ± 0.122 | 0.708 ± 0.101 | 0.852 ± 0.051 | 0.971 ± 0.029 | 0.745 ± 0.065 |
| mobilevit_s/ft_aug | 0.852 ± 0.120 | 0.987 ± 0.019 | 0.983 ± 0.016 | 0.987 ± 0.024 | 0.917 ± 0.047 |
| mobilevit_s/ft_noaug | 0.859 ± 0.100 | 0.969 ± 0.036 | 0.985 ± 0.015 | 0.997 ± 0.008 | 0.899 ± 0.036 |
| topológica bilstm/cum | 0.683 ± 0.157 | 0.936 ± 0.042 | 0.992 ± 0.007 | 0.970 ± 0.023 | 0.905 ± 0.049 |
| topológica bilstm/diff | 0.676 ± 0.159 | 0.939 ± 0.034 | 0.993 ± 0.008 | 0.978 ± 0.017 | 0.902 ± 0.044 |
| topológica mlp/cum | 0.692 ± 0.144 | 0.941 ± 0.032 | 0.994 ± 0.009 | 0.979 ± 0.025 | 0.920 ± 0.043 |
| topológica mlp/diff | 0.758 ± 0.133 | 0.946 ± 0.062 | 0.991 ± 0.012 | 0.983 ± 0.023 | 0.940 ± 0.039 |

Ranking médio entre datasets (descritivo):

| configuração | rank_medio |
|---|---|
| mobilevit_s/ft_aug | 3.80 |
| efficientnet_b0/ft_aug | 4.20 |
| mobilevit_s/ft_noaug | 4.20 |
| topológica mlp/diff | 4.40 |
| topológica mlp/cum | 5.20 |
| deit_tiny/ft_aug | 5.60 |
| efficientnet_b0/frozen | 5.80 |
| topológica bilstm/diff | 6.80 |
| topológica bilstm/cum | 7.40 |
| deit_tiny/frozen | 9.20 |
| mobilevit_s/frozen | 9.40 |

## 2. Configurações visuais contra `mobilevit_s/ft_aug`

Diferença = F1(configuração) − F1(referência). Holm por dataset.

| vs mobilevit_s/ft_aug | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| deit_tiny/frozen | -0.141 [-0.357, +0.074] p=1.00 | -0.031 [-0.062, +0.000] p=0.26 | -0.120 [-0.179, -0.061] p=0.00 | -0.050 [-0.096, -0.003] p=0.22 | -0.196 [-0.298, -0.095] p=0.00 |
| deit_tiny/ft_aug | -0.127 [-0.424, +0.171] p=1.00 | -0.002 [-0.043, +0.038] p=1.00 | -0.015 [-0.052, +0.021] p=1.00 | +0.003 [-0.027, +0.034] p=1.00 | -0.062 [-0.112, -0.013] p=0.05 |
| efficientnet_b0/frozen | +0.065 [-0.143, +0.273] p=1.00 | -0.000 [-0.030, +0.030] p=1.00 | -0.057 [-0.106, -0.008] p=0.10 | -0.015 [-0.061, +0.032] p=1.00 | -0.161 [-0.252, -0.070] p=0.01 |
| efficientnet_b0/ft_aug | +0.059 [-0.113, +0.232] p=1.00 | -0.003 [-0.035, +0.030] p=1.00 | +0.005 [-0.016, +0.027] p=1.00 | +0.002 [-0.023, +0.027] p=1.00 | -0.021 [-0.094, +0.052] p=1.00 |
| mobilevit_s/frozen | -0.110 [-0.330, +0.111] p=1.00 | -0.279 [-0.410, -0.148] p=0.00 | -0.131 [-0.197, -0.065] p=0.00 | -0.017 [-0.061, +0.028] p=1.00 | -0.172 [-0.258, -0.085] p=0.00 |
| mobilevit_s/ft_noaug | +0.006 [-0.194, +0.206] p=1.00 | -0.018 [-0.060, +0.024] p=1.00 | +0.001 [-0.027, +0.030] p=1.00 | +0.010 [-0.021, +0.040] p=1.00 | -0.018 [-0.085, +0.048] p=1.00 |
Significativas após Holm (p<0.05): 6 de 30.

## 3. `mobilevit_s/ft_aug` contra a via topológica (mesmos folds)

Diferença = F1(visual) − F1(topológica); positiva favorece a via visual. Holm por dataset.

| visual − topológica | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| topológica bilstm/cum | +0.169 [-0.124, +0.463] p=0.70 | +0.051 [-0.006, +0.108] p=0.22 | -0.009 [-0.030, +0.012] p=1.00 | +0.018 [-0.019, +0.054] p=1.00 | +0.012 [-0.074, +0.097] p=1.00 |
| topológica bilstm/diff | +0.177 [-0.088, +0.442] p=0.70 | +0.048 [-0.005, +0.102] p=0.22 | -0.009 [-0.030, +0.011] p=1.00 | +0.010 [-0.016, +0.036] p=1.00 | +0.015 [-0.065, +0.095] p=1.00 |
| topológica mlp/cum | +0.160 [-0.092, +0.413] p=0.70 | +0.046 [-0.001, +0.093] p=0.21 | -0.011 [-0.034, +0.013] p=1.00 | +0.008 [-0.025, +0.041] p=1.00 | -0.003 [-0.092, +0.087] p=1.00 |
| topológica mlp/diff | +0.094 [-0.158, +0.347] p=0.70 | +0.041 [-0.035, +0.118] p=0.27 | -0.007 [-0.033, +0.018] p=1.00 | +0.005 [-0.024, +0.034] p=1.00 | -0.023 [-0.094, +0.047] p=1.00 |
Significativas após Holm (p<0.05): 0 de 20.

## 4. Ganho do ajuste (F1 selecionado − F1 padrão) e configurações selecionadas

| configuração | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| deit_tiny/frozen | -0.022 | -0.010 | 0.001 | -0.010 | -0.004 |
| deit_tiny/ft_aug | -0.051 | -0.010 | -0.006 | 0.006 | -0.014 |
| efficientnet_b0/frozen | 0.015 | 0.003 | -0.003 | 0.002 | 0.015 |
| efficientnet_b0/ft_aug | 0.032 | 0.002 | 0.004 | 0.003 | 0.034 |
| mobilevit_s/frozen | 0.002 | -0.007 | 0.001 | -0.000 | -0.005 |
| mobilevit_s/ft_aug | 0.079 | 0.000 | 0.002 | 0.003 | 0.035 |
| mobilevit_s/ft_noaug | 0.159 | -0.005 | 0.022 | 0.017 | 0.056 |

- **deit_tiny/frozen**: lr_hd: 0.0001=0.48, 0.0003=0.12, 0.001=0.27, 0.003=0.13; wd: 0.0001=0.47, 0.001=0.24, 0.01=0.29; dropout: 0.1=0.35, 0.3=0.23, 0.5=0.43; bs: 8=0.07, 16=0.44, 32=0.49; config padrão em 12% dos folds; épocas medianas=51; parâmetros=41730; tempo mediano por fold=29s
- **deit_tiny/ft_aug**: lr_bb: 1e-05=0.12, 3e-05=0.39, 0.0001=0.47, 0.0003=0.03; lr_hd: 0.0001=0.61, 0.0003=0.01, 0.001=0.21, 0.003=0.16; wd: 0.0001=0.43, 0.001=0.2, 0.01=0.37; dropout: 0.1=0.49, 0.3=0.39, 0.5=0.12; bs: 8=0.05, 16=0.65, 32=0.29; config padrão em 11% dos folds; épocas medianas=15; parâmetros=5566146; tempo mediano por fold=421s
- **efficientnet_b0/frozen**: lr_hd: 0.0001=0.59, 0.0003=0.07, 0.001=0.13, 0.003=0.21; wd: 0.0001=0.53, 0.001=0.12, 0.01=0.35; dropout: 0.1=0.37, 0.3=0.24, 0.5=0.39; bs: 8=0.08, 16=0.57, 32=0.35; config padrão em 17% dos folds; épocas medianas=32; parâmetros=180994; tempo mediano por fold=23s
- **efficientnet_b0/ft_aug**: lr_bb: 3e-05=0.04, 0.0001=0.27, 0.0003=0.69; lr_hd: 0.0001=0.35, 0.0003=0.15, 0.001=0.47, 0.003=0.04; wd: 0.0001=0.35, 0.001=0.24, 0.01=0.41; dropout: 0.1=0.33, 0.3=0.2, 0.5=0.47; bs: 8=0.13, 16=0.47, 32=0.4; config padrão em 0% dos folds; épocas medianas=12; parâmetros=4188542; tempo mediano por fold=637s
- **mobilevit_s/frozen**: lr_hd: 0.0001=0.45, 0.0003=0.09, 0.001=0.33, 0.003=0.12; wd: 0.0001=0.48, 0.001=0.13, 0.01=0.39; dropout: 0.1=0.6, 0.3=0.13, 0.5=0.27; bs: 8=0.05, 16=0.48, 32=0.47; config padrão em 4% dos folds; épocas medianas=44; parâmetros=99074; tempo mediano por fold=36s
- **mobilevit_s/ft_aug**: lr_bb: 0.0001=0.29, 0.0003=0.71; lr_hd: 0.0001=0.53, 0.0003=0.05, 0.001=0.4, 0.003=0.01; wd: 0.0001=0.44, 0.001=0.27, 0.01=0.29; dropout: 0.1=0.25, 0.3=0.25, 0.5=0.49; bs: 8=0.17, 16=0.59, 32=0.24; config padrão em 0% dos folds; épocas medianas=15; parâmetros=5036706; tempo mediano por fold=670s
- **mobilevit_s/ft_noaug**: lr_bb: 3e-05=0.01, 0.0001=0.19, 0.0003=0.8; lr_hd: 0.0001=0.52, 0.0003=0.13, 0.001=0.33, 0.003=0.01; wd: 0.0001=0.51, 0.001=0.24, 0.01=0.25; dropout: 0.1=0.23, 0.3=0.15, 0.5=0.63; bs: 8=0.16, 16=0.53, 32=0.31; config padrão em 0% dos folds; épocas medianas=13; parâmetros=5036706; tempo mediano por fold=526s

## 5. Efeito do aumento de dados (ajuste fino): com − sem aumento

Diferença = F1(inversões e rotações) − F1(sem aumento), mesmo backbone e mesmos folds. Holm entre os datasets de cada arquitetura; positiva favorece o aumento.

| aumento (sem → com) | UCSB | CR | LA | LG | NHL |
|---|---|---|---|---|---|
| mobilevit_s | 0.859 → 0.852 (-0.006 [-0.206, +0.194] p=1.00) | 0.969 → 0.987 (+0.018 [-0.024, +0.060] p=1.00) | 0.985 → 0.983 (-0.001 [-0.030, +0.027] p=1.00) | 0.997 → 0.987 (-0.010 [-0.040, +0.021] p=1.00) | 0.899 → 0.917 (+0.018 [-0.048, +0.085] p=1.00) |

| arquitetura | dif_media | aumento_melhor | datasets | signif_holm |
|---|---|---|---|---|
| mobilevit_s | 0.004 | 2 | 5 | 0 |
