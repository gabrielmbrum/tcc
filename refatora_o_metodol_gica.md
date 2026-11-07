# Resumo de Refatoração do TCC: Arquitetura de Dupla Via e Validação Aninhada

## 1. Contexto e Motivação
O projeto propõe uma arquitetura de aprendizado profundo de **dupla via (Two-Stream Network)** para a classificação de imagens histopatológicas. A arquitetura é composta por:
* **Via Visual:** Extrai características diretamente da imagem RGB original, utilizando modelos de visão computacional.
* **Via Topológica:** Modela a imagem como uma Rede Complexa (Spatio-Spectral Network - SSN) e extrai métricas topológicas (ex: grau, força) variando o raio de conexão ($r=1 \dots 15$). A evolução dessas métricas gera uma sequência 1D que é processada por uma Rede Neural Recorrente (RNN).

**Alinhamento com a Proposta Aprovada:** Conforme delineado nos objetivos do projeto enviado à banca avaliadora (Objetivo 4), a meta do trabalho é avaliar a performance do modelo unificado estritamente frente às abordagens de via única. Sendo assim, o escopo metodológico dispensará a comparação de *benchmarking* contra descritores clássicos de textura (GLCM, LBP, HOG). A avaliação focará unicamente em mensurar e comprovar o ganho preditivo proporcionado pela fusão das dinâmicas topológicas com as características visuais em relação ao uso de cada via isoladamente.

## 2. Decisões Metodológicas e Estratégia de Avaliação

### A. Escopo de Avaliação (Single-Stream vs. Two-Stream)
A validação medirá o desempenho em três cenários distintos para isolar a contribuição de cada representação:
1. **Via Visual Isolada:** Imagens originais RGB, com aplicação de aumento de dados rotacional (*data augmentation*), processadas individualmente pelos *backbones* visuais até a camada de classificação.
2. **Via Topológica Isolada:** Sequências multiescala geradas pela SSN, processadas exclusivamente pelos codificadores recorrentes (RNNs) até a camada de classificação.
3. **Via Dupla (Fusão):** Integração dos *embeddings* gerados pelas duas vias para quantificar o ganho na acurácia e F1-score trazido pela arquitetura unificada.

### B. Espaço de Busca de Modelos (Vias Únicas)
Para a fase de seleção das vias únicas, serão investigadas as seguintes arquiteturas:
* **Via Topológica (6 Codificadores RNN):** (1) LSTM, (2) BiLSTM, (3) GRU, (4) BiGRU, (5) Stacked/Deep LSTM e (6) SAC-LSTM (Attention-LSTM).
* **Via Visual (6 Backbones):** Modelos Híbridos (**MobileViT**, **CoAtNet**), CNNs (**ResNet-50**, **EfficientNet**) e Vision Transformers (**ViT-Tiny**, **DeiT**).

### C. Estratégia de Treinamento: Validação Cruzada Aninhada (*Nested K-Fold*)
Para garantir a robustez na otimização dos hiperparâmetros (ex: *learning rate*, *dropout*, dimensões ocultas) sem causar vazamento de dados (*data leakage*), o treinamento adotará uma validação cruzada aninhada:
1. **Divisão Holdout (80/20):** O *dataset* completo será dividido em 80% para a fase de busca e treinamento (Treino/Validação) e 20% mantidos intocados como teste cego (*Holdout*).
2. **Inner K-Fold (Otimização):** O conjunto de 80% será dividido em 5 *folds*. Para cada uma das $n$ combinações de hiperparâmetros testadas, o modelo será treinado em 4 *folds* e validado no 1 *fold* restante, iterando até cobrir todos os 5 *folds*. O desempenho da combinação será a média dessa validação interna.
3. **Seleção e Fusão:** Concluída a etapa de validação interna para as vias isoladas, os **3 melhores modelos/configurações da Via Visual** e os **3 melhores da Via Topológica** serão selecionados.
4. **Treinamento e Avaliação Final:** Essas melhores configurações comporão os modelos finais de Via Dupla. O desempenho definitivo para comparação (Via Visual Isolada vs. Via Topológica Isolada vs. Via Dupla) será reportado avaliando-se os modelos sobre os 20% da base de *Holdout*.

## 3. Fundamentação Teórica (Referências)

O desenho experimental e as escolhas arquiteturais são diretamente embasados na literatura de estado da arte:

1. **Sobre a comparação direta entre vias únicas e via dupla como métrica de sucesso:**
   * *Yousaf, K. et al. "Using two-stream EfficientNet-BiLSTM network..."*: Demonstra a metodologia de treinar e avaliar configurações de fluxo único (apenas estático ou apenas movimento) como *baseline* metodológica, comprovando o ganho de acurácia na fusão subsequente no modelo *two-stream*.
2. **Sobre o treinamento isolado de vias para extração de *embeddings* e fusão posterior:**
   * *Nawaz, W. et al. "Acute Respiratory Distress Identification via Multi-Modality..."*: Justifica a estratégia FCAT-F (*Feature Concatenation with Frozen base models*), que valida o treinamento isolado das vias com suas próprias camadas de classificação e a posterior remoção dessas camadas para utilização das redes como extratoras de características na fusão.
3. **Sobre a viabilidade e arquitetura dos modelos RNN para sequências (Deep LSTM e SAC-LSTM):**
   * *Lin, R. et al. "Remaining useful life prediction...using multi-scale sequence and LSTM"*: Fundamenta o uso de arquiteturas **Deep/Stacked LSTM** (com múltiplas camadas ocultas) no processamento de sequências multiescala.
   * *Shalini, T.G. et al. "An optimal CSBO-based detection and classification of breast cancer using deep SAC-LSTM..."*: Dá base ao uso da variação **SAC-LSTM (Attention-LSTM)** aplicada a imagens histopatológicas de câncer, utilizando a atenção para ponderar os passos sequenciais mais informativos.
4. **Sobre o K-Fold e a Otimização de Hiperparâmetros em bases médicas restritas:**
   * *Priyadarshini, I. "A Firefly Algorithm-Optimized CNN–BiLSTM Model for Automated Detection of Bone Cancer..."*: Reforça a necessidade de busca rigorosa de hiperparâmetros (taxas de aprendizado, unidades ocultas) antes de treinar o modelo unificado (CNN-BiLSTM) com o conjunto completo de dados em domínios médicos limitados.