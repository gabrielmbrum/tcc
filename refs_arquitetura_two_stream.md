# Guia de Referências para a Arquitetura Two-Stream

Este documento compila as referências bibliográficas prioritárias para fundamentar e guiar a implementação da arquitetura de dupla via (Topológica + Visual) com fusão tardia.

## Top 3: Artigos Base da Arquitetura

Estes três trabalhos formam o esqueleto estrutural e metodológico do projeto. Eles devem ser utilizados para justificar o fluxo de dados, a topologia da rede e a escolha do método de fusão.

**1. A Lógica da Rede de Dupla Via (Sequencial + Espacial)**
*   **Artigo:** Yousaf et al. (2024) - *Using two-stream EfficientNet-BiLSTM network for multiclass classification...*
*   **Aplicação:** Justifica a construção de uma macroarquitetura onde uma via extrai características espaciais usando um modelo profundo visual, enquanto a outra processa características sequenciais usando uma rede recorrente (LSTM). O processamento paralelo dos dois fluxos serve como molde para o código da sua rede.

**2. O Molde da Fusão Tardia (Textura + Visão Profunda)**
*   **Artigo:** Huang et al. (2022) - *Fusing hand-crafted and deep-learning features in a convolutional neural network model to identify prostate cancer...*
*   **Aplicação:** Prova a eficácia de concatenar vetores de características clássicas manuais (textura) com vetores de características profundas na histopatologia. Ele guiará a implementação do seu módulo de *Late Fusion* através da função de concatenação e subsequente passagem por uma rede MLP.

**3. A Justificativa Teórica para a Separação de Vias**
*   **Artigo:** Wang et al. (2023) - *STNet: Shape and texture joint learning through two-stream network...*
*   **Aplicação:** Fornece embasamento teórico para defender a separação da "textura topológica" e da "representação visual" em vias distintas. O artigo argumenta que o aprendizado conjunto com domínios separados evita a degradação e confusão das representações frente a uma única rede tentando aprender tudo simultaneamente.

---

## Ranking Top 10: Prioridade de Leitura e Implementação

A ordem abaixo indica a prioridade técnica para consulta durante a escrita da Seção de Materiais e Métodos e a codificação do modelo:

1.  **Huang et al. (2022):** Molde geral para a arquitetura de fusão tardia e fundamentação da concatenação de características manuais com características profundas em imagens histopatológicas.
2.  **Yousaf et al. (2024):** Referência primária para a codificação da integração paralela de uma rede LSTM com um modelo de visão espacial.
3.  **Mehta & Rastegari (2022):** Documentação arquitetural da via visual. Justifica a adoção da MobileViT, a remoção da sua camada de classificação original para extração do \textit{embedding}, e o papel dos mecanismos de atenção.
4.  **Ribas et al. (2024) e Zerati et al. (2026):** Fundamentação matemática exata para o cálculo das características topológicas das redes complexas coloridas (a matriz temporal que alimentará a LSTM).
5.  **Li (2025):** Literatura recente que valida o sucesso da fusão de características clássicas com redes \textit{Transformers} na superação de modelos isolados.
6.  **Wang et al. (2023):** Literatura base para argumentar na dissertação o porquê de separar os domínios (topológico e visual) na arquitetura.
7.  **Kushwah et al. (2026):** Apoio de domínio médico. Mostra o sucesso de redes híbridas CNN-Transformer especificamente em histopatologia mamária.
8.  **Hochreiter & Schmidhuber (1997):** Referência canônica e matemática para descrever o funcionamento das portas e da célula de memória da LSTM.
9.  **Moldovanu et al. (2026):** Valida a combinação de características clássicas e aprendizado profundo no contexto específico de imagens de cólon (relevante para o \textit{dataset} CR).
10. **Baltrušaitis et al. (2019):** Referência teórica fundamental. Deve ser citada ao definir formalmente a técnica de "Fusão Tardia" (\textit{Late Fusion}) e suas vantagens para evitar instabilidade no treinamento.