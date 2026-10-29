# Revisão crítica do TCC: Two-Stream Histopatologia

Oct 28, 2026 · @gabriel

## Resumo: prioridades

Os problemas mais graves aparecem entre os capítulos, não dentro de um só. Esta revisão cobre `introducao.tex`, `objetivos.tex`, `fundamentacao.tex` e `materiais-e-metodos.tex`.

1. **Validação.** O protocolo seleciona configurações nas mesmas partições de teste que usa para reportar, e não há teste estatístico.
2. **Baselines ausentes.** Faltam o descritor original da RaNN/SSN, descritores clássicos multiescala e CNN ou ViT puros.
3. **Objetivos e Introdução descrevem outro trabalho.** Falam em imagens artificiais, "fusão tardia" e "sensibilidade e robustez", que o método atual não faz.
4. **A Fundamentação duplica a Metodologia** e traz resultados próprios (entropias) que não são teoria.

## 1. Inconsistências entre arquivos

Os quatro arquivos usam termos e números que não batem entre si.

- **Terminologia de fusão.** Metodologia e Fundamentação adotam "fusão por concatenação de embeddings" porque "fusão tardia" é ambíguo. Objetivos e Introdução ainda dizem "fusão tardia (*feature-level fusion*)". A Introdução também define fusão precoce como combinação "antes do processamento", o que contradiz a própria arquitetura. O rótulo `sec:fusao-tardia` também é legado.
- **Objetivos desatualizados em relação ao método.**
  - "Imagens artificiais geradas por redes complexas" e "sequência temporal de imagens transformadas": o método não gera imagens, e sim 15 vetores de 24 medidas, ao longo de escalas, não de tempo.
  - "Via espacial/embedding espacial" aqui, "visual" na Metodologia e "sequencial" na Seção 2.3. Escolha um termo por via.
  - "Precisão, sensibilidade e robustez": as métricas são acurácia, F1 macro e AUC. Não há análise de robustez nem sensibilidade por classe.
  - "Baselines consolidadas": as comparações são GLCM/LBP/HOG e ablações próprias, sem CNN/ViT consolidados.
- **Contribuições da Introdução** citam só GLCM/LBP/HOG, mas a Metodologia tem comparações bem mais amplas. Alinhe os dois.
- **Numeração dos capítulos.** `objetivos.tex` é `\chapter`, mas a Introdução diz que o restante tem "quatro capítulos" começando em Fundamentação = Cap. 2. Ou Objetivos vira `\section` da Introdução, ou a numeração está errada.
- **Imagens de treino do UCSB.** A Metodologia diz "cerca de 52" de treino, mas 10% disso vira validação (\~5), então o treino efetivo é \~47.
- **Citação de LBP.** A Introdução cita `he1989texture` e `wang1990texture` para LBP. Esses são o *texture spectrum*, precursor do LBP. O LBP é de Ojala et al. (1996/2002) e nunca é citado nem definido. A Fundamentação o apresenta como "mantém a ideia" do espectro de textura.

## 2. Escolhas fracas

A escolha mais problemática é o redimensionamento para 256×256; as demais enfraquecem a conclusão central de "poucos dados".

- **Redimensionar tudo para 256×256.** É a pior escolha para uma via que depende de intensidades inteiras e de vizinhanças em pixels.
  - Os originais do IICBU são bem maiores, então o fator de redução é grande, e o CR sofre distorção de proporção.
  - Dizer que o efeito "é considerado na interpretação" não basta. Faça um experimento de sensibilidade (128/256/512, ou recortes sem distorção) ou ao menos mantenha a proporção.
- **Escolha dos datasets.**
  - Só UCSB, CR e NHL são diagnósticos. LA e LG são texturas de fígado de camundongo, o que limita conclusões do tipo "detecção de câncer".
  - O motivo da escolha (provavelmente comparabilidade com Scabini/Ribas) não está dito.
  - Bases desse tipo costumam estar perto do teto de acurácia, e com 58 imagens (teste de \~6 por fold) diferenças serão estatisticamente indetectáveis.
  - Considere ao menos uma base maior (BreakHis, NCT-CRC-HE, LC25000, BACH), possivelmente subamostrada.
- **Motivação de "poucos dados" sem experimento.** É o argumento central, mas nenhum experimento varia o tamanho do treino (10/25/50/100%). Esse experimento daria muito mais força que a tabela final.
- **MobileViT-S ImageNet como única via visual.**
  - A justificativa ("baixo custo, adequado a bases pequenas") confunde custo com adequação a poucos dados. O modelo tem \~5,6 M de parâmetros e é ajustado inteiro em \~47 imagens.
  - Faltam ao menos uma CNN (ResNet-18/EfficientNet-B0) e um ViT pequeno para sustentar "híbrido" como diferencial.
  - Em 2026, um backbone pré-treinado em patologia (CTransPath, UNI, Phikon etc.) é o baseline esperado em histopatologia com poucos dados.
- **Early stopping com validação de \~5 imagens (UCSB).** A perda de validação é ruidosa demais. Para os conjuntos minúsculos, considere número fixo de épocas.
- **Via topológica com apenas média e desvio global.**
  - A média global do grau de saída e a do grau de entrada tendem a ser quase redundantes, pois cada aresta conta uma vez como saída e outra como entrada. O mesmo vale para a força.
  - Sem empates, grau de entrada + saída é praticamente constante, então a média carrega pouco além de empates e bordas.
  - Faça análise de correlação e ablação das 24 features. Assimetria, curtose ou histogramas poderiam ser alternativas, e a entropia, que a Fundamentação já calcula, nem entra no vetor.
- **Redundância da recorrência por construção.** As medidas são acumuladas, então a feature no raio r já contém os raios anteriores. A "dependência histórica" é parcialmente embutida nos dados, e uma BiLSTM de 15 passos pode ser desnecessária. A ablação com MLP sobre o vetor achatado cobre isso, e foi bom prevê-la. Mas a Introdução afirma a hipótese como fato.
- **Desbalanceamento superestimado.** A Metodologia fala em "diferentes graus de desbalanceamento" e justifica o F1 macro por conjuntos "como o CR", que é 74/91. Mantenha o F1, mas a justificativa está exagerada.

## 3. O que falta justificar

Várias escolhas de projeto aparecem sem argumento ou com argumento que não as sustenta.

- **r\_max = 15.** O argumento é que a entropia varia em todo o intervalo, mas isso não implica informação discriminativa: a entropia de medidas acumuladas cresce naturalmente com r. A Shannon ainda cresce em r = 15, então não há saturação que sustente o corte. A ablação {4, 7, 10, 15} não testa valores maiores. Justifique por desempenho (curva acurácia × r) e inclua r > 15.
- **Hiperparâmetros** (64 unidades por direção, MLP de 128, dropout 0,5, taxas de aprendizado): nenhuma justificativa nem varredura. O argumento de "limitar sobreajuste" para a BiLSTM (46 mil parâmetros) soa estranho quando a via visual tem \~5,6 M.
- **Concatenação simples como fusão.** Justifique frente a alternativas (fusão com portas, atenção cruzada) ou diga explicitamente que é a opção mínima para poucos dados.
- **Dominância da via visual.** O backbone pré-treinado (lr 1e-5) contra a BiLSTM do zero (lr 1e-4) pode levar a rede a ignorar a via topológica. Um teste simples é permutar z\_top entre amostras e ver se o desempenho cai.
- **Hipótese por trás do NHL.** "A variabilidade cromática motivou a inclusão" não diz o que se espera. Um argumento defensável: graus *within-channel* dependem só da ordem das intensidades, logo seriam invariantes a transformações monotônicas de brilho e contraste, ao contrário da força. Ou retire a frase.
- **"Fusão de escores" e comparação com Huang (2022).** São boas comparações, mas explicite o que cada uma isola.
- **Aumento de dados.** Se houver, só flips e rotações de 90° preservam as medidas topológicas globais. Brilho, contraste, recorte e escala criam pares inconsistentes entre as vias, a menos que as redes sejam recalculadas por imagem aumentada.
- **Afirmação de lacuna** ("nenhum dos trabalhos citados combina…"): só vale com uma busca sistemática. Descreva a estratégia de busca ou suavize.

## 4. O que falta fundamentar

Faltam baselines, rigor estatístico e referências para decisões já tomadas no texto.

**Baselines ausentes que mais importam**

- O descritor original da RaNN (Ribas 2024) e o descritor SSN com classificador clássico (SVM/LDA), já que a contribuição é substituir a RaNN pela BiLSTM.
- GLCM com várias distâncias e LBP multirraio sobre a mesma BiLSTM. Isso isola "redes complexas" de "multiescala + recorrência" e responde à crítica de que GLCM/LBP/HOG são de "vizinhança fixa". O LBP multiresolução existe desde Ojala 2002.
- Um classificador clássico sobre as 360 features topológicas.

**Validação e estatística**

- O texto diz que o teste "não participou da escolha de hiperparâmetros", mas a Seção 3.5 escolhe "a melhor" estratégia e depois varia r\_max nas mesmas partições. Isso é seleção no teste. Use CV aninhado ou selecione só pela validação.
- As 30 avaliações (10 folds × 3 sementes) não são independentes, então o desvio padrão é subestimado. Não há teste de significância. Use *corrected resampled t-test* (Nadeau & Bengio) por dataset e Friedman/Nemenyi (Demšar) entre datasets.
- **Vazamento por origem** (16 seções no CR, poucos animais em LA/LG): está nas limitações, mas é grave o bastante para tentar `GroupKFold` se houver metadados. Se não houver, diga isso e modere as conclusões.

**Fundamentação faltante**

- Histopatologia e patologia computacional (H&E, variação de coloração).
- Transferência de aprendizado/fine-tuning e aumento de dados (a Introdução os cita, mas não há fundamento).
- Layer Normalization (Ba et al., 2016), AdamW (Loshchilov & Hutter, 2019) e ImageNet (Deng et al., 2009), sem citações.
- AUC-ROC e F1 (ex.: Fawcett 2006).
- Entropias de Shannon e amostral: citação e definição de como foram calculadas, pois isso não está descrito em lugar nenhum.
- Definição formal de LBP, e fórmulas ou parâmetros de GLCM e HOG.

**Pontos técnicos a conferir**

- Os pesos de força dependem de r (normalização por r²−1), então a força "acumulada" não é estritamente monótona. A frase "crescem de forma monotônica" vale com certeza para o grau, mas precisa ser verificada para a força.
- A condição 0 < ρ² também exclui a conexão entre canais no mesmo pixel. Confirme se esse é o comportamento da SSN original e explicite.
- Confira venue e qualidade das referências de 2026 (`mathiyalagi2026`, `priyadarshini2026`, `shalini2026`, `kushwah2026`, `moldovanu2026`, `ma2026cihm`), que sustentam argumentos importantes.
- "Modelos superaram patologistas" (Bejnordi 2017) merece contexto, pois a comparação foi em condições específicas.

## 5. Coesão da escrita por seção

A Fundamentação lê em parte como revisão bibliográfica e em parte como Metodologia. Regra prática: o Capítulo 2 diz o que cada técnica é; o Capítulo 3 diz o que foi escolhido e por quê. Cada fato fica em um só lugar.

**Fundamentação**

- **2.1.1** narra autor por autor (Haralick, He, Dalal) em vez de definir os conceitos. Isso é revisão. Conceitue cada descritor, com formulação mínima, e mova "são usados como comparação" para a Metodologia.
- **2.2.3** contém resultados próprios (6,7 → 7,6 bits; 1,9 → 2,15 → 2,03) e a figura de entropias. Isso é resultado ou justificativa de projeto, não teoria. Mova para a Metodologia ou para os Resultados.
- **2.3 a 2.5** estão cheias de "neste trabalho" e repetem a Metodologia quase literalmente:
  - o parágrafo sobre "fusão tardia" aparece em 2.5 e em 3.2.3;
  - a justificativa de escala e LN;
  - a padronização e a saturação das portas;
  - z\_top = \[h\_R ‖ h\_1\];
  - MobileViT-S com 256×256, ImageNet e z\_vis ∈ ℝ⁶⁴⁰.
- Faltam **trabalhos relacionados**. A Introdução acumula esse papel (parágrafos 5 a 7). Uma seção própria, com tabela comparando abordagens, deixaria a Introdução mais enxuta.

**Introdução**

A lógica de funil é boa, mas duas frases ficam sem citação ("dificuldade de convergência com poucas amostras" e "descritores calculados sobre vizinhanças de tamanho fixo"). O parágrafo de RNNs lista domínios sem relação com o trabalho (marcha, deepfake).

**Metodologia**

Bem escrita e honesta quanto às limitações. Os problemas são os blocos de \[XX\] e o fato de misturar protocolo com ablações e limitações de forma extensa.

## 6. Pendências marcadas no texto

Todas estão em `materiais-e-metodos.tex`.

- [ ] Weight decay, batch size, scheduler da taxa de aprendizado e número máximo de épocas.
- [ ] Paciência do early stopping.
- [ ] Uso de aumento de dados e lista de transformações (só no treino).
- [ ] Parâmetros de GLCM, LBP e HOG.
- [ ] Versão do PyTorch e origem dos pesos da MobileViT-S.
- [ ] Custo por imagem da implementação em C++ (256×256, r\_max = 15).
- [ ] GPU, tempo médio por avaliação e sementes aleatórias.
- [ ] `TODO`: conferir se as taxas de aprendizado (1e-5 no backbone, 1e-4 nas cabeças) correspondem ao que foi executado.
