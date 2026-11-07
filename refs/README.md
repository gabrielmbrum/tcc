# 📚 Índice de Artigos e Referências (`refs/`)

Este diretório contém os artigos científicos em PDF e arquivos de citação utilizados na pesquisa do TCC. A estrutura de pastas reflete rigorosamente a divisão temática definida no arquivo bibliográfico [`tex/referencias.bib`](../tex/referencias.bib).

---

## 🗂️ Sumário de Pastas

| # | Pasta | Descrição Temática | Qtd. Arquivos |
|---|---|---|:---:|
| 01 | [`01_rnn/`](./01_rnn/) | Redes Neurais Recorrentes (RNN, LSTM, BiLSTM, CRNN) | 4 |
| 02 | [`02_cnn/`](./02_cnn/) | Redes Neurais Convolucionais e Surveys Médicos (CNN) | 3 |
| 03 | [`03_vit/`](./03_vit/) | Vision Transformers e Grafos no Contexto Visual (ViT, GNN) | 1 |
| 04 | [`04_cnn_vit_hibridos/`](./04_cnn_vit_hibridos/) | Modelos Híbridos CNN-ViT / Convolução + Atenção / Mamba | 9 |
| 05 | [`05_two_stream_fusion/`](./05_two_stream_fusion/) | Dupla Via, Redes Multi-Stream e Fusão Tardia/Multimodal | 16 |
| 06 | [`06_histopatologia/`](./06_histopatologia/) | Diagnóstico e Análise de Imagens Histopatológicas | 9 |
| 07 | [`07_data_augmentation/`](./07_data_augmentation/) | Técnicas de Aumento de Dados (Data Augmentation) | 0 |
| 08 | [`08_feature_extraction/`](./08_feature_extraction/) | Extração de Características (GLCM, LBP, HOG, Radiomics, DL) | 14 |
| 09 | [`09_complex_networks/`](./09_complex_networks/) | Redes Complexas aplicadas a Análise de Textura e Cor | 3 |
| 10 | [`10_outros/`](./10_outros/) | Fundamentos, Classificação, Otimização e Datasets | 2 |
| | **Total** | | **61** |

---

## 📖 Catálogo Detalhado de Artigos

### 01. Redes Neurais Recorrentes (RNN, LSTM, BiLSTM, CRNN) (`01_rnn/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [Dynamic_Facial_Expression_Feature_Learning_Based_on_Sparse_RNN.pdf](./01_rnn/Dynamic_Facial_Expression_Feature_Learning_Based_on_Sparse_RNN.pdf) | `zhi2019slstm` | 2019 | Dynamic facial expression feature learning based on sparse RNN |
| [Implementation_of_Potential_Leukemia_Detection_using_Recurrent_Neural_Networks_with_Blood_Cell_Counting.pdf](./01_rnn/Implementation_of_Potential_Leukemia_Detection_using_Recurrent_Neural_Networks_with_Blood_Cell_Counting.pdf) | `akanksha2022leukemia` | 2022 | Implementation of potential leukemia detection using recurrent neural networks with blood cell counting |
| [Medical_Image_Character_Recognition_Based_on_Multi-scale_Neural_Convolutional_Network.pdf](./01_rnn/Medical_Image_Character_Recognition_Based_on_Multi-scale_Neural_Convolutional_Network.pdf) | - | 2021 | Medical Image Character Recognition Based on Multi-scale Neural Convolutional Network (CRNN) |
| [Remaining useful life prediction in prognostics using multi-scale sequence and Long Short-Term Memory network.pdf](./01_rnn/Remaining useful life prediction in prognostics using multi-scale sequence and Long Short-Term Memory network.pdf) | `lin2022mslstm` | 2022 | Remaining useful life prediction in prognostics using multi-scale sequence and long short-term memory network |

### 02. Redes Neurais Convolucionais e Surveys Médicos (CNN) (`02_cnn/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [10.1007_978-3-319-46723-8_14-citation.bib](./02_cnn/10.1007_978-3-319-46723-8_14-citation.bib) | `alexnetMed` | 2016 | Multimodal Deep Learning for Cervical Dysplasia Diagnosis (MICCAI 2016) |
| [A review on deep learning in medical image analysis.pdf](./02_cnn/A review on deep learning in medical image analysis.pdf) | - | 2022 | A review on deep learning in medical image analysis (Suganyadevi et al.) |
| [Predictive_Modelling_for_Cardiovascular_Health_A_Survey_on_Deep_Learning_Approaches.pdf](./02_cnn/Predictive_Modelling_for_Cardiovascular_Health_A_Survey_on_Deep_Learning_Approaches.pdf) | - | 2025 | Predictive Modelling for Cardiovascular Health: A Survey on Deep Learning Approaches |

### 03. Vision Transformers e Grafos no Contexto Visual (ViT, GNN) (`03_vit/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [GTUNet: a GNN and transformer enhanced U-Net for coronary artery.pdf](./03_vit/GTUNet: a GNN and transformer enhanced U-Net for coronary artery.pdf) | `sheng2026gtunet` | 2026 | GTUNet: a GNN and transformer enhanced U-Net for coronary artery segmentation |

### 04. Modelos Híbridos CNN-ViT / Convolução + Atenção / Mamba (`04_cnn_vit_hibridos/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [10.1007_s10278-026-02249-8-citation.ris](./04_cnn_vit_hibridos/10.1007_s10278-026-02249-8-citation.ris) | `clinicalReview` | 2026 | A Clinically Grounded Review of Medical Image Classification (Arquivo de Citação RIS) |
| [A Clinically Grounded Review of Medical Image Classification: Quantitative Insights into CNNs, Vision Transformers, and Hybrid CNN-ViT Models.pdf](./04_cnn_vit_hibridos/A Clinically Grounded Review of Medical Image Classification: Quantitative Insights into CNNs, Vision Transformers, and Hybrid CNN-ViT Models.pdf) | `clinicalReview` | 2026 | A Clinically Grounded Review of Medical Image Classification: Quantitative Insights into CNNs, Vision Transformers, and Hybrid CNN-ViT Models |
| [Accurate and lightweight oral cancer detection using SE-MobileViT on clinically validated image dataset.pdf](./04_cnn_vit_hibridos/Accurate and lightweight oral cancer detection using SE-MobileViT on clinically validated image dataset.pdf) | `kabir2025semobilevit` | 2025 | Accurate and lightweight oral cancer detection using SE-MobileViT on clinically validated image dataset |
| [Advanced deep learning architectures for enhanced mammography classification: a comparative study of CNNs and ViT.pdf](./04_cnn_vit_hibridos/Advanced deep learning architectures for enhanced mammography classification: a comparative study of CNNs and ViT.pdf) | `sharma2025mammography` | 2025 | Advanced deep learning architectures for enhanced mammography classification: a comparative study of CNNs and ViT |
| [CIHM: context-insight hybrid Mamba for efficient medical image segmentation.pdf](./04_cnn_vit_hibridos/CIHM: context-insight hybrid Mamba for efficient medical image segmentation.pdf) | `ma2026cihm` | 2026 | CIHM: Context-insight hybrid Mamba for efficient medical image segmentation |
| [GSAC-UFormer: Groupwise Self-Attention Convolutional Transformer-Based UNet for Medical Image Segmentation.pdf](./04_cnn_vit_hibridos/GSAC-UFormer: Groupwise Self-Attention Convolutional Transformer-Based UNet for Medical Image Segmentation.pdf) | `garbaz2025gsac` | 2025 | GSAC-UFormer: Groupwise self-attention convolutional transformer-based UNet for medical image segmentation |
| [HI-MViT: A lightweight model for explainable.pdf](./04_cnn_vit_hibridos/HI-MViT: A lightweight model for explainable.pdf) | `ding2023himvit` | 2023 | HI-MViT: A lightweight model for explainable skin disease classification based on modified MobileViT |
| [Hybrid_CNN-Transformer_Models_in_Histopathology_Image_Analysis_A_Scoping_Review.pdf](./04_cnn_vit_hibridos/Hybrid_CNN-Transformer_Models_in_Histopathology_Image_Analysis_A_Scoping_Review.pdf) | `oliveira2025hybrid` | 2025 | Hybrid CNN-Transformer models in histopathology image analysis: A scoping review |
| [Unveiling_Prognostic_Pattern_A_Hybrid_Approach_For_Predictive_Analytics_in_Medical_Image_Data.pdf](./04_cnn_vit_hibridos/Unveiling_Prognostic_Pattern_A_Hybrid_Approach_For_Predictive_Analytics_in_Medical_Image_Data.pdf) | - | 2024 | Unveiling Prognostic Pattern: A Hybrid Approach For Predictive Analytics in Medical Image Data |

### 05. Dupla Via, Redes Multi-Stream e Fusão Tardia/Multimodal (`05_two_stream_fusion/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [A Firefly Algorithm-Optimized CNN–BiLSTM Model for Automated Detection of Bone Cancer and Marrow Cell Abnormalities.pdf](./05_two_stream_fusion/A Firefly Algorithm-Optimized CNN–BiLSTM Model for Automated Detection of Bone Cancer and Marrow Cell Abnormalities.pdf) | `priyadarshini2026cnnbilstm` | 2026 | A firefly algorithm-optimized CNN--BiLSTM model for automated detection of bone cancer and marrow cell abnormalities |
| [A_Lightweight_Triple-Stream_Network_With_Multisensor_Fusion_for_Enhanced_Few-Shot_Learning_Fault_Diagnosis.pdf](./05_two_stream_fusion/A_Lightweight_Triple-Stream_Network_With_Multisensor_Fusion_for_Enhanced_Few-Shot_Learning_Fault_Diagnosis.pdf) | `peng2025tfsdfn` | 2025 | A lightweight triple-stream network with multisensor fusion for enhanced few-shot learning fault diagnosis |
| [Acute Respiratory Distress Identification via Multi-Modality Using Deep Learning.pdf](./05_two_stream_fusion/Acute Respiratory Distress Identification via Multi-Modality Using Deep Learning.pdf) | `nawaz2025ard` | 2025 | Acute respiratory distress identification via multi-modality using deep learning |
| [Bidirectional feature fusion via cross-attention transformer for.pdf](./05_two_stream_fusion/Bidirectional feature fusion via cross-attention transformer for.pdf) | `chen2025chrysanthemum` | 2025 | Bidirectional feature fusion via cross-attention transformer for chrysanthemum classification |
| [DAPR-Net: Symmetric Dual-Stream Pyramid Architecture for Non-Rigid.pdf](./05_two_stream_fusion/DAPR-Net: Symmetric Dual-Stream Pyramid Architecture for Non-Rigid.pdf) | `yang2026daprnet` | 2026 | DAPR-Net: Symmetric dual-stream pyramid architecture for non-rigid brain MRI registration |
| [Deep_hybrid_model_for_Mpox_disease_diagnosis_from_.pdf](./05_two_stream_fusion/Deep_hybrid_model_for_Mpox_disease_diagnosis_from_.pdf) | - | 2024 | Deep hybrid model for Mpox disease diagnosis from skin lesion images (Khan et al.) |
| [Dual-Stream CoAtNet models for accurate breast ultrasound image segmentation.pdf](./05_two_stream_fusion/Dual-Stream CoAtNet models for accurate breast ultrasound image segmentation.pdf) | `zaidkilani2025dualstream` | 2024 | Dual-Stream CoAtNet models for accurate breast ultrasound image segmentation |
| [Hybrid Decision-Level Fusion of CNN-Based Deep and Handcrafted Features for Colon Cancer Classification.pdf](./05_two_stream_fusion/Hybrid Decision-Level Fusion of CNN-Based Deep and Handcrafted Features for Colon Cancer Classification.pdf) | `moldovanu2026hybrid` | 2026 | Hybrid decision-level fusion of CNN-based deep and handcrafted features for colon cancer classification |
| [Hybrid-NET_A_fusion_of_DenseNet169_and_advanced_ma.pdf](./05_two_stream_fusion/Hybrid-NET_A_fusion_of_DenseNet169_and_advanced_ma.pdf) | `khan2024hybridnet` | 2024 | Hybrid-NET: A fusion of DenseNet169 and advanced machine learning classifiers for enhanced brain tumor diagnosis |
| [Improved breast ultrasound tumor classification using dual-input CNN with.pdf](./05_two_stream_fusion/Improved breast ultrasound tumor classification using dual-input CNN with.pdf) | `zou2023dualinput` | 2023 | Improved breast ultrasound tumor classification using dual-input CNN with GAP-guided attention loss |
| [MMDB: Multimodal dual-branch model for multi-functional bioactive peptide prediction.pdf](./05_two_stream_fusion/MMDB: Multimodal dual-branch model for multi-functional bioactive peptide prediction.pdf) | `kang2024mmdb` | 2024 | MMDB: Multimodal dual-branch model for multi-functional bioactive peptide prediction |
| [MSSFN: A multi-scale sequence fusion network for CT-based diagnosis of pulmonary complications.pdf](./05_two_stream_fusion/MSSFN: A multi-scale sequence fusion network for CT-based diagnosis of pulmonary complications.pdf) | `zeng2025mssfn` | 2025 | MSSFN: A multi-scale sequence fusion network for CT-based diagnosis of pulmonary complications |
| [Oral cancer detection using feature-level fusion and novel self-attention mechanisms.pdf](./05_two_stream_fusion/Oral cancer detection using feature-level fusion and novel self-attention mechanisms.pdf) | `khan2024oral` | 2024 | Oral cancer detection using feature-level fusion and novel self-attention mechanisms |
| [Revolutionizing lung cancer detection using hybrid CNN-LSTM (HCLSTM) for accurate diagnosis and enhanced nodule classification.pdf](./05_two_stream_fusion/Revolutionizing lung cancer detection using hybrid CNN-LSTM (HCLSTM) for accurate diagnosis and enhanced nodule classification.pdf) | `mathiyalagi2026hclstm` | 2026 | Revolutionizing lung cancer detection using hybrid CNN-LSTM (HCLSTM) for accurate diagnosis and enhanced nodule classification |
| [STNet: shape and texture joint learning through two-stream network for knowledge-guided image recognition.pdf](./05_two_stream_fusion/STNet: shape and texture joint learning through two-stream network for knowledge-guided image recognition.pdf) | `wang2023stnet` | 2023 | STNet: Shape and texture joint learning through two-stream network for knowledge-guided image recognition |
| [Using two‑stream EfficientNet‑BiLSTM network for multiclass.pdf](./05_two_stream_fusion/Using two‑stream EfficientNet‑BiLSTM network for multiclass.pdf) | `yousaf2024twostream` | 2024 | Using two-stream EfficientNet-BiLSTM network for multiclass classification of disturbing YouTube videos |

### 06. Diagnóstico e Análise de Imagens Histopatológicas (`06_histopatologia/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [A comprehensive review of the deep learning-based tumor analysis approaches in histopathological images.pdf](./06_histopatologia/A comprehensive review of the deep learning-based tumor analysis approaches in histopathological images.pdf) | - | 2023 | A comprehensive review of the deep learning-based tumor analysis approaches in histopathological images (Abdel-Nabi et al.) |
| [An optimal CSBO-based detection and classification of breast cancer using deep SAC-LSTM design from histopathological image.pdf](./06_histopatologia/An optimal CSBO-based detection and classification of breast cancer using deep SAC-LSTM design from histopathological image.pdf) | `shalini2026saclstm` | 2026 | An optimal CSBO-based detection and classification of breast cancer using deep SAC-LSTM design from histopathological image |
| [EVC-Net: A Hybrid Deep Learning Network for Breast Cancer.pdf](./06_histopatologia/EVC-Net: A Hybrid Deep Learning Network for Breast Cancer.pdf) | `kushwah2026evcnet` | 2026 | EVC-Net: A hybrid deep learning network for breast cancer classification from histopathological images |
| [EVC-Net_ A Hybrid Deep Learning Network for Breast Cancer_compressed.pdf](./06_histopatologia/EVC-Net_ A Hybrid Deep Learning Network for Breast Cancer_compressed.pdf) | `kushwah2026evcnet` | 2026 | EVC-Net: A hybrid deep learning network for breast cancer classification (Versão Comprimida) |
| [Fusing hand-crafted and deep-learning features in a convolutional neural network model to identify prostate cancer in pathology images.pdf](./06_histopatologia/Fusing hand-crafted and deep-learning features in a convolutional neural network model to identify prostate cancer in pathology images.pdf) | `huang2022fusing` | 2022 | Fusing hand-crafted and deep-learning features in a convolutional neural network model to identify prostate cancer in pathology images |
| [Fusion feature-based hybrid methods for diagnosing oral squamous cell carcinoma in histopathological images.pdf](./06_histopatologia/Fusion feature-based hybrid methods for diagnosing oral squamous cell carcinoma in histopathological images.pdf) | `li2025fusion` | 2025 | Fusion feature-based hybrid methods for diagnosing oral squamous cell carcinoma in histopathological images |
| [Graph neural networks in histopathology.pdf](./06_histopatologia/Graph neural networks in histopathology.pdf) | `brussee2025graph` | 2025 | Graph neural networks in histopathology: Emerging trends and future directions |
| [Weighted ColorMorphology Feature Fusion for Tuberculosis.pdf](./06_histopatologia/Weighted ColorMorphology Feature Fusion for Tuberculosis.pdf) | - | 2024 | Weighted ColorMorphology Feature Fusion for Tuberculosis Bacilli Detection from Cytopathology Images |
| [histopathology image analysis for colorectal cancer using deep learning techniques.pdf](./06_histopatologia/histopathology image analysis for colorectal cancer using deep learning techniques.pdf) | `histImgColorectal` | 2026 | Histopathology image analysis for colorectal cancer using deep learning techniques: A comprehensive survey |

### 07. Técnicas de Aumento de Dados (Data Augmentation) (`07_data_augmentation/`)

*Nenhum arquivo no momento. Pasta reservada para inclusão futura.*

### 08. Extração de Características (GLCM, LBP, HOG, Radiomics, DL) (`08_feature_extraction/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [Exploring_Facial_Feature_Extraction_and_Temporal_Dynamics_for_Deepfake_Video_Detection_A_CNN-RNN_Study_on_Faceforensics.pdf](./08_feature_extraction/Exploring_Facial_Feature_Extraction_and_Temporal_Dynamics_for_Deepfake_Video_Detection_A_CNN-RNN_Study_on_Faceforensics.pdf) | `jeswani2026deepfake` | 2026 | Exploring facial feature extraction and temporal dynamics for deepfake video detection: A CNN-RNN study on FaceForensics++ |
| [Feature_Extraction_Using_an_RNN_Autoencoder_for_Skeleton-Based_Abnormal_Gait_Recognition.pdf](./08_feature_extraction/Feature_Extraction_Using_an_RNN_Autoencoder_for_Skeleton-Based_Abnormal_Gait_Recognition.pdf) | `jun2020rnnae` | 2020 | Feature extraction using an RNN autoencoder for skeleton-based abnormal gait recognition |
| [Feature_Selection_and_Extraction_in_Sequence_Labeling_for_Arrhythmia_Detection.pdf](./08_feature_extraction/Feature_Selection_and_Extraction_in_Sequence_Labeling_for_Arrhythmia_Detection.pdf) | `ye2021arrhythmia` | 2021 | Feature selection and extraction in sequence labeling for arrhythmia detection |
| [GLCM - Textural_Features_for_Image_Classification.pdf](./08_feature_extraction/GLCM - Textural_Features_for_Image_Classification.pdf) | `haralick1973textural` | 1973 | Textural Features for Image Classification (Haralick et al.) |
| [HOG - Histograms_of_oriented_gradients_for_human_detection.pdf](./08_feature_extraction/HOG - Histograms_of_oriented_gradients_for_human_detection.pdf) | `dalal2005histograms` | 2005 | Histograms of Oriented Gradients for Human Detection (Dalal & Triggs) |
| [LBP - TEXTURE CLASSIFICATION USING TEXTURE SPECTRUM.pdf](./08_feature_extraction/LBP - TEXTURE CLASSIFICATION USING TEXTURE SPECTRUM.pdf) | `wang1990texture` | 1990 | Texture classification using texture spectrum (Wang & He) |
| [LBP - Texture_Unit_Texture_Spectrum_And_Texture_Analysis.pdf](./08_feature_extraction/LBP - Texture_Unit_Texture_Spectrum_And_Texture_Analysis.pdf) | `he1989texture` | 1989 | Texture Unit, Texture Spectrum, and Texture Analysis (He & Wang) |
| [MIAFEx An attetion-based feature extraction.pdf](./08_feature_extraction/MIAFEx An attetion-based feature extraction.pdf) | `ramossoto2025miafex` | 2025 | MIAFEx: An attention-based feature extraction method for medical image classification |
| [Multi-Modal_Biometric_Authentication_Using_Attention_Based_RNN_Feature_Extraction_and_Walrus-Assisted_Sculptor_Optimized_Fusion.pdf](./08_feature_extraction/Multi-Modal_Biometric_Authentication_Using_Attention_Based_RNN_Feature_Extraction_and_Walrus-Assisted_Sculptor_Optimized_Fusion.pdf) | `nair2025multimodal` | 2025 | Multi-modal biometric authentication using attention based RNN feature extraction and walrus-assisted sculptor optimized fusion |
| [ReviewonMedicalImageSegmentationandFeatureExtractionApproaches.pdf](./08_feature_extraction/ReviewonMedicalImageSegmentationandFeatureExtractionApproaches.pdf) | `ahmed2023review` | 2023 | Review on medical image segmentation and feature extraction approaches |
| [Statistical_Feature_Extraction_and_Classification_of_Complex_Data_Based_on_Deep_Learning.pdf](./08_feature_extraction/Statistical_Feature_Extraction_and_Classification_of_Complex_Data_Based_on_Deep_Learning.pdf) | `wu2024statistical` | 2024 | Statistical feature extraction and classification of complex data based on deep learning |
| [VMD-based iterative Boruta feature extraction and CNNA-BiLSTM for short-term load forecasting.pdf](./08_feature_extraction/VMD-based iterative Boruta feature extraction and CNNA-BiLSTM for short-term load forecasting.pdf) | `xu2025vmdib` | 2025 | VMD-based iterative Boruta feature extraction and CNNA-BiLSTM for short-term load forecasting |
| [a comparative study of statistical randiomics and DL feature extraction.pdf](./08_feature_extraction/a comparative study of statistical randiomics and DL feature extraction.pdf) | `dehbozorgi2025` | 2025 | A comparative study of statistical, radiomics, and deep learning feature extraction techniques for medical image classification |
| [mms-net multi level multi scale feature extraction for med img segmentation.pdf](./08_feature_extraction/mms-net multi level multi scale feature extraction for med img segmentation.pdf) | `zhao2023mmsnet` | 2023 | MMS-Net: Multi-level multi-scale feature extraction network for medical image segmentation |

### 09. Redes Complexas aplicadas a Análise de Textura e Cor (`09_complex_networks/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [CNT an image transfrom based on cn for enchanced texture classification.pdf](./09_complex_networks/CNT an image transfrom based on cn for enchanced texture classification.pdf) | `CNT` | 2026 | CNT: An image transform based on complex networks for enhanced texture characterization |
| [Color-texture classification based on spatio-spectral complex network representations.pdf](./09_complex_networks/Color-texture classification based on spatio-spectral complex network representations.pdf) | `ribas2024colorspectral` | 2024 | Color-texture classification based on spatio-spectral complex network representations |
| [Spatio-spectral networks for color-texture analysis.pdf](./09_complex_networks/Spatio-spectral networks for color-texture analysis.pdf) | `Scabini_2020` | 2020 | Spatio-spectral networks for color-texture analysis |

### 10. Fundamentos, Classificação, Otimização e Datasets (`10_outros/`)

| Arquivo | Chave BibTeX | Ano | Título da Obra |
|---|---|:---:|---|
| [TriGWONet a lightweight multibranch convolutional neural network using gray wolf optimization for accurate oral cancer image classification.pdf](./10_outros/TriGWONet a lightweight multibranch convolutional neural network using gray wolf optimization for accurate oral cancer image classification.pdf) | `kabir2026trigwonet` | 2026 | TriGWONet: A lightweight multibranch convolutional neural network using gray wolf optimization for accurate oral cancer image classification |
| [research-on-tissue-section-negative-detection-algorithm-based-on-multispectral-microscopic-imaging.pdf](./10_outros/research-on-tissue-section-negative-detection-algorithm-based-on-multispectral-microscopic-imaging.pdf) | `wang2026msi` | 2026 | Research on tissue section negative detection algorithm based on multispectral microscopic imaging |

