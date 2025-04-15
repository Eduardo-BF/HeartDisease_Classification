# 💓 Heart Disease Classification ❤️‍🩹

🇧🇷 **Português** | 🇺🇸 [English below](#english-version)

Este repositório contém um projeto de aprendizado de máquina aplicado ao clássico dataset de **doença cardíaca** do [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

O objetivo é **construir e comparar diferentes classificadores de machine learning** para prever a presença de doença cardíaca com base em variáveis clínicas.

---

## 📁 Estrutura do Projeto

- `Classif_xyz.py`: script de classificação utilizando método _XYZ_
- `data_abc.pkl`: dados processados para multiclasse e binário (`_bin.pkl`)
- `PreProcess.py`: pré-processamento de dados e divisão para treino/teste
- `README.md`: descrição do projeto

---

## ⚙️ Funcionalidades Implementadas

- Carregamento e limpeza de dados
- Conversão do problema para classificação **binária** (`0`: sem doença, `1`: com doença)
- Pré-processamento com `OneHotEncoder` e `StandardScaler`
- Exportação dos dados tratados para `.pkl`
- Classificação com **Regressão Logística**
- Validação cruzada com **tuning de hiperparâmetros** via `GridSearchCV`

---

## 📊 Dataset

- Dataset original: [Heart Disease – UCI](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Classes:
  - `0`: ausência de doença cardíaca
  - `1, 2, 3, 4`: diferentes níveis de presença da doença  
  > Para simplificação, as classes 1 a 4 são agrupadas como `1` (problema binário)

---

## 🚀 Como Rodar o Projeto

Clone o repositório:
   ```bash
   git clone https://github.com/Eduardo-BF/HeartDisease_Classification.git
   cd HeartDisease_Classification
   ```

---

## English Version

This repository contains a machine learning project applied to the classic **heart disease** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

The goal is to **build and compare different ML classifiers** to predict heart disease presence based on clinical variables.

---

## 📁 Project Structure

- `Classif_xyz.py`: classification script using method _XYZ_
- `data_abc.pkl`: processed data for multiclass and binary problems (`_bin.pkl`)
- `PreProcess.py`: data preprocessing and train/test split
- `README.md`: project description

---

## ⚙️ Features Implemented

- Data loading and cleaning
- Conversion to **binary classification** (`0`: no disease, `1`: has disease)
- Preprocessing with `OneHotEncoder` and `StandardScaler`
- Export of processed data to `.pkl`
- Classification using **Logistic Regression**
- Cross-validation and **hyperparameter tuning** with `GridSearchCV`

---

## 📊 Dataset

- Original dataset: [Heart Disease – UCI](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Classes:
  - `0`: no heart disease
  - `1, 2, 3, 4`: different levels of disease  
  > For simplification, classes 1–4 are grouped as `1` (binary problem)

---

## 🚀 How to Run the Project

Clone the repository:
   ```bash
   git clone https://github.com/Eduardo-BF/HeartDisease_Classification.git
   cd HeartDisease_Classification
   ```


