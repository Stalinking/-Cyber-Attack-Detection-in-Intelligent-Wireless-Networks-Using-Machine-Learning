
# Explainable AI-Based Cyber Attack Detection
# in Intelligent Wireless Networks

## Project Overview

This project implements a machine learning-based
intrusion detection prototype for intelligent
wireless networks using Python.

Random Forest is used to classify network traffic
as normal or suspicious.

SHAP is used to explain the predictions made by
the machine learning model.

A Streamlit dashboard provides interactive
visualization and traffic classification.

## Objectives

1. Identify suspicious network traffic.
2. Train a Random Forest classifier.
3. Evaluate model performance.
4. Explain predictions using SHAP.
5. Visualize results using a web dashboard.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- SHAP
- Matplotlib
- Streamlit

## Dataset

The initial implementation uses a generated
synthetic dataset containing 2000 records.

Features include:
- Packet Rate
- Packet Size
- Failed Authentication
- Signal Strength
- Retransmission Rate
- Connection Count

Labels:
- 0: Normal
- 1: Suspicious

## Project Structure

Explainable-AI-Wireless-Intrusion-Detection/

- app.py
- requirements.txt
- README.md
- dataset/
- results/

## Installation

pip install -r requirements.txt

## Run the Application

streamlit run app.py

## Explainable AI

SHAP is used for:
- Global feature explanations
- Individual prediction explanations
- Understanding model decisions

## Outputs

- Classification performance
- Confusion matrix
- Feature importance
- SHAP beeswarm plot
- SHAP waterfall plot
- Interactive prediction dashboard

## Research Limitation

The initial model is trained using synthetic
data. Real-world performance must be evaluated
using an appropriate labeled wireless or IoT
network intrusion dataset.

## Conclusion

The project demonstrates an explainable
machine learning approach for identifying
suspicious traffic in intelligent wireless
network environments.

