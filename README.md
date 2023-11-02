# Minimum Description Length and Generalization Guarantees for Representation Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/PiotrKrasnowski/MDL_and_Generalization_Guarantees_for_Representation_Learning/blob/main/LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.12.1-%237732a8)

This repository contains the code for our paper [Minimum Description Length and Generalization Guarantees for Representation Learning](), which has been accepted by NeurIPS 2023.

## Installation

Most of the requirements of this project are standard Python packages and the CUDA toolkit. A full list of packages can be found below. 

### Requirements:
- Python >= 3.10
- PyTorch >= 1.12.0 (ours: CUDA 11.0)
- torchvision >= 0.14 (ours: CUDA 11.0)
- matplotlib
- time
- os
- numpy
- numbers

## Evaluation

To evaluate the results reported in our paper, run firstly the script ['main.py']. The full training time may take several hours, but it could be shortened by reducing the number of repetitions (line 13) from 5 to 1. The training results and the trained models are stored in the directory ['results'], with a proper timestamp. 

The training results could be visualized using the script ['analyze_results.py']. In line 5 of the script, it is necessary to select a timestamp pointing to the right folder with the results. The produced graphs will be saved in the directory ['figures'], with the same timestamp. 

## Citation

We welcome you to contact us (e-mail: ```p.g.krasnowski@gmail.com```) if you have any problem when reading the paper or reproducing the code.

## Acknowledgment
