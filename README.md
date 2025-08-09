# 📂 Synthetic Datasets Generation

Code about using LLMs to generate synthetic labelled datasets for downstream model training.

**IMPORTANT:** This is just a playground project for my personal tests.

**NOTE:** Work in progress, for now only for classification datasets, but maybe more task-types included in the future 

## 🚀 Overview


This repository contains code and notebooks for personal tests and experiments related to synthetic data generation. 
It explores various techniques, models, and libraries to create realistic artificial datasets for different use cases.

## 🛠️ Key Technologies & Libraries

  * **Python:** The primary programming language.
  * **Ollama:** to quickly deploy and explore different open LLMs locally.
  * **Asyncio:** to make the LLM calls asynchronous and more efficient.

## 📁 Repository Structure

Describe the main directories and files in your repository to help users navigate it.

  * `notebooks/`: Jupyter notebooks with code examples and experiments.
  * `src/`: Source code for any functions or modules.
  * `data/`: Sample data or generated data files (if applicable).
  * `README.md`: This file.

## ⚙️ Getting Started

Provide instructions on how someone can get the project up and running.

### Prerequisites

  * Python 3.11+
  * A running instance of Ollama with the required LLM loaded 
(could easily implement clients for third-party commercial LLMs)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd your-repository
    ```
3.  Create a virtual environment and install the required dependencies from pyproject.toml file:
    ```bash
    uv pip install -r pyproject.toml
    ```

## 📝 Usage


  * "To see the synthetic data generation examples, open [notebooks/classification_data_example.ipynb](notebooks/classification_data_example.ipynb)."

## 🤝 Contribution & License


### License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

### Status

This is a **personal repository for testing and learning**. I welcome feedback, suggestions, and ideas, but please note that the code is not intended for production use and may not be actively maintained.