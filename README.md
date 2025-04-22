# 🧠 LoRA Fine-Tuning for AG News with RoBERTa

This repository contains a notebook that fine-tunes a RoBERTa model using LoRA (Low-Rank Adaptation) on the [AG News dataset](https://huggingface.co/datasets/ag_news) for text classification. The project leverages the Hugging Face Transformers and PEFT libraries, along with Weights & Biases for experiment tracking.

---

## 📁 Files

- `DL_project2.ipynb` – Main notebook to fine-tune and evaluate the model  
- `submission_85_9.csv` – Sample output file with predictions on unlabelled data  
- `test_unlabelled.pkl` – (Not included) Expected test file to make final predictions on

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   https://github.com/Diksha942/Deep_Learning_P2.git
   cd Deep_Learning_P2
   ```

2. **Open the Notebook**
   - Use any IDE or environment that supports Jupyter notebooks, such as:
     - VS Code with the Jupyter extension
     - JupyterLab
     - Google Colab (recommended for GPU)

3. **Update the Unlabelled Test Path**
   - Find this line in the notebook:
     ```python
     unlabelled_dataset = pd.read_pickle("/content/test_unlabelled.pkl")
     ```
   - Replace the path with the location of your own `test_unlabelled.pkl` file.

4. **Run All Cells**
   - Execute the notebook top-to-bottom. This will:
     - Load and preprocess data
     - Fine-tune the model using LoRA
     - Evaluate on the labelled test set
     - Predict on unlabelled test data
     - Save predictions as `submission.csv`

---

## 📊 Weights & Biases Logging

- During training and evaluation, all logs, metrics, and plots (e.g. loss curves, accuracy) are automatically saved to your [Weights & Biases (wandb)](https://wandb.ai/) account.
- Make sure you're logged in using:
  ```bash
  wandb login
  ```
- All plots would be saved on you wandb account.
---

## 📌 Notes

- Model uses MC Dropout for uncertainty-aware predictions.
- LoRA adapters allow for efficient training with minimal additional parameters.
- Default model: `roberta-base`

