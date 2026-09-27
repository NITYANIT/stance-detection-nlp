# ZeroStance: Open-Domain Stance Detection

An implementation and experimental study based on **ZeroStance**, a framework for open-domain stance detection that uses ChatGPT to generate a synthetic, diverse stance detection dataset called **CHATStance**.

The main objective is to build a stance detection model that can generalize to **unseen targets across multiple domains**, rather than being restricted to targets or domains observed during training.

> **Original Paper:**
> *ZeroStance: Leveraging ChatGPT for Open-Domain Stance Detection via Dataset Generation*
> Findings of the Association for Computational Linguistics: ACL 2024

---

# 📌 Table of Contents

* [Overview](#-overview)
* [What is Stance Detection?](#-what-is-stance-detection)
* [In-Domain vs Cross-Domain vs Open-Domain](#-in-domain-vs-cross-domain-vs-open-domain)
* [What is ZeroStance?](#-what-is-zerostance)
* [CHATStance](#-chatstance)
* [Open-Domain Experimental Setup](#-open-domain-experimental-setup)
* [Datasets](#-datasets)
* [Model Architecture](#-model-architecture)
* [Project Workflow](#-project-workflow)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Training](#-training)
* [Evaluation](#-evaluation)
* [Example Predictions](#-example-predictions)
* [Results](#-results)
* [Technologies](#-technologies)
* [Credits and Attribution](#-credits-and-attribution)
* [Citation](#-citation)
* [Acknowledgements](#-acknowledgements)
* [License](#-license)

---

# 🔎 Overview

**Stance detection** is an NLP task that determines the attitude expressed by a text toward a particular target.

Given:

```text
Text + Target
```

the model predicts the stance expressed by the text toward that target.

For example:

| Text                                               | Target            | Stance  |
| -------------------------------------------------- | ----------------- | ------- |
| "Vaccination is important for protecting society." | Vaccination       | FAVOR   |
| "I strongly disagree with this policy."            | Government Policy | AGAINST |
| "The policy was announced yesterday."              | Government Policy | NONE    |

The three stance classes used in this project are:

* **FAVOR** — the text supports the target
* **AGAINST** — the text opposes the target
* **NONE** — the text does not express a clear stance toward the target

---

# 🎯 Project Objective

Traditional stance detection systems often rely heavily on the targets and domains represented in their training data.

For example, a model trained specifically on:

```text
COVID-19 → Vaccination
```

may not automatically generalize well to:

```text
Politics → Elections
```

or:

```text
Environment → Climate Change
```

The **ZeroStance** approach addresses this problem by generating a large synthetic dataset covering a broad range of domains and then training a model on this diverse data.

The resulting model is evaluated on **unseen targets from multiple domains**.

---

# 🧠 What is Stance Detection?

Stance detection is different from simply determining whether a sentence is positive or negative.

The prediction depends on the **relationship between the text and a specific target**.

For example:

```text
Text:
"Electric vehicles are an excellent alternative."

Target:
Electric Vehicles

→ FAVOR
```

The same text could have a different stance toward another target.

Therefore, the task can be represented as:

```text
                ┌──────────────┐
                │     TEXT     │
                └──────┬───────┘
                       │
                       │
                ┌──────▼───────┐
                │    TARGET    │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │    MODEL     │
                └──────┬───────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       FAVOR        AGAINST        NONE
```

---

# 🆚 In-Domain vs Cross-Domain vs Open-Domain

One of the most important concepts in this project is understanding the difference between these three settings.

## 1. In-Domain Stance Detection

In an **in-domain** setting, the training and test data contain the **same targets**.

For example:

```text
TRAIN
COVID19
Targets:
- Vaccination
- Fauci
- School closures

        ↓

TEST
COVID19
Same / known targets
```

The model is therefore evaluated on a target distribution that it has encountered during training.

### Simple representation

```text
KNOWN TARGETS
     ↓
TRAIN
     ↓
MODEL
     ↓
TEST
     ↓
SAME TARGETS
```

The original ZeroStance paper describes this as the traditional stance detection setting.

---

# 2. Cross-Domain / Cross-Target Stance Detection

In **cross-target** stance detection, the model is trained on labeled data associated with one target and evaluated on a different target that was not seen during training.

For example:

```text
TRAIN
Target A
"Donald Trump"

        ↓

MODEL

        ↓

TEST
Target B
"Joe Biden"
```

The destination target is unseen during training.

This is more difficult than ordinary in-domain stance detection because the model has to transfer what it learned from one target to another.

The ZeroStance paper discusses this progression from traditional in-domain stance detection to cross-target and zero-shot stance detection.

---

# 3. Open-Domain Stance Detection

This is the **main task addressed by ZeroStance**.

Open-domain stance detection aims to train a model that can generalize to **unseen targets across multiple domains**.

The important point is:

> The model should not be restricted to one particular domain such as finance, politics, or COVID-19.

Instead, the training data should expose the model to a broad variety of domains so that it can generalize to unseen targets from different domains.

### ZeroStance setup

```text
                 CHATStance
              Synthetic Dataset
                     │
                     │ TRAIN
                     ▼
              ┌─────────────┐
              │   BERTweet   │
              │     Large    │
              └──────┬──────┘
                     │
                     │ TEST
                     ▼
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
      VAST        IBM30K        COVID19
       │             │             │
       ├─────────────┼─────────────┤
       │             │             │
       ▼             ▼             ▼
 SemEval2016       WTWT        P-Stance
```

So, in the **original ZeroStance open-domain experiment**:

```text
TRAIN → CHATStance

TEST →
    VAST
    IBM30K
    COVID19
    SemEval2016
    WTWT
    P-Stance
```

This is the central experimental setup of this project.

---

# 📊 Quick Comparison

| Setting          | Training               | Testing                                | Main Idea                                     |
| ---------------- | ---------------------- | -------------------------------------- | --------------------------------------------- |
| **In-Domain**    | Known targets/domain   | Same targets/domain                    | Generalization within a familiar distribution |
| **Cross-Target** | Target A               | Unseen Target B                        | Transfer between targets                      |
| **Open-Domain**  | Diverse synthetic data | Unseen targets across multiple domains | Broad generalization                          |

### Easy way to remember

```text
IN-DOMAIN
Known → Known

CROSS-TARGET
Target A → Target B

OPEN-DOMAIN
Diverse Training → Unseen Targets across Domains
```

---

# 🤖 What is ZeroStance?

**ZeroStance** is the dataset-generation approach proposed by Zhao et al. for open-domain stance detection.

Instead of relying only on existing human-annotated stance datasets, the authors use **ChatGPT to construct a synthetic dataset called CHATStance** covering a wide range of domains.

The model is then trained on the filtered synthetic dataset and evaluated on unseen targets from diverse domains.

The motivation is to improve generalization beyond the limited domains represented in many existing stance datasets.

---

# 💬 CHATStance

**CHATStance** is the synthetic open-domain stance dataset generated using ChatGPT as part of the ZeroStance framework.

The original ZeroStance repository identifies:

```text
chatgpt_carto_bertweet_var_0.99_seed0
```

as the **final CHATStance dataset after data filtering**.

The repository also contains other CHATStance variants used for ablation studies and analysis.

The final filtered dataset is the primary dataset used for the open-domain model in the original ZeroStance setup.

---

# 🌎 Open-Domain Experimental Setup

The core experiment can be summarized as:

## Step 1 — Generate Diverse Training Data

ChatGPT is used to construct:

```text
CHATStance
```

The dataset is designed to cover a wide range of domains and targets.

---

## Step 2 — Filter the Generated Dataset

The generated data undergoes filtering to produce the final CHATStance version used for training.

```text
Raw CHATStance
      ↓
Data Filtering
      ↓
Final CHATStance
```

---

## Step 3 — Train the Stance Model

The model is trained **only on CHATStance** for the main open-domain experiment.

```text
CHATStance
     ↓
Training
     ↓
Open-Domain Stance Model
```

---

## Step 4 — Evaluate on Unseen Benchmarks

The trained model is evaluated on multiple human-annotated stance datasets:

```text
                    MODEL
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      VAST          IBM30K       COVID19
        │             │             │
        ▼             ▼             ▼
   SemEval2016       WTWT        P-Stance
```

The purpose is to determine whether knowledge learned from the synthetic open-domain dataset transfers to **unseen targets across different domains**.

---

# 📚 Datasets

The ZeroStance evaluation uses six human-annotated benchmark datasets.

## 1. VAST

**VAST** is a stance detection dataset containing diverse topics from debate-style text.

It is one of the benchmark datasets used to evaluate open-domain generalization.

---

## 2. IBM30K

**IBM30K** is another stance detection benchmark used in the ZeroStance evaluation.

---

## 3. COVID19

The COVID19 stance dataset contains stance examples related to COVID-19 topics.

It provides a domain distinct from several of the other evaluation datasets.

---

## 4. SemEval2016

**SemEval-2016 Task 6** is a well-known Twitter stance detection benchmark.

It contains targets including:

* Atheism
* Climate Change
* Donald Trump
* Feminist Movement
* Hillary Clinton
* Legalization of Abortion

The original SemEval task uses three stance categories.

---

## 5. WTWT

**WTWT** is a stance detection dataset used as another benchmark for evaluating generalization.

The ZeroStance paper specifically discusses WTWT as an example of a dataset concentrated within a particular domain and contrasts this with the broader open-domain objective.

---

## 6. P-Stance

**P-Stance** is a large political-domain stance detection dataset.

Its targets include:

* Donald Trump
* Joe Biden
* Bernie Sanders

It provides an important political-domain benchmark for testing whether a model trained on broad synthetic data can generalize to unseen political targets.

---

# 🏗️ Model Architecture

The implementation uses a **BERTweet-large / RoBERTa-large architecture** for stance classification.

The model receives the text and target and predicts one of three stance classes.

```text
                 TEXT
                  +
                TARGET
                  │
                  ▼
        ┌──────────────────┐
        │  BERTweet-Large  │
        │   Transformer    │
        └────────┬─────────┘
                 │
                 ▼
        Classification Layer
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     FAVOR     AGAINST     NONE
```

---

# 🔄 Complete ZeroStance Pipeline

The complete idea can be summarized as:

```text
                ChatGPT
                   │
                   ▼
        Synthetic Data Generation
                   │
                   ▼
              CHATStance
                   │
             Data Filtering
                   │
                   ▼
          Final CHATStance
                   │
                   │
                 TRAIN
                   │
                   ▼
          BERTweet-Large Model
                   │
                   │
                  TEST
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
     VAST        IBM30K      COVID19
       │           │           │
       ├───────────┼───────────┤
       ▼           ▼           ▼
 SemEval2016     WTWT       P-Stance
```

---

# 📁 Project Structure

```text
stance-detection-nlp/
│
├── nlpcovid19.ipynb
│
├── ZeroStance/
│   │
│   ├── config/
│   │   └── config-roberta_large.txt
│   │
│   ├── data/
│   │   ├── vast/
│   │   ├── ibm30k/
│   │   ├── covid19/
│   │   ├── semeval2016/
│   │   ├── wtwt/
│   │   ├── pstance/
│   │   └── chatgpt_carto_bertweet_var_0.99_seed0/
│   │
│   └── src/
│       ├── train_model_v2.py
│       ├── pytorchtools.py
│       └── utils/
│
└── README.md
```

---

# ⚙️ Installation

Clone this project:

```bash
git clone https://github.com/NITYANIT/stance-detection-nlp.git
cd stance-detection-nlp
```

Clone the original ZeroStance repository:

```bash
git clone https://github.com/chenyez/ZeroStance.git
```

Install the required Python packages:

```bash
pip install -U transformers sentencepiece tweet-preprocessor wordninja tensorboard nltk
```

---

# 🤗 Pretrained Model

The project uses:

```text
vinai/bertweet-large
```

The model can be downloaded using Hugging Face:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vinai/bertweet-large",
    local_dir="/content/model_hub/bertweet-large"
)
```

---

# 🏋️ Training

The main ZeroStance open-domain experiment trains the model on **CHATStance** and evaluates it on the six benchmark datasets.

The exact training configuration should follow the configuration files and scripts provided by the original ZeroStance implementation.

Example configuration:

```text
model_select:RoBERTa
bert_lr:1e-5
fc_lr:1e-5
batch_size:8
total_epochs:4
max_tok_len:250
dropout:0.1
```

> **Note:** The batch size and number of epochs above reflect the experimental configuration used in this implementation and may differ from the original authors' exact hardware/configuration.

---

# 🧪 Evaluation

The model is evaluated independently on:

```text
VAST
IBM30K
COVID19
SemEval2016
WTWT
P-Stance
```

The primary evaluation metric is **Macro-F1**.

Macro-F1 calculates the F1 score independently for each stance class and then averages them:

```text
Macro-F1 =
(F1_FAVOR + F1_AGAINST + F1_NONE) / 3
```

This gives equal importance to all three stance classes.

---

# 📝 Example Predictions

## Example 1 — FAVOR

```text
Text:
"Vaccination is essential for protecting people."

Target:
Vaccination

Prediction:
FAVOR
```

---

## Example 2 — AGAINST

```text
Text:
"I strongly disagree with this policy."

Target:
Government Policy

Prediction:
AGAINST
```

---

## Example 3 — NONE

```text
Text:
"The policy was announced on Monday."

Target:
Government Policy

Prediction:
NONE
```

---

# 🔬 Why This is Open-Domain

The important property of the ZeroStance setup is **not simply using multiple datasets during training**.

The key idea is:

```text
TRAINING
     │
     ▼
Diverse synthetic CHATStance
     │
     ▼
Model learns general stance patterns
     │
     ▼
TESTING
     │
     ├── Unseen targets
     ├── Different domains
     └── Multiple benchmark datasets
```

This differs from simply combining several existing datasets and testing on another dataset.

The original paper specifically defines open-domain stance detection as generalization to **unseen targets across multiple domains**.

---

# 📊 Results

Results from the experiments can be added below once the final runs are completed.

| Dataset     | Domain                    | Macro-F1 |
| ----------- | ------------------------- | -------: |
| VAST        | Diverse debate topics     |        — |
| IBM30K      | Stance benchmark          |        — |
| COVID19     | COVID-19                  |        — |
| SemEval2016 | Social / political topics |        — |
| WTWT        | Domain-specific stance    |        — |
| P-Stance    | Political                 |        — |

### Overall Evaluation

The final results should be reported separately for each benchmark dataset rather than combining all datasets into a single score.

This makes it possible to observe how the model generalizes across different domains.

---

# 🧩 In-Domain Experiment in This Project

For comparison, an **in-domain** experiment can also be performed.

For example:

```text
TRAIN → CHATStance
TEST  → CHATStance
```

or, depending on the selected dataset:

```text
TRAIN → COVID19
TEST  → COVID19
```

This provides a baseline for comparison with the more challenging unseen-target evaluation.

---

# 🔀 Cross-Domain Experiment

A cross-domain experiment can be represented as:

```text
TRAIN
Domain A
   │
   ▼
 MODEL
   │
   ▼
TEST
Domain B
```

For example:

```text
TRAIN → COVID19
TEST  → P-Stance
```

Here, the model is trained on one domain and evaluated on another.

This is different from the original ZeroStance open-domain experiment because ZeroStance specifically uses **CHATStance as the synthetic training resource** and evaluates generalization across multiple unseen targets/domains.

---

# ⭐ Main Idea of the Project

The project demonstrates the progression:

```text
Traditional Stance Detection
            │
            ▼
       In-Domain
       Known Targets
            │
            ▼
      Cross-Target
      Unseen Target
            │
            ▼
      Open-Domain
Unseen Targets + Multiple Domains
            │
            ▼
         ZeroStance
            │
            ▼
       CHATStance
            │
            ▼
Generalization to Benchmark Datasets
```

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **BERTweet-large**
* **RoBERTa**
* **Pandas**
* **NumPy**
* **NLTK**
* **TensorBoard**
* **CUDA / Google Colab**
* **Git / GitHub**

---

# 👥 Credits and Attribution

This project is based on the research and implementation released by the authors of **ZeroStance**.

### Original Authors

* **Chenye Zhao**
* **Yingjie Li**
* **Cornelia Caragea**
* **Yue Zhang**

The original work was published in the **Findings of the Association for Computational Linguistics: ACL 2024**.

Original repository:

https://github.com/chenyez/ZeroStance

The original authors should be credited for:

* The ZeroStance methodology
* CHATStance dataset generation
* The open-domain stance detection formulation
* The original implementation
* The experimental framework

This repository represents an implementation/experimental study based on those publicly released resources.

---

# 📖 Original Research Paper

**Zhao, Chenye; Li, Yingjie; Caragea, Cornelia; Zhang, Yue.**

> **ZeroStance: Leveraging ChatGPT for Open-Domain Stance Detection via Dataset Generation**

*Findings of the Association for Computational Linguistics: ACL 2024*, pages 13390–13405.

Paper:

https://aclanthology.org/2024.findings-acl.794/

Original repository:

https://github.com/chenyez/ZeroStance

---

# 📚 Citation

If you use the ZeroStance methodology, implementation, or CHATStance dataset, please cite the original paper:

```bibtex
@inproceedings{zhao-etal-2024-zerostance,
    title = "{Z}ero{S}tance: Leveraging {C}hat{GPT} for Open-Domain Stance Detection via Dataset Generation",
    author = "Zhao, Chenye and
              Li, Yingjie and
              Caragea, Cornelia and
              Zhang, Yue",
    editor = "Ku, Lun-Wei and
              Martins, Andre and
              Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    pages = "13390--13405",
    url = "https://aclanthology.org/2024.findings-acl.794/"
}
```

---

# 🙏 Acknowledgements

We acknowledge the authors of **ZeroStance** for releasing the research code and data that make experimentation with open-domain stance detection possible.

We also acknowledge the creators of the benchmark datasets used for evaluation.

---

# ⚠️ Dataset and Code Attribution

The datasets used in this project originate from their respective research works and repositories.

Users should consult the original sources and licenses before redistributing datasets or modified versions of the original code.

This project does not claim ownership of the original ZeroStance methodology, CHATStance dataset, or third-party benchmark datasets.

---

# 📜 License

Please refer to the original ZeroStance repository and the individual dataset licenses for the applicable usage and redistribution terms.

---

# 🔗 References

### ZeroStance

Zhao, C., Li, Y., Caragea, C., & Zhang, Y. (2024).

**ZeroStance: Leveraging ChatGPT for Open-Domain Stance Detection via Dataset Generation.**

Findings of ACL 2024.

### Original Repository

https://github.com/chenyez/ZeroStance

### SemEval-2016

Mohammad et al. (2016).

**SemEval-2016 Task 6: Detecting Stance in Tweets.**

---

# 📌 Final Project Summary

```text
                    ZEROStance
                        │
                        ▼
               ChatGPT-based
             Dataset Generation
                        │
                        ▼
                  CHATStance
               Synthetic Dataset
                        │
                        │ TRAIN
                        ▼
                BERTweet-Large
                        │
                        │ TEST
                        ▼
       ┌────────────────────────────────┐
       │                                │
       ▼                                ▼
 Unseen Targets                  Multiple Domains
       │                                │
       └───────────────┬────────────────┘
                       ▼
               Benchmark Datasets
                       │
       ┌───────┬───────┼───────┬───────┐
       ▼       ▼       ▼       ▼       ▼
     VAST   IBM30K  COVID19  WTWT  P-Stance
                       │
                       ▼
                  SemEval2016
```

## 🎯 Core Objective

> **Train on a diverse synthetic open-domain dataset (CHATStance) and evaluate whether the resulting model can generalize to unseen targets across multiple domains.**

This is the central idea behind the **ZeroStance** approach.
