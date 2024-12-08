# FinMR: A Novel Knowledge-Intensive Multimodal Reasoning Benchmark in Financial Domain🔥 Dataset
🌟 This repo includes an original implementation of Shuangyan Deng, Haizhou Peng, ChunHou Liu, Jiachen Xu ,Zhuoheng Li

["FinMR: A Novel Knowledge-Intensive Multimodal Reasoning Benchmark in Financial Domain"](https://arxiv.org) , which contains both evaluation code and data for the **FinMR** benchmark.

[[🌐 Homepage](https://jason08022000.github.io/financial-reasoning.github.io/)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/aminous1/FinMR)]   [[📖 ArXiv Paper](https://arxiv.org)]



## 📖 Introduction
We present FinMR, a novel dataset designed to evaluate and enhance financial reasoning capabilities in multimodal models. FinMR includes 3,200 expert-authored examples with detailed annotations covering both expertise and mathematical reasoning. This dataset addresses the unique challenges of the financial domain, such as complex numerical calculations and domain-specific knowledge-intensive tasks.
Additionally, we propose a training-free framework for generating dynamic negative examples with AI-driven feedback, which effectively enhances reasoning performance and supports iterative learning. FinMR serves as a valuable resource for advancing multimodal financial reasoning research.

![FinMR schematics](https://img520.com/MBZ94b.png)
**Figure**: *Illustrating a multimodal reasoning framework for answering questions step-by-step with iterative feedback. 
The system takes multimodal inputs (text and images) and generates reasoning steps along with an answer inference. 
If the output is correct, the reasoning and answer are accepted. If incorrect, the process involves rationale step 
correction via manual annotation and error analysis. Feedback generation introduces negative examples and AI feedback to 
improve reasoning. The improved insights are used for in-context learning, enhancing the reasoning steps and answer inference. 
This approach emphasizes iterative improvement and error correction for better accuracy.*


You can refer to our [[project homepage](https://jason08022000.github.io/financial-reasoning.github.io/)] and [[the paper](https://arxiv.org)] for more details.

## 📊 Dataset Examples
<details>
<summary>Topics in Our Dataset</summary><p align="center">
    <img src="https://img520.com/12EYvx.png" width="50%"> <br>
</p></details>

<details>
<summary>Various QA Types</summary><p align="center">
    <img src="https://img520.com/BR0HLG.png" width="50%"> <br>
</p></details>

<details>
<summary>Question Sample</summary><p align="center">
    <img src="https://img520.com/wzIe0D.png" width="50%"> <br>
</p></details>

<details>
    <summary>Image Sample</summary><p align="center">
    <img src="https://img520.com/hts5nu.jpg" width="70%"> <br>
</p></details>

## 🏆 Leaderboard

The leaderboard is available [here](https://mathllm.github.io/mathvision/#leaderboard).

## :wrench: Installation
Our implementation is based on PyTorch and HuggingFace (transformers + datasets). 

Requirements:
* Python 3.8.5
* Pytorch 1.10.0
* transformers 4.20.0
* datasets 2.3.2
* tqdm, sklearn, numpy, pandas

Step-by-step instructions to get you code and environment:
### 1) Clone this repository to your local machine:

```bash
git clone https://github.com/biggestCjb/FinMR.git    
```

A folder called ```FinMR``` with all the codebase should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://docs.conda.io/en/latest/miniconda.html).

To run Progressive Prompts code on GPU, make sure that you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. In our implementation, we used and CUDA 12.1.

You can re-create our conda enviroment from ```environment.yaml``` file:

```bash
cd FinMR
conda env create -f environment.yaml
```

Your conda should start downloading and extracting packages.

### 3) Activate the environment:

Your environment should be called ```FinMR```, and you can activate it now to run the scripts:

```bash
conda activate FinMR
```
### 4) Get model weight:
Download the weights of the model from the link below

```
https://drive.google.com/file/d/1s41RoUx8L65EVOkEo-g73e2Mc8jTGdVb/view?usp=sharing
```
Extract the downloaded zip one level below the root directory of the project, which is our code structure:

```
|_FinMR/
    ├─ask_LLM
    │  ├─Multimodal        
    │  │  ├─models
    │  │  └─tools
    │  │      
    │  └─text
    │  │  ├─models
    │  │  └─tools
    ├─data
    ├─errorLog
    │  ├─Multimodal
    │  │  ├─claude
    │  │  ├─gemini
    │  │  ├─gpt
    │  │  ├─Llama
    │  │  ├─llava
    │  │  └─qwen
    │  └─text
    │  │  ├─claude
    │  │  ├─gemini
    │  │  ├─gpt
    │  │  ├─Llama
    │  │  ├─llava
    │  │  └─deepseek
    ├─evaluation
    ├─images
    ├─model_weight ——>*You should unzip here!
    │  └─clip-vit-base-patch32
    └─outputs
        ├─Multimodal
        │  ├─claude
        │  ├─gemini
        │  ├─gpt
        │  ├─Llama
        │  ├─Llava
        |  |─qwen
        └─text
        │  ├─claude
        │  ├─gemini
        │  ├─gpt
        │  ├─Llama
        │  ├─deepseek
```

## 📈 How to Evaluation

### Generating Outputs of Different Models

#### Claude

```bash
cd ask_LLM/Multimodal/models
python Claude_3.5.py
```

This will run the Claude API and save the outputs to `outputs/Multimodal/claude` path. 

#### GPT

```bash
cd ask_LLM/Multimodal/models
python GPT_4o.py
```
This will run the GPT API and save the outputs to `outputs/Multimodal/gpt` path. 

### Evaluation of Model Outputs

Once all the model outputs have been generated, execute the `evaluation/evaluate.ipynb` function to assess these outputs. This will examine all outputs located in the `outputs/` directory, computing overall accuracy as well as accuracy for each subject and level.

**Here is our result:**
![FinMR schematics](https://img520.com/0GmPiK.jpg)
![FinMR schematics](https://img520.com/KqxI4k.png)

You can refer to the Appendix of [the paper](https://arxiv.org) for some evaluation results of the above models.

## 📝 Citation

If you find this benchmark useful in your research, please consider citing this BibTex:

```
    @article{article_id,
    title     = {Enhancing Multimodal Financial Math Reasoning with Reflection Module and Error Log},
    author    = {Shuangyan Deng, Haizhou Peng, ChunHou Liu, Jiachen Xu, Zhouheng Li},
    year      = {2024},
    journal   = {arXiv},
    primaryClass={cs.CV}
    }
```

## 🧠 Related Work

# Still under maintenance...🔧✨
