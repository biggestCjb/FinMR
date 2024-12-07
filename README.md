# Enhancing Multimodal Financial Math Reasoning withReflection Module and Error Log🔥 Dataset

🌟  This is the official repository for the paper "[Enhancing Multimodal Financial Math Reasoning withReflection Module and Error Log](https://arxiv.org/pdf/2402.14804.pdf)", which contains both evaluation code and data for the **MATH-V** benchmark.

[[🌐 Homepage]] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/aminous1/FinMR)]  [[📖 ArXiv Paper]]


## 👀 Introduction

This project introduces an innovative approach to enhance QA systems in financial reasoning tasks through the integration of a reflection module and error log. We apply this method to a Financial Math Reasoning dataset, which features multimodal reasoning challenges. Our system retrieves past errors, provides corrective feedback, and stores mistakes for long-term improvement. The experiments show significant gains in reasoning accuracy, particularly when dealing with complex financial questions.


<p align="center">
    <img src="Problem_Formalization_2.png" width="66%">  Each incorrect prediction, along with its corrected reasoning and feedback, is stored in the error database Derror. This database serves as a dynamic repository of past errors,
enabling the model to continuously learn and improve by incorporating insights from previousmistakes.
</p>
You can refer to the [project homepage] for more details.

Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on MATH-V, underscoring the imperative for further advancements in LMMs.



You can refer to our [project homepage]() and [the paper]() for more details.

## 📐 Dataset Examples



You can refer to the Appendix D.3 of [the paper](https://arxiv.org/pdf/2402.14804.pdf) for example images of 16 subjects.

## 🏆 Leaderboard

The leaderboard is available [here](https://mathllm.github.io/mathvision/#leaderboard).



## 📈 Evaluation

### Generating Outputs of Different Models

#### Gemini

`python models/Gemini.py --in_path ./data/test.jsonl --save_path ./Gemini.jsonl`

This will run the Gemini API and save the outputs to `./Gemini.jsonl` path. You can modify the system prompt, max tokens, etc. in the `benchmark_gemini` function.

#### GPT_with_caption

Generate image captions using GPT-4V:

`python models/GPT_with_caption.py --model gpt-4-vision-preview --in_path ./data/test.jsonl --save_path ./data/gpt4v-captions.jsonl`

Generate outputs using ChatGPT-3.5 or GPT-4 with image captions:

`python models/GPT_with_caption.py --model gpt-3.5-turbo-0125 (gpt-4-turbo-preview) --in_path ./data/test.jsonl --save_path ./gpt3.5_caption.jsonl (./gpt4_caption.jsonl)`



### Evaluation of Model Outputs

Once all the model outputs have been generated, execute the `python evaluation/evaluate.py` function to assess these outputs. This script will examine all outputs located in the `outputs/` directory, computing overall accuracy as well as accuracy for each subject and level.

You can refer to the Appendix E and F of [the paper](https://arxiv.org/pdf/2402.14804.pdf) for some evaluation results of the above models.

## 📝 Citation

If you find this benchmark useful in your research, please consider citing this BibTex:

```
    @article{article_id,
    title     = {Enhancing Multimodal Financial Math Reasoning with Reflection Module and Error Log},
    author    = {Shuangyan Deng, Haizhou Peng, ChunHou Liu, Jiachen Xu},
    year      = {2024},
    journal   = {arXiv},
    primaryClass={cs.CV}
    }
```

## 🧠 Related Work
