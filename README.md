<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: Workshop-Level Automated</b><br>
    <b>Scientific Discovery via Agentic Tree Search</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2026-Workshop-Experiment"> [ICLR2026 Workshop Experiment]</a>
</p>

Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor (AI Scientist-v1)](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across Machine Learning (ML) domains, and employs a progressive agentic tree search, guided by an experiment manager agent.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. v1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. v1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

> **Caution!**
> This codebase will execute Large Language Model (LLM)-written code. There are various risks and challenges associated with this autonomy, including the potential use of dangerous packages, uncontrolled web access, and the possibility of spawning unintended processes. Ensure that you run this within a controlled sandbox environment (e.g., a Docker container). Use at your own discretion.

## Table of Contents

1.  [Requirements](#requirements)
    *   [Installation](#installation)
    *   [Model Configuration](#model-configuration)
2.  [Generate Research Ideas](#generate-research-ideas)
3.  [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments)
4.  [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)
5.  [Frequently Asked Questions](#frequently-asked-questions)
6.  [Acknowledgement](#acknowledgement)

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch.

### Installation

```bash
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt
```

Installation usually takes no more than one hour.

### Model Configuration

All model configurations are managed centrally in `config.yaml`. This includes API endpoints, API keys, and model names.

#### Configuring Models

Edit `config.yaml` to configure your models. Each model type has the following fields:

```yaml
models:
  llm:
    client_type: openai          # or 'anthropic'
    base_url: https://api.openai.com/v1
    api_key: "your-api-key-here"
    model_name: gpt-4o-2024-11-20
```

Available model types:
- `llm` - Main language model for general tasks
- `vlm` - Vision-language model for image understanding
- `code` - Model for code generation
- `plot_aggregation` - Model for plot aggregation tasks
- `writeup` - Model for paper writing
- `citation` - Model for citation collection
- `small_model` - Smaller/faster model for simple tasks
- `review` - Model for reviewing tasks

#### Supported Providers

**OpenAI-compatible APIs**
- Set `client_type: openai`
- Configure `base_url` for any OpenAI-compatible endpoint (OpenAI, DeepSeek, vLLM, etc.)
- Set `api_key` directly in config.yaml

**Anthropic (Claude)**
- Set `client_type: anthropic`
- For standard Anthropic API: leave `base_url` empty or omit
- For AWS Bedrock: configure `base_url` with your Bedrock endpoint
- Set `api_key` directly in config.yaml

#### Literature Search APIs

The system uses **OpenAlex** by default for literature search during ideation and paper writing stages, with Semantic Scholar as a fallback. You can configure which service to use as the primary search tool.

**Configuring the Default Search Tool**

Set `academic_search.default_tool` in `config.yaml` to choose which service to use first:

```yaml
academic_search:
  default_tool: "open_alex"  # Options: "open_alex" or "semantic_scholar"
```

The system will use the configured default tool first, and automatically fall back to the other service if it fails.

**OpenAlex Configuration**

[OpenAlex](https://openalex.org/) is an open online bibliographic database.

```yaml
open_alex:
  api_key: "your-openalex-key-here"  # Optional but recommended for higher rate limits
  email: "your-email@example.com"     # Optional but recommended for rate limiting
  base_url: "https://api.openalex.org"  # API endpoint URL
```

**Semantic Scholar Configuration**

[Semantic Scholar](https://www.semanticscholar.org/product/api) is used as a fallback if OpenAlex search fails or returns no results.

```yaml
semantic_scholar:
  api_key: "your-s2-key-here"
  base_url: "https://api.semanticscholar.org/graph/v1/paper/search"  # API endpoint URL
```

> **Note:** If you don't have API keys, the system will still work but may be subject to stricter rate limits. You can also skip the citation phase during paper generation if you experience issues with the APIs.

## Generate Research Ideas

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `llm_inference_optimization.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/llm_inference_optimization.md" \
     --model llm \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas (ensure you have the corresponding API key set).
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/llm_inference_optimization.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.

## Run AI Scientist-v2 Paper Generation Experiments

Using the JSON file generated in the previous ideation step, you can now launch the main AI Scientist-v2 pipeline. This involves running experiments via agentic tree search, analyzing results, and generating a paper draft.

Specify the models used for the write-up and review phases via command-line arguments.
The configuration for the best-first tree search (BFTS) is located in `bfts_config.yaml`. Adjust parameters in this file as needed.
Model types specified via command-line arguments (e.g., `--model_writeup writeup`) reference the model type keys defined in `config.yaml`.

Key tree search configuration parameters in `bfts_config.yaml`:

-   `agent` config:
    -   Set `num_workers` (number of parallel exploration paths) and `steps` (maximum number of nodes to explore). For example, if `num_workers=3` and `steps=21`, the tree search will explore up to 21 nodes, expanding 3 nodes concurrently at each step.
    -   `num_seeds`: Should generally be the same as `num_workers` if `num_workers` is less than 3. Otherwise, set `num_seeds` to 3.
    -   Note: Other agent parameters like `k_fold_validation`, `expose_prediction`, and `data_preview` are not used in the current version.
    -   Model references (`code`, `feedback`, `vlm_feedback`) use model type keys from `config.yaml`.
-   `search` config:
    -   `max_debug_depth`: The maximum number of times the agent will attempt to debug a failing node before abandoning that search path.
    -   `debug_prob`: The probability of attempting to debug a failing node.
    -   `num_drafts`: The number of initial root nodes (i.e., the number of independent trees to grow) during Stage 1.

> **Note:** Before running experiments, ensure you have configured valid API keys in `config.yaml` for the model types you intend to use.

Example command to run AI-Scientist-v2 using a generated idea file (e.g., `llm_inference_optimization.json`). Review `bfts_config.yaml` for detailed tree search parameters and ensure your API keys are configured in `config.yaml`. Do not set `load_code` if you do not want to initialize experimentation with a code snippet.

```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/llm_inference_optimization.json" \
 # --load_code \ ## 没有模板代码
 --add_dataset_ref \
 --model_writeup writeup \
 --model_citation citation \
 --model_review review \
 --model_agg_plots plot_aggregation \
 --num_cite_rounds 20 \ 
 --page_limit 20
```

Once the initial experimental stage is complete, you will find a timestamped log folder inside the `experiments/` directory. Navigate to `experiments/"timestamp_ideaname"/logs/0-run/` within that folder to find the tree visualization file `unified_tree_viz.html`.
After all experiment stages are complete, the writeup stage begins. The writeup stage typically takes about 20 to 30 minutes in total. Once it finishes, you should see `timestamp_ideaname.pdf` in the `timestamp_ideaname` folder.
For this example run, all stages typically finish within several hours.

## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite our work as follows:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2026}
}
```

## Frequently Asked Questions

**Why wasn't a PDF or a review generated for my experiment?**

The AI Scientist-v2 completes experiments with a success rate that depends on the chosen foundation model, and the complexity of the idea. Higher success rates are generally observed when using powerful models like Claude 3.5 Sonnet for the experimentation phase.

**What is the estimated cost per experiment?**

The ideation step cost depends on the LLM used and the number of generations/reflections, but is generally low (a few dollars). For the main experiment pipeline, costs vary based on the model type configured in `config.yaml`. Using powerful models like Claude 3.5 Sonnet for the experimentation phase typically costs around $15–$20 per run. The subsequent writing phase adds approximately $5 when using the default models specified in the example command. Using GPT-4o for `model_citation` is recommended as it can help reduce writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

First, perform the [Generate Research Ideas](#generate-research-ideas) step. Create a new Markdown file describing your desired subject field or topic, following the structure of the example `ai_scientist/ideas/i_cant_believe_its_not_better.md`. Run the `perform_ideation_temp_free.py` script with this file to generate a corresponding JSON idea file. Then, proceed to the [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments) step, using this JSON file with the `launch_scientist_bfts.py` script via the `--load_ideas` argument.

**What should I do if I have problems accessing the literature search APIs?**

The system uses **OpenAlex** by default for literature search during ideation and paper writing, with **Semantic Scholar** as a fallback. These APIs are used to assess the novelty of generated ideas and to gather citations during the paper write-up phase.

- If you don't have API keys, the system will still work but may be subject to stricter rate limits
- If one API fails or returns no results, the system automatically tries the other
- You can skip the citation phase during paper generation if you experience persistent issues with both APIs

**I encountered a "CUDA Out of Memory" error. What can I do?**

This error typically occurs when the AI Scientist-v2 attempts to load or run a model that requires more GPU memory than available on your system. To resolve this, you can try updating your ideation prompt file (`ai_scientist/ideas/llm_inference_optimization.md`) to suggest using smaller models for the experiments.

## Acknowledgement

The tree search component implemented within the `ai_scientist` directory is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. We thank the AIDE developers for their valuable contributions and for making their work publicly available.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

## 修改标题和作者
### icbinb
修改```ai_scientist/blank_icbinb_latex/icbinb.sty``` :
- 第 95 行, 标题: ```Under review as a workshop paper at ICLR 2026```
- 第 100 行, 作者: ```Anonymous authors\\Paper under double-blind review```

### icml
修改```ai_scientist/blank_icml_latex/icml.sty```: 
- 第 129-130 行,匿名评审版本(默认), 页脚: ```Preliminary work.  Under review by the
International Conference on Machine Learning (ICML)\@.  Do not distribute.```
- 第 119-122 行,正式审核版本, 页脚: ```{Proceedings of the
$\mathit{41}^{st}$ International Conference on Machine Learning},
Vancouver, Canada. PMLR 267, 2026.
Copyright 2026 by the author(s).```
- 第 482 行, 作者: ```Anonymous Authors``` 
- 第 513 行, 单位: ```Anonymous Institution...```
- 第 525 行, 作者信息: ```Anonymous Author <anon.email@domain.com>```

## 语言
批量替换 ```prompt.yaml``` 中 ```Chinese(中文)```,修改需要的语言,例如: ```English(英文)```

## qwen3.5-plus 提示词
- 修改整个项目的模型调用,不要在硬编码模型名称,而是通过config.yaml中的配置的模型类型,写个函数统一读取config.yaml文件,获取模型的主要配置,包括api_key,base_url,model_name, 现有代码传递的model修改为model_type,方法内根据model_type获取api_key,base_url,model_name.模型的配置从配置文件读取,不要读取环境变量.
- 去掉ollama支持,去掉max_tokens参数.
- bfts_config.yaml中配置的模型名称,修改为config.yaml中的model_type值,并修改对应的代码.
- 所有模型的调用都根据model_type读取config.yaml的api_key,base_url,model_name.
- 不再计算模型的价格,只记录消耗的token.
- 类似 if model_name.startswith("o1") or model_name.startswith("o3"): 这种模型判断都要去掉,不要硬编码处理特定模型
- get_ai_client 要返回client和model_name,并修改用到的代码逻辑
- 模型调用参数不要把model_type当成model使用.
- 检索论文时 增加 open_alex 论文库,使用 pyalex 库进行检索,最后的格式和内容要和现在使用的semantic_scholar保持一致,就是等于多了一种论文检索方式,其他逻辑保持不变,默认使用 open_alex 进行论文检索,open_alex的配置在config.yaml

base_url = "https://api.openalex.org/works" 应该从config.yaml中读取,通过 academic_search:
  default_tool: "open_alex"  设置使用哪个工具检索,默认使用 open_alex进行检索

semantic_scholar 的 base_url 设置为完整的 https://api.semanticscholar.org/graph/v1/paper/search, 也要从配置文件读取,不要硬编码

- open_alex 返回的数据格式要和semantic_scholar 保持一致

- Stage 4 (消融实验): 可能有多个子阶段,因为 AI 可能根据实验进展决定追加更多消融分析,将多个completed正常完成的消融实验结果融合

- 在 Journal 类中添加了 completed: bool = field(default=True) 属性，默认值为 True，表示阶段已完成。这样在生成最终报告时，所有阶段都会被正常处理。

- 创建  prompt.yaml ,把硬编码的提示词按照层级放到 prompt.yaml 中,使用中文注释,根据key读取,不要修 prompt的内容,只是从硬编码修改成从 prompt.yaml 读取,不要对业务逻辑有任何影响

- 论文模板中添加了```\usepackage{ctex}```支持中文,要安装 XeLaTeX,原来的 pdflatex 对中文支持不好


Method章节是全论文最核心的一章节：要使用框架图、原理图、公式，伪代码等力求详尽介绍自己的创新方法。
Experiments章节介绍实验过程，要求分别进行定量和消融实验，说明使用了哪些数据集，通过图表直观展示实验效果。
a）定量实验：使用学界公认的一些定量指标。
b）消融实验：在自己的方法上面做控制变量法，比如自己的创新点有1，2，3三点，去掉1后实验，发现指标下降，就说明1必不可少。同样去掉2、3进行相同实验。也可以进行去掉1+2或者2+3或者1+3进行相同实验和原模型比较，证明创新点的效果。





