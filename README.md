# Search MAS (Inference-Only)

This repository contains an inference-only multi-agent system (MAS) extracted
from the DrMAS `search` RL use case. It is the reference external MAS for the
[uni-agent](https://github.com/verl-project/uni-agent) `examples/multi_policy`
recipe. It keeps only:

- multi-agent orchestration (Verifier routing + Search/Answer execution);
- LLM inference through OpenAI-compatible endpoints;
- the bundled dataset and black-box validation.

It does not include training logic or training dependencies.

## Layout

```text
search_mas_app
├── configs/
│   ├── inference.yaml          # multi_policy training subprocess template
│   └── search_mas_example.yaml # standalone example config
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── deploy_searchr1_retrieval_service.sh
│   ├── env_mas_runtime.sh
│   ├── retrieval_server.py
│   ├── run_search_mas.py
│   ├── searchr1_download.py
│   └── validate_search_mas.py
├── search_mas/
│   ├── core/              # shared config/LLM/agent base classes
│   └── apps/search/       # Search MAS implementation
├── test_sampled.parquet   # 210 evaluation samples (30 per dataset)
└── train.parquet          # 169,615 NQ + HotpotQA training rows
```

## Agent flow

1. The Verifier Agent decides whether the accumulated context is sufficient:
   `<verify>yes|no</verify>`.
2. If `no`, the Search Agent emits `<search>query</search>`, calls the
   retrieval service, and appends the result to the
   `<information>...</information>` context.
3. If `yes`, the Answer Agent emits `<answer>final answer</answer>`.
4. If the maximum number of turns is reached before the Verifier routes to the
   Answer Agent, the application can force one Answer turn
   (`force_final_answer_on_max_turn=true`).

## Quick start

### 1) Install dependencies

```bash
cd /path/to/search_mas_app
python -m pip install -r requirements.txt
```

### 2) Dataset (bundled)

The repository ships both Parquet files, so no download or conversion is
required:

| File | Rows | Content |
| --- | --- | --- |
| `train.parquet` | 169,615 | NQ (79,168) + HotpotQA (90,447) training split |
| `test_sampled.parquet` | 210 | 30 samples from each of seven evaluation sets (nq/triviaqa/popqa/hotpotqa/2wikimultihopqa/musique/bamboogle) |

Rows use the verl schema: `data_source`, `prompt`, `reward_model`,
`extra_info`, and `uid`. `extra_info.ground_truth_raw.target` holds the clean
answer list; `reward_model.ground_truth` can contain multiple answers
concatenated into one stringified list, so training and evaluation prefer the
former.

The data derives from Search-R1's
[`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train).
To regenerate or extend it, use the Search-R1
[`nq_search.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py)
preprocessing script.

### 3) Start the DrMAS SearchR1 retrieval service

`faiss-gpu` cannot be installed with pip, so the retrieval server needs its own
conda environment. Deploy it on separate machine/GPU resources from the training
system; the MAS communicates with it over HTTP. The server uses roughly 6 GB of
GPU memory per GPU on the retrieval host.

#### 3.1) Create the retriever environment

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install transformers datasets pyserini
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

#### 3.2) Download the index and corpus

```bash
conda activate retriever

local_dir=~/data/searchR1
python scripts/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

#### 3.3) Start the e5 retrieval service

```bash
conda activate retriever

# Redirect output to a file: logging to the terminal can spike response latency.
python scripts/retrieval_server.py \
  --index_path $local_dir/e5_Flat.index \
  --corpus_path $local_dir/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu \
  --port 8010 \
  > retrieval_server.log
```

Point the MAS at the service:

```bash
export SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://<retrieval-host>:8010/retrieve
```

> **Note:** When this application runs as the external MAS for uni-agent, deploy
> the retrieval service on a separate host and set
> `SEARCH_MAS_RETRIEVAL_SERVICE_URL` to an address reachable from the Ray workers
> that run the MAS (inject it through `launcher_kwargs.command.env` in the
> uni-agent Task Config). `127.0.0.1` would refer to the worker node itself, so
> do not use it unless the service runs there.

Quick check:

```bash
curl -X POST http://127.0.0.1:8010/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"Who won the 2022 FIFA World Cup?","topk":3}'
```

**Notes:**
- If the environment and data are already prepared, you can use the deployment
  helper: `bash scripts/deploy_searchr1_retrieval_service.sh`.
- It defaults to `~/data/searchR1`, port `8010`, and `intfloat/e5-base-v2`.
- Override with `SEARCH_MAS_SEARCHR1_LOCAL_DIR`, `SEARCH_MAS_SEARCHR1_PORT`, and
  `SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL`.

### 4) Configure the LLM (OpenAI / vLLM)

The application lets environment variables override the YAML LLM settings, which
is the recommended way to switch between OpenAI and a local vLLM endpoint.

Global variables (they take priority over YAML):
- `SEARCH_MAS_LLM_BASE_URL`
- `SEARCH_MAS_LLM_API_KEY`
- `SEARCH_MAS_LLM_MODEL`

Compatible aliases:
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Optional overrides:
- Per-agent backends: `SEARCH_MAS_VERIFIER_LLM_*`,
  `SEARCH_MAS_SEARCHER_LLM_*`, `SEARCH_MAS_ANSWERER_LLM_*`.
- Retrieval URL: `SEARCH_MAS_RETRIEVAL_SERVICE_URL`.

OpenAI example:

```bash
export SEARCH_MAS_LLM_BASE_URL=https://api.openai.com/v1
export SEARCH_MAS_LLM_API_KEY=sk-xxx
export SEARCH_MAS_LLM_MODEL=gpt-4.1-mini
```

Local vLLM example:

```bash
export SEARCH_MAS_LLM_BASE_URL=http://127.0.0.1:8000/v1
export SEARCH_MAS_LLM_API_KEY=EMPTY
export SEARCH_MAS_LLM_MODEL=/path/to/Qwen3-4B-Instruct-2507
```

You can also source the helper script:

```bash
source scripts/env_mas_runtime.sh
```

### 5) Single-question inference

```bash
python scripts/run_search_mas.py \
  --config configs/search_mas_example.yaml \
  --question "Who won the 2014 FIFA World Cup?"
```

### 6) Black-box validation (full pass + scoring + accuracy)

```bash
python scripts/validate_search_mas.py \
  --config configs/search_mas_example.yaml \
  --input_file ./test_sampled.parquet
```

Outputs:
- `output.prediction_path`: per-sample predictions and differences.
- `output.report_path`: overall accuracy and per-source statistics.

## Configuration reference

`configs/search_mas_example.yaml` is the standalone example configuration;
`configs/inference.yaml` is the template used by the uni-agent `multi_policy`
recipe when it launches the external MAS subprocess.

Key fields in `configs/search_mas_example.yaml`:
- `llm.base_url` + `llm.api_key` + `llm.model`: OpenAI-compatible endpoint
  (OpenAI or vLLM).
  - Recommended env overrides: `SEARCH_MAS_LLM_BASE_URL`,
    `SEARCH_MAS_LLM_API_KEY`, `SEARCH_MAS_LLM_MODEL`.
  - Compatible aliases: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`.
- `search.retrieval_service_url`: retrieval service address (compatible with
  the DrMAS Search-R1 retrieval service).
  - Recommended env override: `SEARCH_MAS_RETRIEVAL_SERVICE_URL`.
- `application.max_turns`: maximum number of multi-turn steps.
- `agents.<agent>.llm.*`: per-agent backend overrides for
  `verifier`/`searcher`/`answerer`
  (`base_url`/`api_key`/`timeout`/`max_retries`/`retry_backoff_sec`/`model`).
  - Env prefixes: `SEARCH_MAS_VERIFIER_LLM_*`,
    `SEARCH_MAS_SEARCHER_LLM_*`, `SEARCH_MAS_ANSWERER_LLM_*`.
- `agents.*`: sampling parameters per agent
  (`temperature`/`top_p`/`max_tokens`/`stop`/`extra_body`).
- `validation.use_substring_em`: validation mode (EM or SubEM).

Example (separate model and retrieval services):

```yaml
llm:
  base_url: http://127.0.0.1:8000/v1

agents:
  verifier:
    llm:
      base_url: http://127.0.0.1:8001/v1
      model: /path/to/Qwen3-4B-Instruct-2507
  searcher:
    llm:
      base_url: http://127.0.0.1:8002/v1
  answerer:
    # Inherits the top-level llm config when llm is omitted.
    temperature: 0.4

search:
  provider: http
  retrieval_service_url: ${SEARCH_MAS_RETRIEVAL_SERVICE_URL}
```

## License

The code in this repository is licensed under the Apache License 2.0; see
[LICENSE](LICENSE).

Portions of `scripts/retrieval_server.py` derive from
[Search-R1](https://github.com/PeterGriffinJin/Search-R1), also licensed under
Apache-2.0.

The bundled datasets derive from Search-R1's
[`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)
(NQ + HotpotQA). They are not covered by the Apache-2.0 license; refer to the
upstream dataset for its terms.
