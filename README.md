# Track 1 General-Purpose AI Agent

A Docker-ready agent for the AMD Developer Hackathon Track 1. It handles
factual knowledge, mathematical reasoning, sentiment, summarisation, named
entity recognition, debugging, logic, and code generation.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
```

Put development-only Fireworks values in `.env`. Never include `.env` or a
credential in a submitted image; Docker excludes it through `.dockerignore`.

## Run locally

```bash
python main.py --batch
python -m unittest discover -s tests -v
```

Batch mode reads `input/tasks.json` and writes `output/results.json`. Use
`--input` and `--output` to override the paths. When `/input/tasks.json` and
`/output` exist, those container paths are selected automatically.

For submission, the evaluation harness injects `FIREWORKS_API_KEY`,
`FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`. The agent selects only a model in
`ALLOWED_MODELS` and sends every Fireworks request through the supplied base
URL. Exact arithmetic expressions are solved locally; all other task types use
the permitted Fireworks model. If a task cannot be answered in batch mode, the
process exits non-zero instead of writing an API error as an answer.

## Self-contained local model (optional)

The submitted image can run a local model without Ollama or any network service.
Download a **2B-3B, 4-bit GGUF instruct model** and save it as
`models/model.gguf` before building. A 2B-3B Q4 model is the guide's safe range
for the 4 GB / 2 vCPU evaluator; do not use a 7B model. The image loads this
file lazily with `llama.cpp`, uses two CPU threads by default, and falls back to
the harness-provided Fireworks model when its three required environment
variables are present.

Keep the model filename exactly `models/model.gguf` (or set
`LOCAL_MODEL_PATH`), then test the full offline path:

```bash
docker build --platform linux/amd64 -t routing-agent:local .
docker run --rm -v "${PWD}/input:/input:ro" -v "${PWD}/output:/output" routing-agent:local
```

This writes `/output/results.json` and exits. The local GGUF file is included
in the image, so the final compressed image must remain under 10 GB. Do not
depend on `host.docker.internal` or `localhost:11434` in a submission: those
refer to a service outside the judging container. Ollama is retained only as a
development fallback when no GGUF is bundled.

## Container submission

```bash
docker build --platform linux/amd64 -t routing-agent .
docker run --rm -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS -v "${PWD}/input:/input:ro" -v "${PWD}/output:/output" routing-agent
```

Build and publish a public `linux/amd64` image before submission. The judge
also requires a compressed image under 10 GB, a maximum 10-minute run, and a
valid `/output/results.json` before exit.

For Apple Silicon, publish an amd64 manifest with:

```bash
docker buildx build --platform linux/amd64 --tag <registry>/<user>/routing-agent:latest --push .
```
