# General Agent

A small Python project scaffold for routing tasks to local solvers or a Fireworks AI model.

## Setup

```bash
pip install -r requirements.txt
```

Optional Fireworks configuration:

```bash
copy .env.example .env
```

Then put your Fireworks key in `.env`.

## Run

```bash
python main.py
```

When `input/tasks.json` exists, the agent reads all tasks, writes `output/results.json`, and exits. Use interactive mode for manual testing:

```bash
python main.py --interactive
```

In interactive mode, the program stays open for more queries. Press Enter to submit a query and type
`/exit` to close it. On Windows, a multiline clipboard paste is recovered as a
single query when its first line reaches Enter, and the remaining pasted input
is cleared so it cannot create extra queries.

Each result includes the answering backend and an estimated answer accuracy.
For Fireworks AI responses, the CLI also displays the total token count reported
by the Fireworks API (or reports that usage was unavailable).

## JSON batch mode

You can also force the container-style input/output flow locally without Docker:

```bash
python main.py --batch
```

It reads [`input/tasks.json`](input/tasks.json) and writes
[`output/results.json`](output/results.json). Use `--input` and `--output` to
override the paths. In a future container, paths `/input/tasks.json` and
`/output/results.json` are automatically selected when those directories exist.

For a submission environment, provide `FIREWORKS_API_KEY`,
`FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` at runtime. The agent only selects
models listed in `ALLOWED_MODELS` and routes Fireworks requests through the
provided base URL. Once those values are supplied, Fireworks handles every
non-deterministic language task for accuracy; only exact arithmetic expressions
remain local because they are faster and deterministic.

