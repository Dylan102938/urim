# urim

**CLI utilities for LLM research: quick dataset creation, evaluations, chat, and inference.**

urim is a Python library designed to streamline LLM research workflows by providing powerful abstractions for dataset manipulation, model evaluation, and fine-tuning orchestration. Built with async-first design and intelligent caching, it enables researchers to rapidly iterate on experiments while maintaining reproducibility.

## 🎯 Motivation & Use Cases

### Why urim?

Modern LLM research involves repetitive patterns: preparing datasets, running evaluations, comparing models, and managing fine-tuning jobs. urim abstracts these workflows into composable, high-level operations that:

- **Reduce boilerplate**: Transform datasets with natural language hints instead of writing pandas code
- **Enable rapid experimentation**: Built-in caching prevents redundant API calls
- **Scale gracefully**: Async-first design with automatic rate limiting and retries
- **Maintain reproducibility**: Deterministic hashing and persistent storage of all operations

### Ideal Use Cases

- **Dataset Engineering**: Transform and augment datasets using LLMs with natural language descriptions
- **Model Evaluation**: Run systematic evaluations across models with different question types
- **Fine-tuning Workflows**: Orchestrate and track fine-tuning jobs with automatic retry and persistence
- **Research Iteration**: Quickly test hypotheses with cached results and parallel processing

## 🚀 Installation

```bash
pip install urim
```

Or with development dependencies:

```bash
pip install urim[dev]
```

For fine-tuning support with Tinker:

```bash
pip install urim[tinker]
```

## 📚 Quick Start

### Basic Dataset Operations

```python
import asyncio
from urim import Dataset, LLM

# Load a dataset
ds = Dataset("path/to/data.jsonl")  # or from HuggingFace: Dataset("squad")

# Sample and filter
ds_sample = ds.sample(n=100)
ds_filtered = await ds.filter(
    hint="Keep only examples where the question is about science",
    model="gpt-4o-mini"
)

# Transform with natural language
ds_transformed = await ds.apply(
    hint="Extract the main topic from each question and add it as a 'topic' column",
    model="gpt-4o-mini"
)

# Save results
ds_transformed.to_json("output.jsonl")
```

### Generating Answers with Different Question Types

```python
from urim import Dataset
from urim.ai.question import FreeForm, Rating, ExtractJSON

# Create a dataset with questions
df = pd.DataFrame({
    "question": [
        "What is the capital of France?",
        "Explain quantum computing",
        "What are the benefits of exercise?"
    ]
})
ds = Dataset(df)

# Generate free-form answers
ds_answered = await ds.generate(
    model="gpt-4o",
    enable_cot=True  # Enable chain-of-thought reasoning
)

# Generate with different question type
ds_evaluated = await ds_answered.generate(
    question_type=Rating,
    model="gpt-4o-mini",
    min_rating=1,
    max_rating=5
)
```

### Direct LLM Interaction

```python
from urim import LLM
from urim.ai.question import FreeForm, ExtractJSON, ExtractFunction

# Create a question
question = FreeForm(
    prompt="What is the best way to make apple pie?"
)
answer, extra = await question.resolve("gpt-4o")

# Extract structured outputs
question = ExtractJSON(
    prompt="List 3 interesting facts about Paris as a JSON array",
    enable_cot=True
)
facts = await question.json("gpt-4o-mini")

# Extract executable Python functions
fn_question = ExtractFunction(
    prompt="Write a function that calculates the fibonacci sequence up to n terms"
)
fib_function = await fn_question.fn("gpt-4o-mini")
result = fib_function(10)  # Use the generated function
```

### Fine-tuning Models

```python
from urim import model, Dataset

# Prepare training data
train_ds = Dataset("training_data.jsonl")

# Fine-tune a model (automatically manages the job)
finetuned = await model(
    "gpt-4o-mini",
    train_ds=train_ds,
    n_epochs=3,
    learning_rate=1e-5,
    batch_size=4
)

# Use the fine-tuned model
result = await LLM().chat_completion(
    model=finetuned.slug,
    messages=[{"role": "user", "content": "Test prompt"}]
)
```

## 🔧 Core Components

### Dataset

The `Dataset` class provides high-level operations for data manipulation with LLM assistance:

- **`sample(n, frac)`**: Random sampling of rows
- **`filter(fn, hint)`**: Filter rows using a function or natural language description
- **`apply(fn, column, hint)`**: Add new columns with computed values
- **`drop(columns, hint)`**: Remove columns by name or description
- **`rename(columns, hint)`**: Rename columns with mapping or description
- **`reduce(by, agg, hint)`**: Group and aggregate data
- **`generate()`**: Generate responses using LLMs for each row

All operations support both explicit parameters and natural language hints that are converted to code using LLMs.

### Question Types

urim provides specialized question types for different LLM interactions:

- **`FreeForm`**: Standard text generation
- **`ExtractJSON`**: Structured JSON extraction
- **`ExtractFunction`**: Generate executable Python functions
- **`Rating`**: Extract numerical ratings with confidence scores
- **`NextToken`**: Get next token predictions with probabilities

All question types support:
- Chain-of-thought reasoning (`enable_cot=True`)
- Response caching (automatic deduplication)
- Custom system prompts
- Salt values for cache invalidation

### LLM Client

The `LLM` class handles all model interactions:

- Automatic provider detection (OpenAI, OpenRouter, custom endpoints)
- Built-in retry logic with exponential backoff
- Support for multiple API keys with automatic fallback
- Logprobs and token probability extraction

### Fine-tuning Controller

The fine-tuning system provides:

- Automatic job submission and monitoring
- Persistent job tracking across restarts
- Concurrent job management with rate limiting
- Support for OpenAI and Tinker (local) fine-tuning

## 🔍 Advanced Features

### Caching System

urim implements multi-level caching:

```python
# Questions are automatically cached by content hash
q1 = FreeForm(prompt="What is 2+2?")
r1 = await q1.resolve("gpt-4o-mini")  # API call
r2 = await q1.resolve("gpt-4o-mini")  # Cache hit

# Invalidate cache with salt
q2 = FreeForm(prompt="What is 2+2?", salt="v2")
r3 = await q2.resolve("gpt-4o-mini")  # New API call

# Disable caching for specific questions
q3 = FreeForm(prompt="Random number", enable_cache=False)
```

### Parallel Processing

Dataset operations automatically parallelize LLM calls:

```python
# Processes multiple rows concurrently (controlled by semaphore)
ds_large = Dataset("large_dataset.jsonl")
ds_processed = await ds_large.generate(
    question_col="input",
    model="gpt-4o-mini"
)  # Automatically batches and parallelizes
```

### Judge Evaluations

Evaluate generated content with judge models:

```python
ds_with_judges = await ds.generate(
    question_col="question",
    out_col="answer",
    model="gpt-4o",
    judges={
        "quality": "Rate the quality of this answer: {answer}",
        "relevance": "How relevant is this answer to the question? {question} {answer}"
    }
)
# Adds columns: answer, quality, quality_raw, relevance, relevance_raw
```

### Environment Configuration

Configure urim through environment variables:

```bash
# API Keys
export OPENAI_API_KEY=sk-...
export OPENAI_API_KEY_2=sk-...  # Multiple keys for load balancing
export OPENROUTER_API_KEY=sk-or-...

# Storage
export urim_STORAGE_ROOT=/path/to/cache  # Default: ~/.cache/urim

# Logging
export urim_LOG_LEVEL=DEBUG
```

## 🧪 Testing

Run tests with:

```bash
# Run all tests
uv run pytest -s

# Skip tests requiring LLM calls
uv run pytest -s -m "not requires_llm"

# Run with coverage
uv run pytest -s --cov=urim
```

## 📖 API Reference

### Core Classes

#### `Dataset`
- **Constructor**: `Dataset(data: pd.DataFrame | str | Path, **kwargs)`
- **Methods**: `sample()`, `filter()`, `apply()`, `drop()`, `rename()`, `reduce()`, `generate()`, `to_json()`
- **Class Methods**: `concatenate(*datasets)`

#### `LLM`
- **Constructor**: `LLM(base_url=None, api_key=None, timeout=60.0)`
- **Methods**: `chat_completion(model, messages, **kwargs)`

#### `Question` (Abstract Base)
- **Subclasses**: `FreeForm`, `ExtractJSON`, `ExtractFunction`, `Rating`, `NextToken`
- **Methods**: `resolve(model)`, `hash()`, `remove_from_cache(model)`

#### `ModelRef`
- **Attributes**: `slug: str`, `checkpoints: list[str]`
- **Factory**: `await model(name, train_ds=None, **kwargs)`

### Utilities

#### Storage
- `urim.env.storage_subdir(...)`: Get storage subdirectory path
- `urim.env.set_storage_root(path)`: Set custom storage location

#### Logging
- `urim.logging.configure_logger(...)`: Configure logging
- `urim.logging.get_logger(name)`: Get named logger

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `uv run pytest`
2. Code is formatted: `uv run ruff format .`
3. Linting passes: `uv run ruff check .`

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Coming Soon]
- **Issues**: [GitHub Issues](https://github.com/yourusername/urim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/urim/discussions)

## 🎉 Acknowledgments

urim is built on top of excellent libraries:
- [OpenAI Python SDK](https://github.com/openai/openai-python) for LLM interactions
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Datasets](https://huggingface.co/docs/datasets) for HuggingFace integration
- [Tinker](https://github.com/yourusername/tinker) for local fine-tuning support

---

*urim: Making LLM research workflows simple, fast, and reproducible.*
