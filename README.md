# Output Length Prediction in Language Models

A probing experiment to determine if language models have internal representations of their output length during generation.

## Hypothesis

Language models may develop awareness of output length as generation progresses, though likely not before generation begins. We expect hidden states to reflect an increasing representation of final length, particularly for structured or predictable tasks.

## Approach

### 1. Data Collection

Generate model outputs on three task categories:
- **Predictable tasks**: Simple repetition prompts (e.g., "print ten times the word hello")
- **Structured text**: Constrained generation like poems or programming algorithms
- **Generic questions**: Open-ended queries

### 2. Probe Design

- **Input**: Hidden states from model layers (starting with last token of last layer)
- **Output**: Expected number of remaining tokens at each generation step
- **Architecture**: Linear probe or simple MLP
- **Approach**: Start with classification (length bins), then regression

### 3. Evaluation

- **Regression**: MAE, RMSE, R²
- **Classification**: Accuracy, F1 score

## Key Questions

- Do models develop length awareness during generation?
- Does awareness increase monotonically with generation progress?
- Does task complexity affect length prediction accuracy?