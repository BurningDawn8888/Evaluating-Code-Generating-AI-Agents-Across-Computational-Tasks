# Evaluating-Code-Generating-AI-Agents-Across-Computational-Tasks

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Methodology](#methodology)
4. [Task 1: Rolling Can Detection](#task-1-rolling-can-detection)
5. [Task 2: Road Line Detection](#task-2-road-line-detection)
6. [Task 3: Facial Recognition](#task-3-facial-recognition)
7. [Cross-Task Analysis](#cross-task-analysis)
8. [Pain Points & Limitations](#pain-points--limitations)
9. [Future Directions](#future-directions)
10. [Conclusion](#conclusion)

---

## Executive Summary

This research evaluates the performance of three state-of-the-art code-generating AI agents—Claude (Sonnet 4), Gemini (2.5 Pro), and OpenAI Codex (GPT-5)—on college-level computer vision assignments. Our benchmark tests whether these agents can autonomously complete real-world computational tasks that require both coding proficiency and domain-specific knowledge.

**Key Findings:**
- [Summary of which model performed best overall]
- [Self-reflection capabilities and their impact on results]
- [Token efficiency and computational cost considerations]
- [Qualitative differences in problem-solving approaches]

---

## Introduction

As AI coding assistants become increasingly sophisticated, a critical question emerges: **Can these agents handle the complexity of college-level computer vision assignments?** 

We designed a benchmark consisting of three progressively challenging tasks that require:
- Video/image processing capabilities
- Implementation of computer vision algorithms
- Debugging and iterative refinement
- Output validation and quality control

This blog post presents our comparative analysis of Claude, Gemini, and Codex across these tasks.

---

## Methodology

### Agent Setup
Each agent was provided with:
- **Sandbox environment** with read/write access to a designated directory
- **Task description** (`task.md`) with requirements and constraints
- **Input media** (videos/images) via full file paths
- **Grading API** for automated evaluation

### Evaluation Metrics
```python
# Placeholder for grading methodology code
# grade.py evaluates:
# - Detection accuracy (IoU, centroid distance)
# - Circle/line detection precision
# - Frame-by-frame consistency
# - Output format compliance
```

### Models Tested
- **Claude Sonnet 4.1** (agent v1)
- **Gemini 2.5 Pro** (agent v1)
- **OpenAI Codex GPT-5** (medium/high reasoning)

---

## Task 1: Rolling Can Detection

### Task Description
Detect and track a rolling can across video frames, outputting centroid coordinates (x, y) and radius (r) for each frame in CSV format.

**Requirements:**
- Accurate circle detection using computer vision techniques
- Frame-by-frame tracking consistency
- CSV output: `Frame_Index, X, Y, Radius`

### Results Comparison

| Model | Accuracy Score | Avg Centroid Error (px) | Radius Error (px) | Frames Detected |
|-------|----------------|-------------------------|-------------------|-----------------|
| Claude Sonnet 4.1 | [TBD] | [TBD] | [TBD] | [TBD] |
| Gemini 2.5 Pro | [TBD] | [TBD] | [TBD] | [TBD] |
| OpenAI Codex | [TBD] | [TBD] | [TBD] | [TBD] |

### Qualitative Findings

**Claude's Approach:**
```python
# Placeholder for Claude's solution snippet
# Notable: Over-thought the initial approach, implemented multiple
# detection methods and compared results
```

**Gemini's Approach:**
```python
# Placeholder for Gemini's solution snippet
# Notable: Initially claimed inability to read video files despite
# having the capability. Required explicit prompting about video input.
```

**Codex's Approach:**
```python
# Placeholder for Codex's solution snippet
# Notable: Ran inefficiently with excessive iterations
```

**Self-Reflection Analysis:**
- Did agents visually inspect their outputs?
- Did they iterate on failed attempts?
- How many attempts before achieving acceptable results?

---

## Task 2: Road Line Detection

### Task Description
Detect road lane markings in video footage and track them frame-by-frame.

**Requirements:**
- Line detection using edge detection/Hough transforms
- JSON output with line coordinates
- Robust performance across varying lighting conditions

### Results Comparison

| Model | Detection Rate | False Positives | Processing Time | Token Usage |
|-------|----------------|-----------------|-----------------|-------------|
| Claude Sonnet 4.1 | [TBD] | [TBD] | [TBD] | [TBD] |
| Gemini 2.5 Pro | [TBD] | [TBD] | [TBD] | [TBD] |
| OpenAI Codex | [TBD] | [TBD] | [TBD] | [TBD] |

### Qualitative Findings

```python
# Placeholder for code comparison showing different approaches
# to line detection (Canny edges, probabilistic Hough, etc.)
```

**Key Observations:**
- [Discussion of which model handled edge cases better]
- [Analysis of parameter tuning approaches]
- [Comparison of output visualization quality]

---

## Task 3: Facial Recognition

### Task Description
[Task description placeholder]

### Results Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Claude Sonnet 4.1 | [TBD] | [TBD] | [TBD] | [TBD] |
| Gemini 2.5 Pro | [TBD] | [TBD] | [TBD] | [TBD] |
| OpenAI Codex | [TBD] | [TBD] | [TBD] | [TBD] |

### Qualitative Findings

**Notable Issue - Claude's Over-Thinking:**
Claude demonstrated excessive self-doubt during facial recognition, second-guessing valid detections and implementing redundant verification steps.

```python
# Placeholder for Claude's overcomplicated verification logic
```

---

## Cross-Task Analysis

### Overall Performance Rankings
1. **[Winner]** - [Brief justification]
2. **[Second place]** - [Brief justification]
3. **[Third place]** - [Brief justification]

### Self-Reflection Effectiveness
Did adding system prompts encouraging self-review improve results?

**Comparison:**
- **Standard prompt**: [Results]
- **With self-review prompt**: [Results]
- **With multi-attempt prompt**: [Results]

```python
# Placeholder for visualization code comparing prompt variations
```

### Token Efficiency

| Model | Avg Tokens/Task | Cost Estimate | Time to Completion |
|-------|-----------------|---------------|-------------------|
| Claude | [TBD] | [TBD] | [TBD] |
| Gemini | [TBD] | [TBD] | [TBD] |
| Codex | [TBD] | [TBD] | [TBD] |

---

## Pain Points & Limitations

### Model-Specific Issues

**Gemini 2.5 Pro:**
- **Hallucinated limitations**: Believed it couldn't read video files despite having the capability
- **CLI constraints**: Free tier limitations on input/output size
- **Prompt adherence**: Occasionally ignored hints in task descriptions

**OpenAI Codex:**
- **Efficiency**: Ran indefinitely on certain tasks, requiring manual intervention
- **Resource usage**: High token consumption without proportional quality improvement

**Claude Sonnet 4.1:**
- **Over-analysis**: Excessive self-doubt and redundant verification steps
- **Double-guessing**: Second-guessed correct solutions (especially in facial recognition)

### Benchmark Limitations

1. **Task Massaging**: Tasks required some refinement to get models working—they struggled with completely open-ended problems
2. **Grading Coverage**: Road line detection only uses randomly sampled frames. Models could fail on unsampled segments without penalty.
3. **Statistical Significance**: Three tasks provide limited sample size. More problems needed for robust conclusions.
4. **Self-Reflection Challenges**: Our attempts to prompt effective self-review didn't work well. Models need better metacognitive capabilities.

---

## Future Directions

### Expanding the Benchmark
- **Increase task diversity**: Add more computer vision problems (object segmentation, pose estimation, etc.)
- **Reduce noise**: Create multiple similar problems to establish statistical significance
- **Test additional agents**: Evaluate Devin, Cursor, and open-source models

### Improving Evaluation
- **LLM-based grading**: Use vision-capable models to inspect video outputs and grade more holistically
- **Comprehensive frame coverage**: Grade all frames or use smarter sampling strategies
- **Multi-modal assessment**: Combine quantitative metrics with qualitative code review

### Enhanced Self-Reflection
Inspired by the [IMO challenge solution](https://arxiv.org/html/2507.15855v1), implement **self-hinting**:
- Generate multiple generic hints automatically
- Prompt models with different hint combinations
- Select best solution across all hint-guided attempts

```python
# Placeholder for self-hinting implementation concept
hints = [
    "Consider edge cases in lighting conditions",
    "Verify detection consistency across frames",
    "Visualize intermediate processing steps"
]
# Run agent with each hint, compare results
```

---

## Conclusion

Our benchmark reveals that while AI coding agents show impressive capabilities, they still struggle with the nuanced requirements of real-world computer vision tasks. Each model demonstrated distinct strengths and weaknesses:

- **Claude** excels at thorough analysis but can over-complicate solutions
- **Gemini** has strong capabilities but needs explicit guidance on its own features
- **Codex** generates functional code but lacks efficiency optimization

**Key Takeaway**: Current agents benefit significantly from well-structured prompts and task descriptions. The gap between "write code" and "solve a real problem" remains substantial, particularly for tasks requiring iterative refinement and quality validation.

As these tools evolve, the ability to self-reflect, debug effectively, and optimize solutions will be critical for handling college-level computational assignments autonomously.

---

## Code Repository

Full code, task descriptions, and results available at: `[GitHub repository link]`

**Repository Structure:**
```
assignment-bench/
├── tasks/
│   ├── rolling_can/
│   │   ├── task.md
│   │   ├── rollcan.mp4
│   │   └── human_circle_coordinates.csv
│   ├── road_lines/
│   └── facial_recognition/
├── solutions/
│   ├── claude_sonnet4.1_agentv1/
│   ├── gemini2.5_agentv1/
│   └── codex_gpt5/
├── grade.py
└── README.md
```

---

**References:**
- [Reka VIBE Eval](https://github.com/reka-ai/reka-vibe-eval)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [GDP Validation Paper](https://arxiv.org/abs/2510.04374)
