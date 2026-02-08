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

This research evaluates the performance of three state-of-the-art code-generating AI agents—Claude (Sonnet 4.1), Gemini (2.5 Pro), and OpenAI Codex (GPT-5)—on college-level computer vision assignments. Our benchmark tests whether these agents can autonomously complete real-world computational tasks that require both coding proficiency and domain-specific knowledge.

**Key Findings:**
- **Best overall model:** **Claude Sonnet 4.1**, driven by consistently strong accuracy on Rolling Can Detection and the best stability/accuracy balance on Road Line Detection.
- **Self-reflection effectiveness:** **Limited impact overall.** Agents rarely did reliable “visual self-checking” of outputs (videos/images) without being explicitly forced into a review loop. When they *did* iterate, improvements were mostly due to parameter tuning rather than deep self-critique.
- **Token efficiency and computational cost:** **Not fully measured in this benchmark** (no standardized token/time logs across agents). Qualitatively, Codex tended to “run long” and produce more iterations/code churn; Claude tended to over-build; Gemini often required extra prompting before converging.
- **Qualitative differences:** Claude favored thoroughness (sometimes over-complicated), Gemini was capable but needed explicit guidance about inputs/capabilities, and Codex was fast to produce pipelines but less reliable in validation and stability.

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
- **Grading API / scripts** for automated evaluation (when available)

### Evaluation Metrics (Example)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def load_and_clean_data(human_file, solution_file):
    """
    Load and clean the CSV data from both files
    """
    # Load the data
    human_df = pd.read_csv(human_file)
    solution_df = pd.read_csv(solution_file)
    
    # Clean the data - remove rows with empty values
    human_df = human_df.dropna()
    solution_df = solution_df.dropna()
    
    # Convert to numeric
    for col in ['X', 'Y', 'Radius']:
        human_df[col] = pd.to_numeric(human_df[col], errors='coerce')
        solution_df[col] = pd.to_numeric(solution_df[col], errors='coerce')
    
    # Remove any remaining NaN values
    human_df = human_df.dropna()
    solution_df = solution_df.dropna()
    
    return human_df, solution_df

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def evaluate_solution(human_df, solution_df):
    """
    Evaluate how close the solution is to the human results
    """
    results = {}
    
    # Find common frames between human and solution data
    common_frames = set(human_df['Frame_Index']) & set(solution_df['Frame_Index'])
    common_frames = sorted(list(common_frames))
    
    if not common_frames:
        print("No common frames found between human and solution data!")
        return results
    
    print(f"Found {len(common_frames)} common frames for comparison")
    
    # Initialize lists to store metrics
    distances = []
    radius_diffs = []
    x_diffs = []
    y_diffs = []
    
    matched_frames = 0
    
    for frame in common_frames:
        human_frame = human_df[human_df['Frame_Index'] == frame]
        solution_frame = solution_df[solution_df['Frame_Index'] == frame]
        
        # For each human detection, find the closest solution detection
        for _, human_row in human_frame.iterrows():
            min_distance = float('inf')
            best_solution = None
            
            for _, solution_row in solution_frame.iterrows():
                # Calculate distance between circle centers
                dist = calculate_distance(
                    human_row['X'], human_row['Y'],
                    solution_row['X'], solution_row['Y']
                )
                
                if dist < min_distance:
                    min_distance = dist
                    best_solution = solution_row
            
            if best_solution is not None:
                matched_frames += 1
                distances.append(min_distance)
                radius_diffs.append(abs(human_row['Radius'] - best_solution['Radius']))
                x_diffs.append(abs(human_row['X'] - best_solution['X']))
                y_diffs.append(abs(human_row['Y'] - best_solution['Y']))
    
    # Calculate statistics
    if distances:
        results['total_frames_analyzed'] = len(common_frames)
        results['matched_detections'] = matched_frames
        results['match_rate'] = matched_frames / len(common_frames) * 100
        
        # Distance metrics
        results['mean_center_distance'] = np.mean(distances)
        results['median_center_distance'] = np.median(distances)
        results['std_center_distance'] = np.std(distances)
        results['max_center_distance'] = np.max(distances)
        results['min_center_distance'] = np.min(distances)
        
        # Radius difference metrics
        results['mean_radius_diff'] = np.mean(radius_diffs)
        results['median_radius_diff'] = np.median(radius_diffs)
        results['std_radius_diff'] = np.std(radius_diffs)
        results['max_radius_diff'] = np.max(radius_diffs)
        
        # Coordinate difference metrics
        results['mean_x_diff'] = np.mean(x_diffs)
        results['mean_y_diff'] = np.mean(y_diffs)
        results['std_x_diff'] = np.std(x_diffs)
        results['std_y_diff'] = np.std(y_diffs)
        
        # Overall accuracy score (higher is better)
        distance_score = results['mean_center_distance'] / 100  # Normalize to 0-1
        radius_score = results['mean_radius_diff'] / 50         # Normalize to 0-1
        results['overall_accuracy_score'] = 1 - (distance_score + radius_score) / 2
        
    return results
````

### Models Tested

* **Claude Sonnet 4.1** (agent v1)
* **Gemini 2.5 Pro** (agent v1)
* **OpenAI Codex GPT-5** (agent v1; medium/high reasoning)

---

## Task 1: Rolling Can Detection

### Task Description

Detect and track a rolling can across video frames, outputting centroid coordinates (x, y) and radius (r) for each frame in CSV format.

**Requirements:**

* Accurate circle detection using computer vision techniques
* Frame-by-frame tracking consistency
* CSV output: `Frame_Index, X, Y, Radius`

### Results Comparison

| Model                | Accuracy Score | Avg Centroid Error (px) | Radius Error (px) | Frames Detected    |
| -------------------- | -------------- | ----------------------- | ----------------- | ------------------ |
| Claude Sonnet 4.1    | **0.9006**     | **11.19**               | **4.35**          | **153/153 (100%)** |
| Gemini 2.5 Pro       | **-0.7289**    | **339.10**              | **3.34**          | **153/153 (100%)** |
| OpenAI Codex (GPT-5) | **0.8939**     | **13.94**               | **3.64**          | **153/153 (100%)** |

> Note: Gemini’s final CSV matched frame indices but produced extremely large centroid error, which suggests the detector “locked onto” the wrong circle-like features or drifted heavily despite outputting a value every frame.

### Qualitative Findings

**Claude's Approach:**

```python
# Claude solution style (high-level summary):
# - Preprocessing to reduce noise
# - Circle detection (Hough / contour-based variants)
# - Tracking heuristics to keep detection stable frame-to-frame
#
# Notable: Over-thought the initial approach and explored multiple detection paths.
# Upside: Very strong accuracy and stable tracking.
# Downside: More complexity than required for this task.
```

**Gemini's Approach:**

```python
# Gemini solution style (high-level summary):
# - Multiple versions (v1..v5) of processing scripts
# - Iterative parameter changes (thresholds, blur, Hough params)
#
# Notable: Required extra iterations to converge on a workable approach.
# In the final output, it produced a value for every frame, but centroid accuracy was poor.
```

**Codex's Approach:**

```python
# Codex solution style (high-level summary):
# - Implemented detection + overlay quickly
# - Generated multiple files and ran longer iteration loops
#
# Notable: Ran less efficiently and produced more churn,
# but the final circle tracking accuracy was competitive with Claude.
```

**Self-Reflection Analysis:**

* Did agents visually inspect their outputs?

  * **Inconsistently.** Some solutions produced overlay videos/images, but agents did not reliably use them to correct errors unless prompted.
* Did they iterate on failed attempts?

  * **Yes**, especially Gemini (multiple numbered solutions).
* How many attempts before achieving acceptable results?

  * Claude and Codex reached “good enough” quickly; Gemini required more iteration and still did not achieve strong centroid accuracy.

---

## Task 2: Road Line Detection

### Task Description

Detect road lane markings in video footage and track them frame-by-frame.

**Requirements:**

* Line detection using edge detection/Hough transforms
* JSON output with line coordinates
* Robust performance across varying lighting conditions

### Results Comparison

| Model                | Detection Rate                    | False Positives       | Processing Time |
| -------------------- | --------------------------------- | --------------------- | --------------- |
| Claude Sonnet 4.1    | **80.00%**                        | N/A (sparse labeling) | Not measured    |
| Gemini 2.5 Pro       | **N/A** (file was to large for gemini to process) | N/A                   | Not measured    |
| OpenAI Codex (GPT-5) | **100.00%**                       | N/A (sparse labeling) | Not measured    |

**Extra Accuracy Metrics (from the comparison script):**

* **Claude:** Overall Accuracy Score **91.36%**
* **Codex:** Overall Accuracy Score **83.02%**

### Qualitative Findings

```python
# Typical approaches seen in solutions:
# - Canny edge detection
# - ROI masking (focus on lower part of frame)
# - Hough line transform / probabilistic Hough
# - Post-processing: slope filtering, left/right lane separation, smoothing
#
# Claude: more conservative detection (lower detection rate, higher accuracy)
# Codex: detects more often, but with larger endpoint/angle error and more instability
```

**Key Observations:**

* **Which model handled edge cases better?**
  **Claude** performed better on accuracy and angle consistency. Codex tended to “always output something,” which increased detection rate but reduced correctness.
* **Parameter tuning approaches:**
  Both relied heavily on tuning thresholds/edge settings; neither had a truly robust “auto-tuning” strategy for shadows and lighting shifts.
* **Output visualization quality:**
  Both produced output videos; however, **visual debugging was not consistently used** to drive improvements unless explicitly prompted.

---

## Task 3: Facial Recognition

### Task Description

Create a program capable of facial recognition in a full image given the target face. The system should detect and identify the target person's face within a larger crowd image.

**Requirements (from `task.md`):**

* Input: `target.png` and `crowd.jpg`
* Output CSV: `Image_Name,Target_Found,Bounding_Box_X,Bounding_Box_Y,Bounding_Box_Width,Bounding_Box_Height`
* Output image with bounding box
* **Only one identification box**, and it must be on the target face

### Results Comparison

Because this task is evaluated on a **single image** (and no labeled ground-truth bounding box was provided in the benchmark artifacts), precision/recall/F1 cannot be reported in a fully standard way. We report:

* whether the model claims `Target_Found=True`
* and a **bounding-box agreement proxy** (models that agree are more likely correct)

| Model                | Accuracy                                   | Precision          |
| -------------------- | ------------------------------------------ | ------------------ |
| Claude Sonnet 4.1    | Target_Found=True; bbox agrees with Gemini | N/A (single image) |
| Gemini 2.5 Pro       | Target_Found=True; bbox agrees with Claude | N/A (single image) |
| OpenAI Codex (GPT-5) | Target_Found=True; bbox disagrees strongly | N/A (single image) |

**Bounding box agreement (IoU proxy using Claude/Gemini consensus):**

* Claude IoU vs consensus: ~**0.77**
* Gemini IoU vs consensus: ~**0.73**
* Codex IoU vs consensus: **0.00** (box placed far from the consensus region)

### Qualitative Findings

**Notable Issue - Claude's Over-Thinking:**
Claude demonstrated more self-doubt during this task, sometimes adding verification steps instead of committing to a simpler pipeline.

```python
# Claude-style pattern (high-level):
# - Detect face regions
# - Compute embedding / compare similarity (or template match)
# - Extra checks / redundant validation logic
#
# Result: Good localization, but more complicated than needed for "one box" output.
```

---

## Cross-Task Analysis

### Overall Performance Rankings

1. **Claude Sonnet 4.1** - Best overall accuracy and the strongest reliability across tasks (best lane accuracy score; near-top on rollcan; strong face localization).
2. **OpenAI Codex (GPT-5)** - Strong on rolling can and high lane detection rate, but weaker on lane correctness and facial bounding box placement.
3. **Gemini 2.5 Pro** - Showed capability and iteration, but produced weak quantitative results on rolling can and did not have a comparable road-line run artifact in this repo snapshot.

### Self-Reflection Effectiveness

Did adding system prompts encouraging self-review improve results?

**Comparison:**

* **Standard prompt**: Improvements mainly came from parameter tuning; output self-checks were inconsistent.
* **With self-review prompt**: Slight improvements when agents actually inspected overlays/images, but not reliable.
* **With multi-attempt prompt**: Helped Gemini the most (multiple attempts), but did not guarantee correctness.

```python
# Placeholder for visualization code comparing prompt variations
# (Not included here because token/time logs were not standardized in this benchmark run.)
```

---

## Pain Points & Limitations

### Model-Specific Issues

**Gemini 2.5 Pro:**

* **Hallucinated limitations / capability confusion**: Needed explicit guidance about inputs and processing steps
* **Prompt adherence**: Occasionally missed constraints without repeated reminders
* **High iteration requirement**: Multiple solution versions before convergence

**OpenAI Codex:**

* **Efficiency**: Tends to generate more code churn and longer iterative runs
* **Validation gap**: Produced outputs, but often lacked rigorous confirmation that outputs were correct (especially facial bbox placement)

**Claude Sonnet 4.1:**

* **Over-analysis**: Redundant checks and over-engineered pipelines
* **Double-guessing**: More likely to second-guess correct outputs (notably in facial recognition)

### Benchmark Limitations

1. **Task Massaging**: Tasks required some refinement to get models working—they struggled with completely open-ended problems
2. **Grading Coverage**: Road line detection uses sparse manual annotations. Many frames are “0” simply because they were not labeled, which makes “false positives” hard to interpret.
3. **Statistical Significance**: Three tasks provide limited sample size. More problems needed for robust conclusions.
4. **Self-Reflection Challenges**: Our attempts to prompt effective self-review didn't work well. Models need better metacognitive capabilities.

---

## Future Directions

### Expanding the Benchmark

* **Increase task diversity**: Add more computer vision problems (object segmentation, pose estimation, etc.)
* **Reduce noise**: Create multiple similar problems to establish statistical significance
* **Test additional agents**: Evaluate Devin, Cursor, and open-source models

### Improving Evaluation

* **LLM-based grading**: Use vision-capable models to inspect video outputs and grade more holistically
* **Comprehensive frame coverage**: Grade all frames or use smarter sampling strategies
* **Multi-modal assessment**: Combine quantitative metrics with qualitative code review

### Enhanced Self-Reflection

Inspired by the [IMO challenge solution](https://arxiv.org/html/2507.15855v1), implement **self-hinting**:

* Generate multiple generic hints automatically
* Prompt models with different hint combinations
* Select best solution across all hint-guided attempts

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

* **Claude** excels at thorough analysis and accuracy but can over-complicate solutions
* **Gemini** has strong capabilities but often needs explicit guidance and multiple attempts
* **Codex** generates functional code quickly but lacks consistent output validation and efficiency

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

* [Reka VIBE Eval](https://github.com/reka-ai/reka-vibe-eval)
* [SWE-bench](https://github.com/SWE-bench/SWE-bench)
* [GDP Validation Paper](https://arxiv.org/abs/2510.04374)
