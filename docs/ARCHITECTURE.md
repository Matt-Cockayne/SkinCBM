# CBM Architecture Guide

A comprehensive guide to understanding Concept Bottleneck Models.

## Table of Contents

1. [What is a CBM?](#what-is-a-cbm)
2. [Architecture Components](#architecture-components)
3. [Information Flow](#information-flow)
4. [Training Strategies](#training-strategies)
5. [Design Decisions](#design-decisions)
6. [Advantages and Limitations](#advantages-and-limitations)

## What is a CBM?

A **Concept Bottleneck Model** is a neural network designed for interpretable predictions by forcing all information to flow through human-understandable concepts.

### Traditional Black-Box Model
```
Image → [Neural Network] → Prediction
         (uninterpretable)
```

### Concept Bottleneck Model
```
Image → [Concept Encoder] → Concepts → [Task Predictor] → Prediction
         (learns concepts)    (↑)        (uses concepts)
                          interpretable
                          + intervention
```

## Architecture Components

### 1. Concept Encoder: Image → Concepts

**Purpose**: Extract human-interpretable concepts from raw images

**Architecture**:
```python
class ConceptEncoder(nn.Module):
    def __init__(self):
        self.backbone = ResNet50(pretrained=True)  # Feature extraction
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
            for _ in range(num_concepts)
        ])
```

**Key Design Choices**:
- **Pretrained backbone**: Transfer learning from ImageNet
- **Separate heads**: Each concept has its own prediction head
- **Sigmoid activation**: Binary concepts in range [0, 1]
- **Dropout**: Prevents overfitting on small medical datasets

**Example Concepts** (Derm7pt):
1. Atypical pigment network
2. Blue-whitish veil  
3. Atypical vascular pattern
4. Irregular streaks
5. Irregular pigmentation
6. Irregular dots and globules
7. Regression structures

### 2. Task Predictor: Concepts → Diagnosis

**Purpose**: Make final prediction using only concept values

**Three Architectures Available**:

#### A. Linear Predictor (Most Interpretable)

```python
class LinearTaskPredictor(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(num_concepts, num_classes)
    
    def forward(self, concepts):
        return self.fc(concepts)
```

**Interpretation**: 
- Each weight shows concept importance: `y = w₁·c₁ + w₂·c₂ + ... + wₙ·cₙ + b`
- Easy to visualize: "Blue-whitish veil increases malignancy score by 0.73"

**Best for**: 
- Maximum interpretability
- Understanding concept-task relationships
- All use cases in this educational implementation

## Information Flow

### Forward Pass (Inference)

```python
# Input: Image [batch, 3, 224, 224]
image = load_image("lesion.jpg")

# Step 1: Extract features
features = backbone(image)  # [batch, 2048]

# Step 2: Predict concepts
concepts = []
for head in concept_heads:
    c = head(features)  # [batch, 1]
    concepts.append(c)
concepts = torch.cat(concepts, dim=1)  # [batch, num_concepts]

# Step 3: Predict task
logits = task_predictor(concepts)  # [batch, num_classes]
prediction = logits.argmax(dim=1)
```

### Concept Intervention

The key advantage of CBMs: modify concepts at test time!

```python
# Original prediction
concepts, logits = model(image)
# concepts: [0.92, 0.15, 0.78, 0.88, ...]  ← concept 1 is wrong!
# prediction: Benign (incorrect)

# Intervene: Fix concept 1
concepts[1] = 1.0  # Correct it to present

# Re-predict with corrected concepts
new_logits = task_predictor(concepts)
# new_prediction: Malignant (correct!)
```

**Why This Works**:
- Information bottleneck forces all reasoning through concepts
- Task predictor never sees raw images
- Changing concepts changes prediction

## Training Strategies

### 1. Joint Training (Default)

Train both components simultaneously:

```python
# Loss = concept_loss + task_loss
concept_loss = BCE(pred_concepts, true_concepts)
task_loss = CrossEntropy(task_logits, labels)
total_loss = λ * concept_loss + task_loss

total_loss.backward()
optimizer.step()
```

**Pros**:
- Simple, single training phase
- End-to-end optimization

**Cons**:
- Concepts may become "shortcuts" that help task but aren't meaningful
- Harder to achieve high concept accuracy

**Use when**: You want quick results and concepts are well-defined

### 2. Sequential Training

Train in two stages:

**Stage 1**: Train concepts only
```python
concept_loss = BCE(pred_concepts, true_concepts)
concept_loss.backward()
optimizer.step()
```

**Stage 2**: Freeze concepts, train task predictor
```python
model.freeze_concepts()
task_loss = CrossEntropy(task_logits, labels)
task_loss.backward()
optimizer.step()
```

**Pros**:
- Higher concept accuracy
- Concepts don't become task-specific shortcuts

**Cons**:
- Two training phases (more complex)
- May hurt task performance slightly

**Use when**: Concept accuracy is critical (e.g., for human validation)

### 3. Independent Training

Train concepts and task with ground-truth concepts:

```python
# Train concepts normally
concept_loss.backward()

# Train task predictor using GROUND TRUTH concepts
task_logits = task_predictor(true_concepts)  # Not predicted concepts!
task_loss.backward()
```

**Pros**:
- Maximum concept accuracy
- Task predictor learns optimal concept usage

**Cons**:
- Requires concept annotations at training time
- Performance gap at test time (using predicted concepts)

**Use when**: You have high-quality concept labels and want to study concept importance

## Design Decisions

### Why Separate Concept Heads?

**Alternative**: Single head predicting all concepts
```python
# Could do this:
self.concepts = nn.Linear(2048, num_concepts)

# Instead of:
self.concept_heads = nn.ModuleList([
    nn.Linear(2048, 1) for _ in range(num_concepts)
])
```

**Reasoning**:
- Concepts are semantically different (irregular border ≠ blue color)
- Separate heads allow per-concept specialization
- Easier to add/remove concepts without retraining

### Why Sigmoid for Concepts?

**Alternatives**: Softmax (mutually exclusive), raw logits

**Reasoning**:
- Medical concepts are not mutually exclusive (can have multiple present)
- Sigmoid gives probabilistic interpretation: P(concept is present)
- Range [0, 1] is intuitive for humans

### Why Linear Task Predictor?

**Reasoning**:
- Interpretability is the core value proposition
- Linear weights directly show concept importance
- Simplicity aids understanding of CBM fundamentals
- Easy to extend to non-linear predictors if needed

## Advantages and Limitations

### Advantages ✓

1. **Interpretability**
   - See which concepts drove each prediction
   - Understand model reasoning

2. **Intervention**
   - Correct wrong concept predictions
   - Incorporate human expertise at test time

3. **Debugging**
   - Identify concept prediction errors
   - Find dataset biases (e.g., relying on image artifacts)

4. **Trust**
   - Medical professionals can validate reasoning
   - Satisfies regulatory requirements for explainability

5. **Concept Discovery**
   - Identify which concepts are most important
   - Guide future dataset collection

### Limitations ✗

1. **Performance Gap**
   - Typically 3-7% lower accuracy than black-box models
   - Bottleneck restricts information flow

2. **Concept Annotations Required**
   - Need expert-labeled concepts for training
   - Expensive and time-consuming to collect

3. **Concept Completeness**
   - Must define all relevant concepts upfront
   - Missing concepts → incomplete reasoning

4. **Concept Quality**
   - Model performance bounded by concept prediction accuracy
   - Noisy concept labels hurt both concept and task performance

5. **Architecture Constraints**
   - Forced bottleneck may be suboptimal for some tasks
   - Less flexible than end-to-end training

## When to Use CBMs

### Good Use Cases ✓
- Medical diagnosis (high stakes, need interpretability)
- Regulated domains (must explain decisions)
- Human-in-the-loop systems (leverage intervention)
- Scientific discovery (understand what matters)

### Poor Use Cases ✗
- Performance-critical applications (object detection, etc.)
- No concept annotations available
- Concepts hard to define (abstract tasks)
- Black-box performance is acceptable

## Further Reading

- [Intervention Tutorial](INTERVENTION.md) - How to use concept intervention
- [Information Theory Guide](INFORMATION_THEORY.md) - Quantifying concept quality
- [Training Strategies](TRAINING.md) - Advanced training techniques

---

**Summary**: CBMs trade ~5% accuracy for full interpretability and intervention capabilities. They're ideal for high-stakes medical applications where understanding *why* a model made a prediction is as important as the prediction itself.
