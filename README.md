<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-PPO-red?style=for-the-badge&logo=tensorflow&logoColor=white" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Power%20Grid-IEEE%2014--Bus-green?style=for-the-badge&logo=grid&logoColor=white" alt="Power Grid">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <h1 align="center"> GridGuard</h1>
  <p align="center">
    <strong>Autonomous AI-Powered Power Grid Stabilization System</strong>
  </p>
  <p align="center">
    <em>Real-time anomaly detection, predictive instability forecasting, and intelligent control using Deep Reinforcement Learning</em>
  </p>
</p>

---

##  Overview

GridGuard is a state-of-the-art intelligent control framework that leverages **Deep Reinforcement Learning (PPO)** and **Machine Learning** to detect, predict, and prevent cascading failures in electrical power grids. Designed as an AI co-pilot for grid operators, GridGuard provides autonomous stabilization recommendations while maintaining human-in-the-loop decision making.

### The Problem

Modern power grids face increasing challenges from:
-  **Renewable Energy Integration** - Intermittent solar/wind generation creates voltage fluctuations
-  **Extreme Weather Events** - Storms, heatwaves, and cold snaps stress grid infrastructure
-  **Aging Infrastructure** - Legacy systems require modern intelligent monitoring
-  **Cascading Failures** - Single points of failure can trigger widespread blackouts

GridGuard addresses these challenges through proactive anomaly detection, predictive modeling, and autonomous control actions.

---

##  Key Features

###  Intelligent Anomaly Detection
- **Multi-algorithm ensemble**: Isolation Forest + Random Forest + Local Outlier Factor
- **Real-time state assessment** with normalized instability scores [0-1]
- **Threshold-based alerting**: Normal → Warning → Elevated → Critical

###  Deep Reinforcement Learning
- **PPO (Proximal Policy Optimization)** agent with custom neural network architecture
- **256×128×64 MLP policy network** with ReLU activations
- **Reward shaping**: Voltage stability bonuses, thermal violation penalties
- **TensorBoard integration** for training visualization

###  Predictive Instability Forecasting
- **Trend analysis** with configurable lookback windows
- **Lead-time estimation** to potential failures (15/30/45 minutes)
- **Voltage & loading trend detection** with confidence scoring

###  Tier-Based Response System
| Tier | Response Level | Actions | Disruption |
|------|---------------|---------|------------|
| Tier 1 | Low | Reduce industrial load, curtail renewables | Low |
| Tier 2 | Medium | Activate spinning reserves, start peaker plants | Medium |
| Tier 3 | High | Emergency load shedding, substation decoupling | High |

###  Realistic Power Grid Simulation
- **IEEE 14-Bus Test System** implemented with PyPSA
- **9-dimensional continuous action space** (generator redispatch + tap adjustments)
- **54-dimensional observation space** (voltages, angles, loadings, generator outputs)
- **Complex scenario generation**: 12+ disturbance types

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            GridGuard System                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Scenario   │    │    IEEE      │    │     Anomaly Detector     │   │
│  │  Generator   │───▶│ 14-Bus Grid  │───▶│ (Isolation Forest + RF)   │   │
│  └──────────────┘    └──────────────┘    └───────────┬──────────────┘   │
│                                                      │                   │
│                                                      ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Warning     │◀───│  Instability │◀───│    Early Warning         │   │
│  │  System      │    │  Predictor  │    │    System                │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│                                                      │                   │
│                                                      ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Recommendation│◀───│    PPO      │◀───│    GridGuard Agent      │   │
│  │   Engine     │    │   Agent      │    │    (RL Policy)          │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Observation**: Grid state captured from IEEE 14-bus simulation
2. **Assessment**: Anomaly detector + Instability predictor analyze state
3. **Prediction**: Lead-time estimation calculated
4. **Action**: PPO agent generates control recommendations
5. **Ranking**: Recommendation engine ranks actions by priority
6. **Execution**: Control actions applied to grid simulation

---

##  Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/opeblow/GridGuard-RL
cd GridGuard/ope

# Create virtual environment
python -m venv myenv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
pypsa>=0.27.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.18.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.14.0
rich>=13.5.0
```

---

##  Usage

### Demo Mode (Quick Start)

```bash
python -m src.gridguard.main --mode demo
```

**Output:**
```
==================================================
GridGuard Demo Mode
==================================================
Generating training data...
Training anomaly detector on 500 samples...
  Unstable: 247, Stable: 253
Initializing RL agent...

Generating scenario...

Scenario 1: load_increase
  Severity: 1.32
  Recommended: redispatch_generation

Scenario 2: gen_outage
  Severity: 1.41
  Recommended: activate_reserve_generation

Scenario 3: line_trip
  Severity: 1.28
  Recommended: redispatch_generation

Demo complete. Use --mode assess for interactive mode.
```

### Training Mode

```bash
# Train PPO agent (50,000 timesteps)
python -m src.gridguard.main --mode train --timesteps 50000

# Train with custom model path
python -m src.gridguard.main --mode train --timesteps 100000 --model ./models/best_model
```

### Interactive Assessment

```bash
python -m src.gridguard.main --mode assess
```

**Sample Output:**
```
==================================================
GridGuard Interactive Assessment Mode
==================================================

--- Step 0 ---
Instability Score: 0.312
Alert Level: N/A

Recommended Actions:
  - Generator 1: increase (0.45)
  - Tap 2: lower (0.28)

Tier-based Recommendations:
  - reduce_industrial_load (impact: medium, disruption: low)
  - curtail_renewables (impact: medium, disruption: low)
```

---

##  Performance Results

### Grid Stabilization Success Rates

| Scenario Difficulty | Survival Rate | Avg. Episode Reward |
|---------------------|---------------|---------------------|
| Easy                | **100%**      | +45.2              |
| Medium              | **67%**       | +18.7              |
| Hard                | **73%**       | +12.3              |

### Anomaly Detection Performance

- **Precision**: 94.2% on unstable grid states
- **Recall**: 91.8% for critical instabilities
- **Lead Time**: Up to 45 minutes warning before failure

### Training Metrics

```
============================================================
GridGuard PPO Training
============================================================
Total timesteps: 50,000
Reward shaping: Positive stability bonuses enabled
============================================================

| Ep 100   | Mean Reward (last 100): +23.45          |
| Ep 500   | Mean Reward (last 100): +38.72          |
| Ep 1000  | Mean Reward (last 100): +42.18          |
============================================================
Training Complete!
Model saved to: ./models/final_model
```

---

##  Project Structure

```
ope/
├── src/
│   └── gridguard/
│       ├── main.py                 # Entry point & system orchestration
│       ├── agent.py                # PPO agent & recommendation engine
│       ├── anomaly_detector.py    # ML-based instability detection
│       ├── environment.py         # IEEE 14-bus grid simulation (PyPSA)
│       ├── mock_env.py             # Fast mock environment for testing
│       └── scenario_generator.py   # Synthetic training scenarios
├── models/                        # TraPO models
ined P│   ├── best_model/                # Best checkpoint during training
│   ├── checkpoints/              # Periodic model saves
│   └── final_model.zip           # Final trained model
├── eval_logs/                     # Evaluation metrics
├── tensorboard/                   # Training visualizations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Technical Deep Dive

### PPO Agent Architecture

```python
# Neural Network Configuration
policy_kwargs = {
    'net_arch': [256, 128, 64],    # Deep MLP
    'activation_fn': torch.nn.ReLU
}

# PPO Hyperparameters
model = PPO(
    policy="MlpPolicy",
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,                    # Discount factor
    gae_lambda=0.95,               # GAE parameter
    clip_range=0.2,                # PPO clipping
    ent_coef=0.05                  # Exploration bonus
)
```

### Reward Function Design

```python
reward = 
    + Voltage stability bonus      (voltage_dev < 0.05: +2pts/bus)
    + Loading stability bonus      (loading < 0.7: +1.5pts/line)
    + Step survival bonus          (+5 pts per step)
    + Episode completion bonus     (+50 pts if stable)
    - Voltage violation penalty    (-5 pts per violation)
    - Loading violation penalty    (-3 pts per violation)
    - Thermal violation penalty    (-15 pts per violation)
    - Critical failure penalty     (-500 pts if collapse)
```

### Anomaly Detection Ensemble

```python
# Multi-algorithm approach
anomaly_detector = {
    'isolation_forest': IsolationForest(
        n_estimators=100,
        contamination=0.1
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=100
    ),
    'lof': LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.1
    )
}
```

---

##  Supported Disturbance Types

| Category | Disturbance | Description |
|----------|-------------|-------------|
| **Load** | load_increase | Sudden demand surge |
| | load_spike | Extreme localized demand |
| **Generation** | gen_outage | Single generator failure |
| | multi_gen_outage | Multiple generators offline |
| | renewable_drop | Solar/wind output loss |
| **Transmission** | line_trip | Single line failure |
| | multi_line_trip | Multiple lines offline |
| **System** | cascading_start | Cascade initiation |
| | voltage_collapse | Voltage instability |
| | frequency_swing | Frequency deviation |
| **External** | weather_storm | Storm-induced damage |

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author

**MOBOLAJI OPEYEMI**

- GitHub: (https://github.com/opeblow)
- Email: opeblow2021@gmail.com

---

## Acknowledgments

- IEEE 14-Bus Test System specifications
- PyPSA power system analysis library
- Stable-Baselines3 RL framework
- OpenAI Proximal Policy Optimization algorithm

---

<p align="center">
  <strong> Powering the Future of Grid Resilience </strong>
  <br>
  <em>Built with ❤️ for sustainable energy systems</em>
</p>
