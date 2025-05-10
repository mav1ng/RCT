# RCT Group Optimizer

Web app for creating participant groups with optimal diversity and size balance.

## Features
- **Stratified Grouping**: Preserve position category ratios using scikit-learn
- **Greedy Optimization**: Maximize diversity scores through iterative placement
- **Multi-Trial Runs**: Test 10+ configurations with different random seeds
- **Automated Validation**: Ensure group sizes in [n, n+1] range
- **Excel Reports**: Export results with assignment methodology tracking

## Diversity Scoring
```python
# Position diversity (primary optimization factor)
+120 per unique position | -60 per duplicate

# Sector diversity (secondary optimization factor)
+60 per unique job sector | -20 per duplicate
```

## Group Formation Algorithm

### Optimization Objective
Maximize total diversity score:
```math
TotalScore = Σ_{groups} [PositionScore + SectorScore]
```
Where:
- `PositionScore = 120 × U_p - 60 × D_p`  
  (U_p: unique positions, D_p: duplicate positions)
- `SectorScore = 60 × U_s - 20 × D_s`  
  (U_s: unique sectors, D_s: duplicate sectors)

### Multi-Trial Optimization
```python
for trial in range(TRIALS):
    random.seed(trial)  # Deterministic randomness
    groups = stratified_split()
    groups = greedy_optimize(groups)
    groups = swap_optimize(groups)
    track_best_config(groups)
```

## Usage
1. Install requirements:  
`pip install -r requirements.txt`
2. Launch app:  
`streamlit run app.py`
3. Upload CSV or Excel with columns:
   - Position_Category
   - Job_Sector  
   - Name
   - Email

## Configuration (app.py)
```python
# Customizable parameters if not running in Streamlit
GROUP_SIZE = 5  # Ideal members per group
TRIALS = 10     # Number of optimization attempts
SEED = 42       # Reproducible randomness
```

## Installation

```bash
pip install -r requirements.txt
```

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Launches an automated pipeline UI that handles:
1. Participant CSV upload
2. Real-time diversity visualization
3. Automated group assignment
4. Export results as ICS calendars and CSV reports

## How It Works

1. **Streamlit Interface** - Fully automated web UI handles the complete workflow
2. **Data Loading** - CSV/Excel input with participant metadata
3. **Stratification** - Balanced career-level representation per group
4. **Diversity Optimization** - Greedy algorithm maximizes job-area diversity and position diversity
5. **Scheduling** - Automated calendar integration for meeting times
6. **Notifications** - Email/Slack reminders with participant bios
7. **Automated Outputs** - Export results as ICS calendars and CSV reports

## Group Formation Process

### Step 1: Base Group Creation
- **Optimized Sorting:**  
  Participants are ordered to minimize:  
  ```math
  Total Cost = \sum (position\_matches \times 100 + sector\_matches \times 33)
  ```
- **Group Formation:**  
  Initial groups are created from this optimized sequence to ensure maximum diversity

### Step 2: Remainder Handling
- If total participants aren't divisible by `group_size`:
  1. **Remove Incomplete Groups:** Any partially filled groups are dissolved
  2. **Optimized Redistribution:** Each remaining participant is assigned to existing groups to maximize:
     - **Position Diversity** (weight: 100)
     - **Sector Diversity** (weight: 33)
     - **Group Size Balance** (prefer less full groups)

## Dependencies
- pandas
- numpy
- scipy
