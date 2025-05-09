# Randomized Coffee Trials (RCT) Automation

**Automated pipeline for optimizing team diversity and career-level representation in coffee trial events**

## Key Features

- **Stratified Splitting** by career level (Junior/Senior/Lead)
- **Greedy Diversity Optimization** across job areas (Engineering/Product/Design)

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

## Configuration (`.env`)

```ini
GROUP_SIZE=4
MAX_RETRIES=100
DIVERSITY_WEIGHT=0.7
STRATIFICATION_FIELDS=career_level,job_area
```

## How It Works

1. **Streamlit Interface** - Fully automated web UI handles the complete workflow
2. **Data Loading** - CSV/Excel input with participant metadata
3. **Stratification** - Balanced career-level representation per group
4. **Diversity Optimization** - Greedy algorithm maximizes job-area diversity
5. **Scheduling** - Automated calendar integration for meeting times
6. **Notifications** - Email/Slack reminders with participant bios
7. **Automated Outputs** - Export results as ICS calendars and CSV reports

## Group Generation Methodology

Our algorithm maximizes diversity across career levels (positions) and job sectors using combinatorial optimization. Here's how it works:

### Key Steps:
1. **Data Preparation**
   - Filter participants with valid position categories
   - Calculate rarity scores for each participant based on:
     - Position frequency (`1 / number_of_similar_positions`)
     - Sector frequency (`1 / number_of_similar_sectors`)

2. **Optimal Assignment**
   - Create a cost matrix where pairing participants with similar positions/sectors incurs high penalties
   - Position mismatch penalty: 100
   - Sector mismatch penalty: 33 (position is 3x more important)
   - Solve using OR-Tools' Linear Sum Assignment for minimal total cost

3. **Group Formation**
   - Create base groups from optimized assignment
   - Handle remainder participants (when total % group_size ≠ 0) by:
     - Identifying rarest participants using precomputed rarity scores
     - Assigning each rare participant to the group where they add maximum diversity

### Diversity Metrics:
- **Position Diversity:** Number of unique career levels per group
- **Sector Diversity:** Number of unique job sectors per group

### Example:
For 10 participants (3 Senior, 4 Mid, 3 Student) with group_size=4:
- Groups: 4, 3, 3
- Senior participants distributed across groups first
- Remainder Mid/Student added to maximize sector diversity
