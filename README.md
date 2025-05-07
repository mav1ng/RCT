# Randomized Coffee Trials (RCT) Automation

**Automated pipeline for optimizing team diversity and career-level representation in coffee trial events**

## Key Features

- **Stratified Splitting** by career level (Junior/Senior/Lead)
- **Greedy Diversity Optimization** across job areas (Engineering/Product/Design)
- **Pipeline Automation** from participant input to meeting scheduling
- **Reporting Tools** for diversity metrics and participation tracking
- **CI/CD Integration** for automated batch processing

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

