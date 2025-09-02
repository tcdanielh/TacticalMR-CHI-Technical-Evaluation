# TacticalMR-CHI-Technical-Evaluation

## Setup

### Prerequisites
- Python 3.8+ (I'm specifically using Python 3.11.7)
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
```

2. Run the setup script:
```bash
python setup.py
```

This will:
- Check Python version compatibility
- Create necessary directories
- Install required packages
- Validate your data file (if exists)

3. Place your CSV data file in the `data/` directory.

## Usage

Run the main analysis script:
```bash
python analyze_study_data.py <path_to_csv>
```
