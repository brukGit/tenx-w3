# AlphaCare Insurance Solutions (ACIS) Data Analysis Project

## Project Overview

This project aims to analyze historical insurance claim data for AlphaCare Insurance Solutions (ACIS) to optimize marketing strategies and identify low-risk targets for potential premium reductions. The analysis covers data from February 2014 to August 2015.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Description](#data-description)
5. [Analysis Components](#analysis-components)
6. [Running Tests](#running-tests)
7. [Contributing](#contributing)
8. [License](#license)

## Project Structure

```
project/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   └── statistical_analysis.py
│   └── hypothesis_testing.py
│   └── statistical_modeling.py
├── notebooks/
│   ├── __init__.py
│   ├── README.md
│   └── exploratory_analysis.ipynb
│   └── hypothesis_testing.ipynb
│   └── statistical_modeling.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_eda.py
│   └── test_statistical_analysis.py
└── scripts/
    ├── __init__.py
    ├── README.md
    └── run_analysis.py
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/brukGit/tenx-w3.git
   cd acis-data-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your data file in the `data/` directory.

2. Run the main analysis script:
   ```
   python scripts/run_analysis.py
   ```

3. Open and run the Jupyter notebook for detailed exploratory data analysis:
   ```
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

## Data Description

The dataset includes information about insurance policies, transactions, client details, car specifications, and claim information. Key columns include:

- Policy information: UnderwrittenCoverID, PolicyID
- Transaction details: TransactionMonth
- Client information: IsVATRegistered, Citizenship, LegalType, etc.
- Car details: ItemType, Make, Model, VehicleType, etc.
- Insurance details: SumInsured, CalculatedPremiumPerTerm, etc.
- Claim information: TotalPremium, TotalClaims

## Analysis Components

1. **Data Loading and Preprocessing**: Handled by `src/data_loader.py`
2. **Exploratory Data Analysis**: Implemented in `src/eda.py` and demonstrated in the Jupyter notebook
3. **Statistical Analysis**: Provided by `src/statistical_analysis.py`
4. **Hypothesis Testing**: Provided by `src/hypothesis_testing.py`
5. **Statistical Modeling**: Provided by `src/statistical_modeling.py`

## Running Tests

To run the unit tests:

```
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

For more information or support, please contact the project maintainers.