# fake_internship_analyzer
A **Streamlit** dashboard for analyzing internship job postings and detecting potentially **fake internships** using data-driven insights.
## 🚀 Features

- **Interactive Visualizations** using Seaborn and Matplotlib
- **Data Cleaning** and Feature Engineering on Internship Postings
- **Fake vs Real Classification** based on missing info & fraudulent flags
- **WordCloud** to highlight frequent words in fake internship titles
- Simple, informative **captions** for each graph to aid understanding

## Insight Highlights

- Many fake internships lack **company profiles**, **requirements**, or **educational details**
- Fake internships often offer **lower or unclear salary ranges**
- They commonly use **vague employment types** like "Unspecified" or "Short Term"
- The use of **"no experience"** in job descriptions is higher among fakes

##  Dataset

- Source: [`fake_job_postings.csv`](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)
- Filtered to include only **internship** related titles

## Data Preprocessing

- Converted all text to lowercase
- Filtered internship postings (`title` contains "intern")
- Engineered features:
  - `avg_salary` calculated from salary range
  - `likely_fake` label based on missing critical fields or fraud flag

## Visualizations

- **Bar Chart**: Fake vs Real Internships
- **Boxplot**: Salary distribution comparison
- **Countplots**:
  - Required Experience
  - Employment Type
- **WordCloud**: Common words in fake internship titles

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
