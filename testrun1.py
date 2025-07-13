import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Fake Internship Detector", layout="wide")
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/fake_job_postings.csv")
        st.write("CSV loaded, rows:", df.shape[0])
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

    
    df['title'] = df['title'].str.lower()
    df['description'] = df['description'].str.lower()
    intern_df = df[df['title'].str.contains('intern')]
    intern_df.set_index('job_id', inplace=True)
    
    # Clean & feature engineering
    intern_df['department'].fillna('unknown', inplace=True)
    def calc_avgsalary(salary_range):
        if pd.isnull(salary_range): return np.nan
        if '-' in salary_range:
            parts = salary_range.replace('$','').split('-')
            try:
                return (int(parts[0]) + int(parts[1])) / 2
            except: return np.nan
        else:
            try: return float(salary_range)
            except: return np.nan
    
    intern_df['avg_salary'] = intern_df['salary_range'].apply(calc_avgsalary)
    intern_df.drop(columns=['salary_range','telecommuting'], inplace=True)
    intern_df['avg_salary'].fillna(intern_df['avg_salary'].median(), inplace=True)
    intern_df['company_profile'].replace(['', 'no profile'], np.nan, inplace=True)
    intern_df['requirements'].replace('', np.nan, inplace=True)
    intern_df['required_education'].replace('', np.nan, inplace=True)
    intern_df['likely_fake'] = (
        intern_df['company_profile'].isna() |
        intern_df['requirements'].isna() |
        intern_df['required_education'].isna() |
        (intern_df['fraudulent'] == 1)
    ).astype(int)
    intern_df['employment_type'].fillna('Unspecified', inplace=True)
    intern_df['employment_type'] = intern_df['employment_type'].replace({'Other': 'Unspecified', 'Temporary': 'Short Term'})
    intern_df['required_experience'].fillna('no experience', inplace=True)
    return intern_df

intern_df = load_data()

st.title("Fake Internship Detection Dashboard")
st.markdown("Explore internship trends and detect potentially fake postings using data analysis.")

# Show raw data
with st.expander("🔍 Show raw internship data"):
    st.dataframe(intern_df.reset_index().head(20))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fake vs Real Internships")
    st.caption("This bar chart shows the number of internships flagged as fake vs real. A significant number lack company profile or requirements.")
    count = intern_df['likely_fake'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=count.index, y=count.values, palette=['#009688', '#FFC107'], ax=ax)
    ax.set_xticklabels(['Fake', 'Real'])
    ax.set_ylabel('Number of Internships')
    ax.set_title('Internship Authenticity Count')
    st.pyplot(fig)

with col2:
    st.subheader("Salary Distribution")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=intern_df, x='likely_fake', y='avg_salary', palette='coolwarm', ax=ax2)
    ax2.set_xticklabels(['Fake', 'Real'])
    ax2.set_xlabel("Internship Type")
    ax2.set_ylabel("Average Salary")
    ax2.set_title("Real vs Fake Salary Distribution")
    st.pyplot(fig2)

# Charts row 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("Experience Level")
    st.caption("Fake internships are more likely to demand 'no experience' or leave this field blank to appear accessible.")
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.countplot(data=intern_df, x='required_experience', hue='likely_fake', palette='coolwarm', ax=ax3)
    ax3.set_title("Experience Required in Real vs Fake Internships")
    ax3.set_xlabel("Experience Level")
    ax3.set_ylabel("Count")
    ax3.legend(labels=['Fake', 'Real'])
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with col4:
    st.subheader("Employment Type")
    st.caption("Fake internships often use vague terms like 'Short Term' or 'Unspecified', while real ones are mostly Full-time or Part-time.")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=intern_df, x='employment_type', hue='likely_fake', palette='coolwarm', ax=ax4)
    ax4.set_title("Employment Type Distribution")
    ax4.set_xlabel("Type")
    ax4.set_ylabel("Count")
    ax4.legend(labels=['Fake', 'Real'])
    plt.xticks(rotation=30)
    st.pyplot(fig4)

# WordCloud
st.subheader("Common Words in Fake Internship Titles")
st.caption("This WordCloud highlights the most frequent words found in fake internship titles. Watch for repeated vague terms.")
fake_titles = intern_df[intern_df['likely_fake'] == 1]['title']
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_titles))
fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.imshow(wordcloud, interpolation='bilinear')
ax5.axis('off')
st.pyplot(fig5)

st.success("Dashboard loaded successfully! Explore more by scrolling and expanding sections.")
