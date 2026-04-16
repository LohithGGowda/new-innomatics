"""
data/sample_data.py
-------------------
Contains sample resumes and job description for testing the pipeline.
Three candidate profiles: Strong, Average, and Weak.
"""

# ─────────────────────────────────────────────
# JOB DESCRIPTION
# ─────────────────────────────────────────────

JOB_DESCRIPTION = """
Position: Senior Data Scientist
Company: TechCorp Analytics

Requirements:
- 3+ years of experience in Data Science or Machine Learning
- Proficiency in Python and R
- Strong knowledge of Machine Learning frameworks: Scikit-learn, TensorFlow, PyTorch
- Experience with SQL and NoSQL databases
- Familiarity with cloud platforms: AWS, GCP, or Azure
- Experience with data visualization tools: Tableau, Power BI, or Matplotlib/Seaborn
- Knowledge of statistical analysis and hypothesis testing
- Experience deploying ML models to production (MLOps)
- Familiarity with version control (Git)
- Strong communication skills for presenting findings to stakeholders

Preferred Qualifications:
- Master's or PhD in Computer Science, Statistics, or related field
- Experience with NLP or Computer Vision
- Knowledge of big data tools: Spark, Hadoop
- Publications or open-source contributions

Responsibilities:
- Build and deploy machine learning models
- Analyze large datasets to derive business insights
- Collaborate with engineering and product teams
- Communicate results to non-technical stakeholders
"""

# ─────────────────────────────────────────────
# RESUME 1 — STRONG CANDIDATE
# ─────────────────────────────────────────────

RESUME_STRONG = """
Name: Aisha Patel
Email: aisha.patel@email.com
LinkedIn: linkedin.com/in/aishapatel

EDUCATION
- PhD in Computer Science (Machine Learning), IIT Bombay, 2020
- B.Tech in Computer Science, IIT Delhi, 2015

EXPERIENCE
Senior Data Scientist | DataDriven Inc. | 2020 – Present (4 years)
- Led end-to-end ML pipeline development using Python, TensorFlow, and PyTorch
- Deployed 10+ production ML models on AWS SageMaker with CI/CD pipelines (MLOps)
- Built NLP models for sentiment analysis improving customer insights by 35%
- Mentored a team of 4 junior data scientists
- Presented quarterly insights to C-suite stakeholders

Data Scientist | Analytics Corp | 2017 – 2020 (3 years)
- Developed predictive models using Scikit-learn and XGBoost
- Designed SQL/PostgreSQL data pipelines for 50M+ row datasets
- Created interactive dashboards in Tableau and Power BI
- Performed A/B testing and statistical hypothesis testing

SKILLS
- Languages: Python, R, SQL, Spark
- ML Frameworks: TensorFlow, PyTorch, Scikit-learn, XGBoost
- Cloud: AWS (SageMaker, S3, EC2), GCP (BigQuery, Vertex AI)
- Databases: PostgreSQL, MongoDB, Redis
- Visualization: Tableau, Power BI, Matplotlib, Seaborn
- Tools: Git, Docker, Kubernetes, Airflow, MLflow
- Specializations: NLP, Computer Vision, Time Series

ACHIEVEMENTS
- Published 3 papers in top ML conferences (NeurIPS, ICML)
- Open-source contributor: 2k+ GitHub stars on ML utility library
- AWS Certified Machine Learning Specialist
"""

# ─────────────────────────────────────────────
# RESUME 2 — AVERAGE CANDIDATE
# ─────────────────────────────────────────────

RESUME_AVERAGE = """
Name: Rahul Sharma
Email: rahul.sharma@email.com

EDUCATION
- Master's in Data Science, University of Pune, 2021
- B.Sc in Mathematics, Pune University, 2019

EXPERIENCE
Data Analyst | StartupXYZ | 2021 – Present (3 years)
- Analyzed data using Python (Pandas, NumPy) and SQL
- Built basic machine learning models with Scikit-learn for churn prediction
- Created visualizations using Matplotlib and Seaborn
- Wrote SQL queries for business reporting

Intern – Data Science | TechFirm | 2020 – 2021 (1 year)
- Assisted in data cleaning and preprocessing
- Worked on a Jupyter Notebook-based ML project
- Learned basics of TensorFlow

SKILLS
- Languages: Python, SQL
- ML: Scikit-learn, basic TensorFlow
- Visualization: Matplotlib, Seaborn, basic Tableau
- Tools: Git, Jupyter Notebook
- Databases: MySQL

PROJECTS
- Customer Churn Prediction using Logistic Regression (Scikit-learn)
- Sales Forecasting with ARIMA models
- Data dashboard in Tableau for retail client
"""

# ─────────────────────────────────────────────
# RESUME 3 — WEAK CANDIDATE
# ─────────────────────────────────────────────

RESUME_WEAK = """
Name: Vikram Singh
Email: vikram.singh@email.com

EDUCATION
- B.Com (Bachelor of Commerce), Delhi University, 2022

EXPERIENCE
Data Entry Operator | LocalBusiness Ltd. | 2022 – Present (2 years)
- Entered data into Excel spreadsheets
- Generated basic Excel charts and pivot tables
- Assisted in monthly sales reporting

Intern – Office Assistant | SME Corp | 2021 – 2022 (6 months)
- Filed documents and maintained records
- Helped with MS Word and PowerPoint presentations

SKILLS
- Microsoft Excel (intermediate)
- Microsoft Word, PowerPoint
- Basic internet research
- Data entry

CERTIFICATIONS
- Completed a 1-month online course: "Introduction to Python"
- Google Analytics Beginner Certificate

INTERESTS
- Interested in learning machine learning
- Watching YouTube tutorials on data science
"""

# ─────────────────────────────────────────────
# CANDIDATE REGISTRY
# ─────────────────────────────────────────────

CANDIDATES = [
    {"name": "Aisha Patel",  "type": "Strong",  "resume": RESUME_STRONG},
    {"name": "Rahul Sharma", "type": "Average", "resume": RESUME_AVERAGE},
    {"name": "Vikram Singh", "type": "Weak",    "resume": RESUME_WEAK},
]
