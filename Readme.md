# DataPulse: Your Easy Data Analysis Tool

Hello! I'm **Rakesh Kapilavayi**, and I created **DataPulse** to help you explore and understand your data in a simple way. This tool is a web app that lets you:

- Upload a dataset (CSV or Excel)
- Clean it manually or automatically
- Create visualizations
- Run machine learning models
- Generate professional insights and recommendations

Whether you're new to data or an expert, this app makes data analysis fun and easy!

## 🌐 Live Demo
Try it out here: **[Click Here](https://rakesh-project-datapulse-automated-eda.streamlit.app/)**  

---

## 🚀 What Can DataPulse Do?

### 📂 Upload Your Data
- Add a CSV or Excel file to start.

### 📊 See a Summary
- Number of rows and columns
- Missing values
- Duplicate rows
- Data types and unique values

### 🧹 Clean Your Data

#### Manual Cleaning
- Choose how to handle missing values (mean, median, mode, drop rows, etc.)
- Delete duplicates
- Checkbox turns **light blue** when selected

#### Auto Cleaning
- Automatically:
  - Handle missing values
  - Remove duplicates
  - Cap outliers using IQR method

### 📈 Explore Your Data (EDA)
- **Histograms** for numerical columns
- **Scatter plot** for the two most correlated numeric columns
- **Heatmap** showing correlation matrix
- **Bar charts** for categorical columns
- **Box plots** to detect outliers

### 🤖 Run Machine Learning

#### Choose Type
- **Classification** (predict categories)
- **Regression** (predict numbers)

#### Select a Model
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM (Support Vector Machine)

**Features:**
- Auto-handles categorical variables
- View accuracy metrics and visual reports
- Confusion matrix for classification
- Feature importance rankings
- Cross-validation scores
- Option to tune model with hyperparameter optimization (for better performance; slower)
  - Checkbox turns **light blue** when selected

### 💡 Get Professional Insights
- **Enhanced Insights**: Get comprehensive, professionally formatted analysis reports including:
  - Dataset overview and characteristics
  - Key statistical findings and patterns
  - Correlation analysis and relationships
  - Data quality assessment
  - Strategic recommendations for next steps
  - Business impact interpretation
- **Quick Summary**: Get instant statistical observations
- **Raw Statistical Data**: Access detailed technical metrics

The insights combine statistical analysis with professional formatting to help you:
- Understand missing value patterns
- Identify important correlations
- Assess data quality
- Get actionable recommendations for further analysis

### 💾 Save Your Data
- Download the cleaned dataset as a CSV file
- Export insights reports as text files

---

## 🛠 What You Need

- A computer with **Python 3.8** or higher
- A web browser (Chrome, Firefox, Safari, Edge, etc.)
- Internet connection (for generating enhanced insights)

---

## ⚙️ How to Set It Up

### 1. Get the Files

Download or clone the project:

```bash
git clone https://github.com/rakeshkapilavayi/DataPulse-Automated-EDA.git
cd DataPulse-Automated-EDA
```

### 2. Set Up a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

This will install:

- Streamlit (web framework)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Plotly (interactive visualizations)
- Scikit-learn (machine learning)
- XGBoost (advanced ML models)
- Scipy (statistical functions)
- Openpyxl (Excel support)
- Google Generative AI (for enhanced insights)

### 4. Add a Logo (optional)
Place a **logo.png** file inside the project folder.

Don't have a logo? Remove or comment out the **st.sidebar.image()** line in **interface.py**.

---

## ▶️ How to Use DataPulse

### 1. Start the App
```bash
streamlit run interface.py
```
It will usually open at: [http://localhost:8501](http://localhost:8501)

### 2. Upload a Dataset
- Click **"Choose a CSV or Excel file"**.
- You'll see a preview of the dataset.
- Uploading a new file **resets the app**.

### 3. Use the Tabs

#### 🔍 Summary
- Basic stats on rows, columns, missing data, duplicates
- Column information with data types
- Unique value counts

#### 🛠 Manual Cleaning
- Fix missing data by choosing strategies per column
- Remove duplicates manually
- Full control over cleaning decisions

#### ⚙️ Auto Cleaning
- Clean your dataset automatically with one click
- See detailed cleaning report:
  - Which columns were handled
  - How many duplicates removed
  - Which outliers were capped

#### 📊 EDA (Exploratory Data Analysis)
- Visualize your data with interactive charts:
  - Distribution histograms
  - Correlation scatter plots (optimized for large datasets)
  - Correlation heatmaps
  - Categorical distribution bar charts

#### 🚨 Outliers
- Detect outliers using **box plots** for all numerical columns
- Visual identification of extreme values

#### 🤖 Machine Learning
- Choose task: **Classification** or **Regression**
- Select **target column** from appropriate data types
- Pick a **model** from the dropdown
- Optional: Enable **hyperparameter tuning** for better performance
- Click **Train Model** to view:
  - Evaluation metrics
  - Confusion matrix (classification)
  - Feature importance rankings
  - Cross-validation scores

#### 📊 Insights
- **Generate Enhanced Insights**: Get a comprehensive, professionally formatted analysis report covering:
  - Dataset characteristics and overview
  - Key statistical findings and their implications
  - Important correlations and relationships
  - Data quality assessment
  - Actionable recommendations
  - Business impact interpretation
- **Generate Quick Summary**: Get instant bullet-point insights
- **View Raw Statistical Data**: Access detailed technical metrics in expandable section

#### 💾 Export
- Download the cleaned dataset as a **CSV file**
- Download insights reports as **text files**

---

## 📁 Project Files

Here's what's inside the project folder:

```
DataPulse-Automated-EDA/
│
├── interface.py           # Main Streamlit application
├── functions.py           # Data cleaning, EDA, and insights logic
├── machinelearning.py     # ML models and training pipelines
├── llm_insights.py        # Enhanced insights generation module
├── requirements.txt       # Python dependencies
├── eda_app.log           # Application log file (auto-generated)
├── logo.png              # Optional sidebar logo
└── README.md             # Project documentation
```

---

## 🎯 Key Features

- ✅ **Easy to Use**: Clean, intuitive interface for all skill levels
- ✅ **Comprehensive EDA**: Multiple visualization types with interactive charts
- ✅ **Smart Cleaning**: Automated and manual data cleaning options
- ✅ **ML Ready**: Train models with just a few clicks
- ✅ **Professional Insights**: Get formatted, actionable analysis reports
- ✅ **Export Everything**: Download cleaned data and insights
- ✅ **Performance Optimized**: Handles large datasets efficiently
- ✅ **Session Management**: Properly handles multiple dataset uploads

---

## 📧 Contact

**Rakesh Kapilavayi**
- Email: rakeshkapilavayi978@gmail.com
- LinkedIn: [Rakesh Kapilavayi](https://www.linkedin.com/in/rakesh-kapilavayi-48b9a0342/)
- GitHub: [rakeshkapilavayi](https://github.com/rakeshkapilavayi)

---

## 📝 License

This project is open source and available for educational and personal use.

---

## 🙏 Acknowledgments

Built with:
- **Streamlit** for the web framework
- **Plotly** for interactive visualizations
- **Scikit-learn** & **XGBoost** for machine learning
- **Pandas** & **NumPy** for data processing

---

**Made with ❤️ by Rakesh Kapilavayi**


*Happy Data Analyzing! 🚀*
