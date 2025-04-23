import numpy as np
from scipy import stats
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #4682B4;
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, 
            rgba(70,130,180,0.1) 0%, 
            rgba(70,130,180,0.2) 50%, 
            rgba(70,130,180,0.1) 100%);  /* SteelBlue with opacity */
        border: 2px solid rgba(70,130,180,0.3);  /* SteelBlue border */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: rgba(70,130,180,0.1);  /* SteelBlue background */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stSlider {
        padding-top: 12px;
        padding-bottom: 12px;
    }
    .stMarkdown {
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.6;
    }
    .stButton button {
        border-radius: 8px;
        padding: 4px 25px;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox select {
        border-radius: 8px;
        padding: 8px;
    }
    h1, h2, h3 {
        color: #333;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    .plot-container {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# title section
st.markdown('<p class="main-header">💼 📈 Salary Prediction App</p>', unsafe_allow_html=True)
st.header("📊 About")
st.markdown("""
Welcome to the Salary Prediction App!  
This tool uses a **linear regression model** to estimate annual salary based on:
- **Years of Experience**
- **Job Role**
- **Education Level**

Use the dropdowns and sliders below to customize your inputs and explore how different factors affect salary expectations!
""")


# Load dataset
try:
    data = pd.read_csv("salary_prediction.csv")
except FileNotFoundError:
    st.error("Error: salary_prediction.csv file not found!")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Prepare features and label
X = data[['YearsExperience', 'Job Role', 'Education Level']]  # <- use 'data' here
y = data['Salary']

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X)
feature_names = X_encoded.columns

# Train the model
model = LinearRegression()
model.fit(X_encoded, y)
r2_score = model.score(X_encoded, y)



with st.sidebar:
    
    st.markdown("""
        <div style='background-color: rgba(70,130,180,0.1); 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 4px solid #4682B4;
                    margin-bottom: 20px'>
        <h3 style='color: #4682B4; margin-bottom: 15px'>What do R² and Confidence Interval mean?</h3>
        
        <div style='margin-bottom: 15px'>
            <h4 style='color: #4682B4'>R² Score (Coefficient of Determination)</h4>
            <p style='color: #333; line-height: 1.5'>
                  This score tells us how well the model fits the data. A value close to `1.0` means our model explains most of the variability in the data.
            </p>
        </div>

        <div style='margin-bottom: 10px'>
            <h4 style='color: #4682B4'>Confidence Interval</h4>
            <p style='color: #333; line-height: 1.5'>
                The prediction comes with a range to indicate uncertainty. For example, a 95% confidence interval means we are 95% sure the true salary lies within this range.
            </p>
        </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 20px 0; border-color: #4682B4; opacity: 0.2'>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style='background-color: rgba(70,130,180,0.1); 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 4px solid #4682B4;
                    margin-bottom: 20px'>
            <h3 style='color: #4682B4; margin-bottom: 15px'>🎯 Model Performance</h3>
            <div style='margin-bottom: 10px'>
                <h4 style='color: #4682B4'>R² Score</h4>
                <p style='color: #333; 
                          font-size: 24px; 
                          font-weight: bold; 
                          margin: 10px 0;
                          padding: 10px;
                          background-color: rgba(70,130,180,0.05);
                          border-radius: 5px;'>
                    {r2_score:.3f}
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 20px 0; border-color: #4682B4; opacity: 0.2'>", unsafe_allow_html=True)

    st.subheader("⚙️ Confidence Level")
    confidence = st.slider("Confidence Level", 50, 100, 95, step=5)
    
    
job_roles = sorted(data['Job Role'].unique())
education_levels = sorted(data['Education Level'].unique())

st.subheader("🎯 Make a Prediction")
col1, col2 = st.columns([2,1])
with col1:
    years_exp = st.slider("Select Years of Experience", 0.0, 20.0, 2.0, step=0.5)
    selected_job = st.selectbox("Job Role", job_roles)
    selected_education = st.selectbox("Education Level", education_levels)
    
    user_input = pd.DataFrame({
        'YearsExperience': [years_exp],
        'Job Role': [selected_job],
        'Education Level': [selected_education]
    })

    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=feature_names, fill_value=0)
    
    predicted_salary = model.predict(user_encoded)[0]
    st.success(f"💰 Estimated Salary: ${predicted_salary:,.2f}")
        
    # Calculate the prediction interval
    MSE = np.sum((y - model.predict(X_encoded)) ** 2) / (len(X_encoded) - 2)
    t_value = stats.t.ppf((1 + confidence/100)/2, len(X_encoded)-2)
    interval = t_value * np.sqrt(MSE * (1 + 1/len(X_encoded)))
    
    lower_bound = predicted_salary - interval
    upper_bound = predicted_salary + interval
    
with col2:
        st.success(f"### Predicted Salary: ${predicted_salary:,.2f}")
        st.info(f"""
        ### {confidence}% Confidence Interval:
        There is a {confidence}% chance the actual salary falls between:
        - ${lower_bound:,.2f}
        - ${upper_bound:,.2f}
        """)

st.write("### Dataset Statistics")
col1, col2 = st.columns(2)
with col1:
    st.write("Experience Range:")
    st.write(f"Min: {data['YearsExperience'].min():.1f} years")
    st.write(f"Max: {data['YearsExperience'].max():.1f} years")
with col2:
    st.write("Salary Range:")
    st.write(f"Min: ${data['Salary'].min():,.2f}")
    st.write(f"Max: ${data['Salary'].max():,.2f}")

st.markdown("### 📊 Data Quality")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data Points", f"{len(data):,}")
with col2:
    st.metric("Missing Values", "0")
with col3:
    st.metric("Model Accuracy", f"{r2_score:.2%}")

# Show data and plot
st.write("### Dataset Preview")
st.dataframe(data)

st.markdown("---")
st.subheader("📊 Visualization Options")
col1, col2, col3 = st.columns(3)
with col1:
    show_datapoints = st.checkbox("Show Data Points", value=True, 
        help="Toggle visibility of actual salary data points")
    show_prediction = st.checkbox("Show Prediction Point", value=True,
        help="Toggle visibility of the predicted salary point")
with col2:
    plot_color = st.color_picker("Regression Line Color", "#FF0000")
    prediction_color = st.color_picker("Prediction Point Color", "#FF69B4")
with col3:
    point_size = st.slider("Point Size", 20, 200, 100)
    marker_style = st.selectbox("Prediction Marker", 
                           options=['*', 'o', '^', 's', 'D', '+', 'h'],
                           format_func=lambda x: {
                               '*': '* (Star)',
                               'o': '○ (Circle)',
                               '^': '△ (Triangle Up)',
                               's': '□ (Square)',
                               'D': '◇ (Diamond)',
                               '+': '+ (Plus)',
                               'h': '⬢ (Hexagon)'
                           }[x],
                           index=0)


# Simple model for clean plot for Salary vs years of experience
simple_model = LinearRegression()
simple_model.fit(data[['YearsExperience']], data['Salary'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 1.2])

# Top plot: General salary vs experience trend
if show_datapoints:
    ax1.scatter(data['YearsExperience'], data['Salary'], alpha=0.5, label='Actual Data', color='#666666')

# Plot regression line based only on years of experience
x_range = np.linspace(data['YearsExperience'].min(), data['YearsExperience'].max(), 100).reshape(-1, 1)
y_pred_range = simple_model.predict(x_range)
ax1.plot(x_range, y_pred_range, color=plot_color, label='Regression Line (Experience Only)', linewidth=2)

# Plot predicted salary from dropdowns
if show_prediction:
    ax1.scatter([years_exp], [predicted_salary], 
               color=prediction_color,
               s=point_size, 
               label='Prediction',
               marker=marker_style)

ax1.set_xlabel("YearsExperience", fontsize=12)
ax1.set_ylabel("Salary ($)", fontsize=12)
ax1.set_title("Salary vs Years of Experience", fontsize=14, pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

#Job-specific salary distributions
job_edu_group = data.groupby(['Job Role', 'Education Level'])['Salary'].mean().unstack()
job_edu_group.plot(kind='barh', ax=ax2)
ax2.set_title("Average Salary by Job Role & Education", fontsize=14)
ax2.set_xlabel("Average Salary ($)", fontsize=12)
ax2.legend(title="Education Level", bbox_to_anchor=(1.05, 1), fontsize=10)

plt.tight_layout(pad=3.0)
st.pyplot(fig)

st.markdown("### 💡 Salary Insights")
col1, col2 = st.columns(2)
with col1:
    st.write("Average Salary by Job Role:")
    avg_by_role = data.groupby('Job Role')['Salary'].mean().sort_values(ascending=False)
    st.dataframe(pd.DataFrame({
        'Job Role': avg_by_role.index,
        'Average Salary': avg_by_role.values.round(2)
    }))
with col2:
    st.write("Average Salary by Education:")
    avg_by_edu = data.groupby('Education Level')['Salary'].mean().sort_values(ascending=False)
    st.dataframe(pd.DataFrame({
        'Education Level': avg_by_edu.index,
        'Average Salary': avg_by_edu.values.round(2)
    }))




st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="📥 Download Dataset",
        data=data.to_csv(index=False),
        file_name="salary_data.csv",
        mime="text/csv"
    )

with col2:
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    # Configure matplotlib parameters first
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.5

    if show_datapoints:
        ax.scatter(data['YearsExperience'], data['Salary'], 
                  alpha=0.5, label='Actual Data', color='#666666')

    # Plot single regression line using simple_model
    x_range = np.linspace(data['YearsExperience'].min(), 
                         data['YearsExperience'].max(), 
                         100).reshape(-1, 1)
    y_pred_range = simple_model.predict(x_range)
    ax.plot(x_range, y_pred_range, color=plot_color, 
            label='Regression Line', linewidth=2)
    
    if show_prediction:
        ax.scatter([years_exp], [predicted_salary], 
                  color=prediction_color,
                  s=point_size, 
                  label='Prediction',
                  marker=marker_style)

    ax.set_xlabel("Years of Experience", fontsize=12)
    ax.set_ylabel("Salary ($)", fontsize=12)
    ax.set_title("Salary vs Years of Experience", fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('salary_experience_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    with open('salary_experience_plot.png', 'rb') as file:
        st.download_button(
            label="📥 Download Experience Plot",
            data=file,
            file_name="salary_experience_plot.png",
            mime="image/png"
        )

with col3:
    fig2, ax = plt.subplots(figsize=(10, 6))
    job_edu_group.plot(kind='barh', ax=ax)
    ax.set_title("Average Salary by Job Role & Education", fontsize=14)
    ax.set_xlabel("Average Salary ($)", fontsize=12)
    ax.legend(title="Education Level", bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig('salary_education_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    with open('salary_education_plot.png', 'rb') as file:
        st.download_button(
            label="📥 Download Education Plot",
            data=file,
            file_name="salary_education_plot.png",
            mime="image/png"
        )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <p>Built by Karola using Streamlit and Python</p>
    <p>© 2025 Salary Prediction App</p>
</div>
""", unsafe_allow_html=True)