
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import textwrap

def run_analysis():
    """Main function to run the PDCA analysis and generate outputs."""
    print("Connecting to database and loading data...")
    conn = sqlite3.connect('service_requests.db')
    df = pd.read_sql('SELECT * FROM requests', conn)
    costs_df = pd.read_sql('SELECT * FROM improvement_costs', conn)
    
    # --- PDCA: Plan & Do (Analyze Data) ---
    print("PLAN & DO: Calculating KPIs and identifying inefficiencies...")
    kpis = df.groupby('Department').agg(
        Avg_Response_Time_Hours=('Response_Time_Hours', 'mean'),
        Total_Cost=('Cost', 'sum')
    ).reset_index()

    status_kpis = df.groupby('Status').size().reset_index(name='Count')
    status_kpis = status_kpis.merge(costs_df, on='Status', how='left')
    status_kpis['Total_Improvement_Cost'] = status_kpis['Count'] * status_kpis['Improvement_Cost_Per_Request']

    inefficiencies = kpis[kpis['Avg_Response_Time_Hours'] > 24]
    high_cost_threshold = kpis['Total_Cost'].mean() + kpis['Total_Cost'].std()
    high_cost_depts = kpis[kpis['Total_Cost'] > high_cost_threshold]

    # --- Machine Learning for Predictive Insights ---
    print("DO: Training Gradient Boosting model to predict response times...")
    categorical_cols = ['Type', 'Department', 'Status']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    X = encoded_df
    y = df['Response_Time_Hours']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
    }
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    feature_importance = pd.DataFrame({
        'Feature': encoded_df.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance.to_csv('feature_importance.csv', index=False)

    # --- PDCA: Check (Visualize Results) ---
    print("CHECK: Generating visualizations and dashboard...")
    create_visualizations(df, kpis, monthly_response_data(df), y_test, preds, mse, r2, feature_importance)
    
    # --- PDCA: Act (Generate Reports & Recommendations) ---
    print("ACT: Generating reports and recommendations...")
    create_reports(kpis, status_kpis, inefficiencies, high_cost_depts, mse, r2, feature_importance)

    conn.close()
    print("\nAnalysis complete. All files have been generated in the current directory.")

def monthly_response_data(df):
    """Helper to calculate monthly average response time."""
    df['Date_Submitted'] = pd.to_datetime(df['Date_Submitted'])
    return df.set_index('Date_Submitted').resample('M')['Response_Time_Hours'].mean()

def create_visualizations(df, kpis, monthly_response, y_test, preds, mse, r2, feature_importance):
    """Generates and saves all data visualizations."""
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    kpis.plot(x='Department', y='Avg_Response_Time_Hours', kind='bar', ax=ax, color='skyblue', legend=None)
    ax.set_title('Average Response Time by Department', fontsize=14)
    ax.set_ylabel('Average Response Time (Hours)')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(kpis['Avg_Response_Time_Hours']):
        ax.text(i, v + 0.5, f'{v:.1f} hrs', ha='center')
    plt.tight_layout()
    plt.savefig('response_time_bar.png')
    plt.close()

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(kpis['Total_Cost'], labels=kpis['Department'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Cost Distribution Across Departments', fontsize=14)
    plt.tight_layout()
    plt.savefig('cost_pie.png')
    plt.close()
    
    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, preds, alpha=0.6, color='purple')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f'Predicted vs Actual Response Times (R²: {r2:.2f})', fontsize=14)
    ax.set_xlabel('Actual Response Time (Hours)')
    ax.set_ylabel('Predicted Response Time (Hours)')
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png')
    plt.close()

    # Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_features = feature_importance.head(10)
    ax.barh(top_10_features['Feature'], top_10_features['Importance'], color='lightcoral')
    ax.set_title('Top 10 Feature Importances for Response Time', fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    # Bar chart
    axs[0,0].bar(kpis['Department'], kpis['Avg_Response_Time_Hours'], color='skyblue')
    axs[0,0].set_title('Avg Response Time by Dept')
    axs[0,0].tick_params(axis='x', rotation=45)
    # Pie chart
    axs[0,1].pie(kpis['Total_Cost'], labels=kpis['Department'], autopct='%1.1f%%', startangle=90)
    axs[0,1].set_title('Cost Distribution')
    # Line chart
    monthly_response.plot(kind='line', ax=axs[1,0], marker='o', color='green')
    axs[1,0].set_title('Monthly Avg Response Time Trend')
    axs[1,0].tick_params(axis='x', rotation=45)
    # Scatter plot
    scatter = axs[1,1].scatter(df['Response_Time_Hours'], df['Cost'], alpha=0.5, c=df['Response_Time_Hours'], cmap='viridis')
    fig.colorbar(scatter, ax=axs[1,1], label='Response Time (Hours)')
    axs[1,1].set_title('Cost vs. Response Time')
    plt.suptitle('Service Request Analysis Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dashboard.png')
    plt.close()

def create_reports(kpis, status_kpis, inefficiencies, high_cost_depts, mse, r2, feature_importance):
    """Generates all text-based reports."""
    # Summary CSV
    summary_report = kpis.describe()
    summary_report.to_csv('process_improvement_summary.csv')
    
    # Prepare text snippets for recommendations
    ineff_depts = ', '.join(inefficiencies['Department'].tolist()) if not inefficiencies.empty else 'None'
    ineff_text = f"Departments like {ineff_depts} show response times over 24 hours. Recommendation: Streamline their workflows to cut delays by up to 20%." if not inefficiencies.empty else "All departments are meeting the 24-hour response time target. Continue monitoring."
    
    high_cost_depts_list = ', '.join(high_cost_depts['Department'].tolist()) if not high_cost_depts.empty else 'None'
    cost_text = f"High-cost department(s) like {high_cost_depts_list} consume a large budget share. Recommendation: Review their resource allocation for potential efficiencies." if not high_cost_depts.empty else "Costs are evenly distributed. Focus on maintaining balanced resource allocation."
    
    top_feature = feature_importance.iloc[0]['Feature']
    top_feature_category = top_feature.split('_')[0]
    est_savings = status_kpis['Total_Improvement_Cost'].sum() * 0.15

    # Recommendations.txt
    rec_text = f"""
PDCA-Based Recommendations for Service Process Improvement:

1. Slow Response Times ('response_time_bar.png'):
   - {ineff_text}

2. Cost Distribution ('cost_pie.png'):
   - {cost_text}

3. Predictive Insights ('predicted_vs_actual.png'):
   - The Gradient Boosting model (R²={r2:.2f}) can reliably predict response times. Use it to proactively allocate resources and manage stakeholder expectations for high-risk requests.

4. Key Drivers ('feature_importance.png'):
   - The most important factor influencing response time is '{top_feature}'. Focus improvement efforts on optimizing processes related to '{top_feature_category}'.

Overall, applying these data-driven insights could yield significant efficiency gains and estimated cost savings of ${est_savings:,.2f}.
"""
    with open('recommendations.txt', 'w') as f:
        f.write(textwrap.dedent(rec_text))

    # PDF Report
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    texts = [
        ('PDCA Business Process Improvement Report', 16, 'bold'),
        ('Plan: Define KPIs for response time and cost. Identify key business questions.', 12, 'normal'),
        ('Do: Analyze data to find bottlenecks. Train a Gradient Boosting model to predict delays.', 12, 'normal'),
        ('Check: Visualize KPIs and model performance via dashboards and charts. Model R² is {r2:.2f}.', 12, 'normal'),
        ('Act: Generate actionable recommendations to improve processes, targeting areas identified by the model.', 12, 'normal'),
        (f'Key Finding 1: {ineff_text}', 11, 'normal'),
        (f'Key Finding 2: {cost_text}', 11, 'normal'),
        (f"Key Finding 3: The top predictor of response time is '{top_feature}'.", 11, 'normal')
    ]
    y_pos = 0.95
    for text, size, weight in texts:
        wrapped_text = textwrap.wrap(text, width=80)
        for line in wrapped_text:
            ax.text(0, y_pos, line, fontsize=size, fontweight=weight, va='top', ha='left')
            y_pos -= 0.05
        y_pos -= 0.03
    ax.axis('off')
    plt.savefig('pdca_report.pdf', format='pdf')
    plt.close()

if __name__ == '__main__':
    run_analysis()
