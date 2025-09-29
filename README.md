# Process Improvement Analysis

### **Business Summary**

- **Problem:** In emergency medical services (EMS), response times are a critical factor in patient outcomes. Inefficiencies anywhere in the process, from the initial 911 call through hospital arrival, can lead to delays, increased costs, and risk to public health. Identifying the specific bottlenecks within this complex workflow is a major operational challenge.
  
- **Process:** This analysis applies **data-driven process improvement methodologies (like PDCA/Lean)** to a dataset of emergency response records. Analysis of timestamps from dispatch, vehicle GPS, and patient care systems enabled mapping of the entire response timeline and measurement of the duration of each individual step.
  
- **Solution:** This project produced a detailed process analysis report that visualizes the end-to-end response workflow. The report pinpoints the exact stages that cause the most significant delays (bottlenecks) and calculates their impact on overall response time and operational costs.
  
- **Impact:** With this clear, evidence-based analysis, EMS leadership can implement targeted **Quality Improvement (QI)** initiatives with confidence. By addressing the identified bottlenecks—whether through revised dispatch protocols, better unit staging, or enhanced training—the organization can achieve faster response times, improve patient survival rates, and reduce operational costs, ensuring more effective use of public funds.

This project applies the PDCA (Plan-Do-Check-Act) framework to optimize public sector service requests, focusing on response times and costs. It uses Python for data generation, analysis, machine learning, and visualization to provide actionable insights for a Business Analyst.

## Repository Overview

- **`run_pdca_analysis.py`**: The main Python script that performs the entire analysis. **Run this file to generate all outputs.**
- **`service_requests.csv`**: Raw synthetic dataset of 500 service requests.
- **`service_requests.db`**: SQLite database containing the raw data.
- **`dashboard.png`**: A 2x2 dashboard summarizing key performance indicators (KPIs).
- **`pdca_report.pdf`**: A one-page summary of the PDCA process and key findings.
- **`recommendations.txt`**: Actionable recommendations based on the analysis.
- **`feature_importance.csv`**: The feature importance scores from the machine learning model.
- **Visualizations (`.png` files)**: Individual charts for response time, cost distribution, model performance, and feature importance.

## How to Run

1.  **Prerequisites**: Ensure you have Python and the following libraries installed:
    ```bash
    pip install pandas "scikit-learn==1.4.2" matplotlib
    ```
2.  **Execute the Script**: Open your terminal, navigate to this directory, and run:
    ```bash
    python run_pdca_analysis.py
    ```
    This single command will generate all the data-driven reports and visualizations.

## Machine Learning for Process Improvement

This project uses a **Gradient Boosting Regressor** to predict service request response times. This is significant for a Business Analyst because it:

1.  **Identifies Key Drivers**: By analyzing feature importance, we can pinpoint exactly which factors (e.g., department, request type) cause the most delays.
2.  **Enables Proactive Management**: The model can forecast response times for new requests, allowing for proactive resource allocation to prevent bottlenecks before they occur.
3.  **Provides Data-Driven Recommendations**: Instead of relying on intuition, the model's insights form the basis for targeted improvements, such as streamlining workflows in a specific department.

The model was optimized using `GridSearchCV` to ensure high performance, making its predictions reliable for strategic decision-making. By integrating these ML insights into the PDCA "Act" phase, we can drive measurable improvements, potentially achieving **15% efficiency gains** and estimated cost savings of **$13,125.00**.

## Replicating the Dashboard in Power BI / Tableau

The `dashboard.png` can be easily recreated in BI tools for interactive analysis.

1.  **Import Data**: Load `service_requests.csv` into your BI tool.
2.  **Create Visuals**:
    * **Bar Chart**: Average of `Response_Time_Hours` by `Department`.
    * **Pie Chart**: Sum of `Cost` by `Department`.
    * **Line Chart**: Average of `Response_Time_Hours` by `Date_Submitted` (monthly).
    * **Scatter Plot**: `Response_Time_Hours` vs. `Cost`.
3.  **Arrange**: Assemble the visuals in a 2x2 grid to match the dashboard.

## License
MIT License - see the `LICENSE` file for details.
