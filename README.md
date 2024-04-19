# Developing machine learning models for lead scoring in customer relationship management
---
## Introduction
- This project aims to develop a machine learning model to automate the process of scoring potential customers (lead scoring) in Customer Relationship Management (CRM). The primary goal is to provide a tool that helps businesses better understand the potential of each customer, thereby optimizing outreach strategies and enhancing sales effectiveness.

<p align="center">
  <img src="image/predictive_lead_scoring_flow-1.png" />
</p>

## Using the Model
- This model allows businesses to automate the scoring of lead customers based on various criteria including online behavior, and interactions with campaigns. Below is a flow illustrating how the model operates:

<p align="center">
  <img src="image/Flowchart and Database - Lead Scoring Model-Flowchart_Lead_score.drawio.png" />
</p>


## Results
- The model has achieved notable results in classifying and predicting potential customers with a high likelihood of conversion. Below are some charts and tables demonstrating the effectiveness of the model:

\begin{table}[h]
\centering
\caption{Kết quả mô hình sau khi tối ưu hóa tham số}
\label{tab:model_results}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{SHAP} & \multicolumn{3}{c|}{\textbf{Train}} & \multicolumn{3}{c|}{\textbf{Test}} \\ \hline
              & \textbf{Accuracy} & \textbf{F1-score} & \textbf{Gini} & \textbf{Accuracy} & \textbf{F1-score} & \textbf{Gini} \\ \hline
\textbf{CatBoost} & & & & & & \\
0             & 0.8388          & 0.870             & 0.830         & 0.8371           & 0.866             & 0.816         \\
1             & 0.787           & 0.787             &               & 0.793            &                   &               \\ \hline
\textbf{LightGBM} & & & & & & \\
0             & 0.8347          & 0.867             & 0.818         & 0.8355           & 0.865             & 0.817         \\
1             &                 & 0.781             &               &                  & 0.789             &               \\ \hline
\end{tabular}
\end{table}

- This model helps businesses identify and focus resources on potential customers with high conversion potential, thereby optimizing sales and marketing strategies.

