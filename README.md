 Introduction
1.1   Overview
Heart disease is a serious health problem . Health details such as age, blood pressure, cholesterol and chest pain can help show possible heart problems.
HeartCare AI is a simple website that uses machine learning to predict a heart disease category from 13 health values. It compares four models: Logistic Regression, Decision Tree, Random Forest and KNN. The result is for learning only and is not a medical diagnosis.
1.2   Objectives
The main aims of this project are:
1.To use a public heart disease dataset.
2.To prepare the data and train four machine learning models.
3.To compare the models and choose one based on its results.
4.To build a simple website for entering 13 health values.
5.To show predictions and test if the website works.
1.3   Literature Review
Some studies have used machine learning to predict heart disease. Mohan et al. [3] used a combined method. Ali et al. [4] compared different models, and Hossain et al. [5] found good results with Random Forest in their study.
These studies show why comparing models is useful. However, results may change with a different dataset. HeartCare AI compares four common models and puts the selected model into a simple website so users can see how prediction works.
2   Methodology
The project follows a practical system-development method. First, the available dataset is studied and prepared. Several machine learning models are then trained and compared. After that, a selected model is connected to the website and the whole process is checked with test inputs. The method is suitable for this project because it allows both the machine learning results and the working application to be assessed.
2.1   Design
The system has two connected parts. The first is an offline training and evaluation process. The second is the website used to make predictions. Keeping these parts separate means that the model does not have to be trained each time someone submits the form. The saved scaler is applied to new inputs in the same way as the training data.

Figure 1. Main workflow of HeartCare AI, based on the supplied Python and Flask project files.
The dataset is tabular. Its 13 input fields are age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar category (fbs), resting ECG result (restecg), maximum heart rate (thalach), exercise-induced angina (exang), ST depression (oldpeak), slope of the ST segment (slope), number of major vessels (ca) and thalassemia-related test category (thal). The output variable, target, represents the class learned from the dataset. The UCI Heart Disease repository [2] documents the commonly used medical attribute set, but the supplied download script uses a separate GitHub-hosted CSV. The local copy of that CSV was not included with the submitted files, so its exact rows and coding should be verified before final submission.
The user interface has labels and short explanations beside the input fields. On submission, Flask converts the values into numbers, creates a 13-column input array, scales it and asks the model for a predicted class and class probability. The result is displayed on the same web page. The website includes a warning explaining that the prediction is educational and is not a medical diagnosis.
2.2   Implementation
2.2 Implementation
Python is used to develop this project. First, the create_dataset.py file downloads the heart disease dataset and saves it. The eda.py file helps us understand the data by creating graphs of age, heart disease cases, and the relationship between different health factors.
Next, the dataset is divided into two parts: 80% for training and 20% for testing. The data is prepared using StandardScaler so the machine learning models can work properly.
Four machine learning algorithms are used in this project: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). These models are trained and tested to find which one gives better results.
The evaluate_model.py file checks the performance of all four models using accuracy, precision, recall, F1-score, and ROC-AUC. The results are saved in model_results.csv. The train_model.py file selects the best model based on accuracy and saves it for future predictions. Both files use slightly different methods to divide the data.
Finally, Flask is used to connect the trained model with the website. The website allows users to enter their health information, such as age, blood pressure, cholesterol, and heart rate. After submitting the information, the system shows whether heart disease is predicted and displays the prediction probability. This system is developed for educational purposes and does not replace a doctor's diagnosis.
Table 1. Project files and their functions.
Project file	Main purpose
create_dataset.py	Downloads and stores the public heart dataset.
eda.py	Defines plots for target distribution, age and correlations.
train_model.py	Trains classifiers, selects by test accuracy and saves model and scaler.
evaluate_model.py	Compares four classifiers using five measures and outputs CSV results.
plot_confusion_matrix.py	Generates a confusion matrix for a separately trained Random Forest.
app.py	Loads the saved model, processes 13 form fields and returns a prediction.
index.html and CSS	Provide the user-facing form, result display and page styling.

2.3   Testing
Testing must check both prediction code and the website. A machine learning result is not useful if the form sends the wrong values, if a required file is missing or if users can enter unsupported categories. The following table describes the checks required before the final demonstration. These are planned system checks, not tests claimed to have passed.
Table 2. Test cases planned for final system checking.
Test case	Expected outcome
Open the home page	The form loads and all 13 input fields are visible.
Submit a valid sample	A predicted class and a numerical probability are returned.
Leave a required field blank	The browser prevents submission or the server gives a clear error.
Enter invalid numeric values	Out-of-range or non-numeric values are rejected safely.
Check category codes	Each dropdown value matches the category coding used during training.
Restart the Flask application	The correct saved model and scaler load from their expected folders.
Review results on a small screen	Input fields and results remain readable on mobile and desktop.
Check use of personal data	No private patient record is saved or shared by the prototype.

Special care is needed for the codes used in chest pain, slope, thal and other categorical fields. A label on the webpage must refer to the same numerical value as the training dataset. The program should also reject unexpected values on the server side, not rely only on HTML input controls. Debug mode should be switched off before a public demonstration or deployment. No clinical user testing or use with real patient records is claimed in this proposal.
2.4   Evaluation
The uploaded model_results.csv contains results for the four classifiers. Its evaluation script used a stratified 80:20 split. The table reports these recorded results as percentages, except ROC-AUC, which is shown on a 0 to 1 scale.

Table 3. Performance obtained from the supplied evaluation file.
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	80.33%	76.92%	90.91%	83.33%	0.869
Decision Tree	70.49%	70.27%	78.79%	74.29%	0.698
Random Forest	83.61%	78.05%	96.97%	86.49%	0.909
KNN	80.33%	76.92%	90.91%	83.33%	0.863

Random Forest gives the best results in this project, with 83.61% accuracy, 78.05% precision, 96.97% recall, 86.49% F1-score, and 0.909 ROC-AUC. Logistic Regression and KNN both achieved 80.33% accuracy, while Decision Tree achieved 70.49%. These results show that Random Forest performs better than the other three models on our test data.
Recall is important because it shows how well the model identifies people with heart disease. Precision shows how many positive predictions are correct. The results are good, but the system may still make mistakes. In the future, we can improve the system by testing it with more data and checking its accuracy. The prediction shown on the website is only an estimated result and should not be used as a final medical diagnosis.
3   Project Planning
3.1   Work Schedule Plan
This is a simple six-month plan, from October 2026 to March 2027. Each listed task takes one week. Other weeks are available for supervisor feedback and small changes.
Table 4. Six-month work schedule (one week per task)
Activity	Time	Duration
Read related papers	Oct 2026 - W1	1 week
Set project aims	Oct 2026 - W1	1 week
Get and check dataset	Oct 2026 - W1	1 week
Explore and clean data	Oct  2026 - W1	1 week
Prepare data for training	Oct  2026 - W1	1 week
Train four ML models	Oct  2026 - W1	1 week
Compare model results	Oct  2026 - W1	1 week
Save the selected model	Oct  2026 - W3	1 week
Create the website form	Oct  2026 - W1	1 week
Connect website and model	Oct 2026 - W2	1 week
Test and fix the system	Oct  2026 - W2	1 week
Finish report and slides	Oct  2026 - W2	1 week

4   Required Hardware and Software
4.1   Hardware
This project does not require expensive hardware. A normal laptop or computer is enough to develop and test the system.
• Laptop or Computer: For developing and running the project.
• RAM: At least 8 GB for smooth performance.
• Storage: Enough space to save the dataset and project files.
• Internet Connection: To download the dataset and required software.
4.2   Software
This project uses free and easy-to-use software tools for developing and testing the heart disease prediction system.
• Python: Used for programming and data processing.
• Pandas and NumPy: Used to manage and process the dataset.
• Scikit-learn: Used to train and test machine learning models.
• Matplotlib and Seaborn: Used to create graphs and charts.
• Flask: Used to connect the machine learning model with the website.
• HTML and Tailwind CSS: Used to design the website.
• Joblib: Used to save and load the trained model.
• Visual Studio Code: Used for writing and running the project code.
• Web Browser: Used to open and test the website.
5   Budget Description
This project uses free software and a public dataset. We can use a personal laptop or a university computer, so no special equipment is needed. Only small internet or printing costs may be required.
7   Description of Industrial Support (If Any)
No formal hospital or industrial partnership is documented in the uploaded project files. This prototype uses public data and free software; any later clinical testing would need institutional permission and suitable privacy safeguards.
8   References(This was written by Ai)
[1] World Health Organization, “Cardiovascular diseases (CVDs),” Fact sheet, Jul. 31, 2025. [Online]. Available: https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
[2] UCI Machine Learning Repository, “Heart Disease,” Dataset, doi: 10.24432/C52P4X. [Online]. Available: https://archive.ics.uci.edu/dataset/45/heart+disease
[3] S. Mohan, C. Thirumalai, and G. Srivastava, “Effective heart disease prediction using hybrid machine learning techniques,” IEEE Access, vol. 7, pp. 81542–81554, 2019, doi: 10.1109/ACCESS.2019.2923707.
[4] M. M. Ali, B. K. Paul, K. Ahmed, F. M. Bui, J. M. W. Quinn, and M. A. Moni, “Heart disease prediction using supervised machine learning algorithms: Performance analysis and comparison,” Computers in Biology and Medicine, vol. 136, Art. no. 104672, 2021, doi: 10.1016/j.compbiomed.2021.104672.
[5] S. Hossain, M. K. Hasan, M. O. Faruk, N. Aktar, R. Hossain, and K. Hossain, “Machine learning approach for predicting cardiovascular disease in Bangladesh: Evidence from a cross-sectional study in 2023,” BMC Cardiovascular Disorders, vol. 24, Art. no. 214, 2024, doi: 10.1186/s12872-024-03883-2.
[6] G. S. Collins et al., “TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ, vol. 385, Art. no. e078378, 2024, doi: 10.1136/bmj-2023-078378.
[7] R. F. Wolff et al., “PROBAST: A tool to assess the risk of bias and applicability of prediction model studies,” Annals of Internal Medicine, vol. 170, no. 1, pp. 51–58, 2019, doi: 10.7326/M18-1376.
[8] United Nations, “Goal 3: Ensure healthy lives and promote well-being for all at all ages,” Sustainable Development Goals. [Online]. Available: https://sdgs.un.org/goals/goal3
[9] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html
[10] GitHub user “amankharwal,” “heart.csv,” hosted CSV file referenced by the supplied create_dataset.py script. [Online]. Available: https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv (Original dataset provenance to be checked before f
