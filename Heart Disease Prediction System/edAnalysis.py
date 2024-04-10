# Importing Libraries

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm


# 2. Data Cleaning and Preprocessing
file_path = 'heart.csv'
data = pd.read_csv(file_path)
print(data.head())
print(data.describe())
print(data.info())
print(data.columns)

# print(data.isnull().sum())  # Checking for NULL and duplicate values, and removing them
# print(data.duplicated().sum())
# data.drop_duplicates(inplace=True)
# print(data.duplicated().sum())

# 3. Basic Measures and Averages

# Let's separate the genders as we want to see how men compare to women
male_data = data[data['sex'] == 1]
female_data = data[data['sex'] == 0]

# Count of Males & Females in Dataset
male_count = data['sex'].value_counts()[1]
female_count = data['sex'].value_counts()[0]

# Avg Age of Males & Females in Dataset
male_age_avg = male_data['age'].mean()
female_age_avg = female_data['age'].mean()

oa_chol = data['chol'].mean()  # Avg Cholesterol Level of People in the Dataset

# Number of Positive Cases (Who have Heart Disease)
fbs_count = data['fbs'].value_counts()[1]
trg_count = data['target'].value_counts()[1]

# Creating tabular data for above metrics
table_data = {
    'Basic Measures': ['Number of male participants', 'Number of female participants'],
    'Values': [f'{male_count} men', f'{female_count} women']
}
basic_measures_df = pd.DataFrame(table_data)

basic_measures_df = basic_measures_df._append(
    {'Basic Measures': 'Average age of male participants', 'Values': f'{male_age_avg:.0f} years'},
    ignore_index=True
)
basic_measures_df = basic_measures_df._append(
    {'Basic Measures': 'Average age of female participants', 'Values': f'{female_age_avg:.0f} years'},
    ignore_index=True
)
basic_measures_df = basic_measures_df._append(
    {'Basic Measures': 'Average cholesterol level', 'Values': f'{oa_chol:.0f} mg/dL'},
    ignore_index=True
)
basic_measures_df = basic_measures_df._append(
    {'Basic Measures': 'Number with fasting blood sugar', 'Values': f'{fbs_count} cases'},
    ignore_index=True
)
basic_measures_df = basic_measures_df._append(
    {'Basic Measures': 'Number with heart disease', 'Values': f'{trg_count} cases'},
    ignore_index=True
)

# Display the DataFrame
print(basic_measures_df)
basic_measures_df = pd.DataFrame(table_data)


# Define the features of interest
features = ['age', 'chol', 'trestbps']

# Calculating means, standard deviations, and variances for male
male_stats = {}
for feature in features:
    mean_male = male_data[feature].mean()
    std_dev_male = male_data[feature].std()
    variance_male = male_data[feature].var()
    male_stats[feature] = {'Mean': mean_male, 'Std Dev': std_dev_male, 'Variance': variance_male}

# Calculating means, standard deviations, and variances for females
female_stats = {}
for feature in features:
    mean_female = female_data[feature].mean()
    std_dev_female = female_data[feature].std()
    variance_female = female_data[feature].var()
    female_stats[feature] = {'Mean': mean_female, 'Std Dev': std_dev_female, 'Variance': variance_female}

# Display the results
for feature in features:
    print(f"\nStatistics for {feature}:")
    print(f"  Male - Mean: {male_stats[feature]['Mean']:.2f}, Std Dev: {male_stats[feature]['Std Dev']:.2f}, Variance: {male_stats[feature]['Variance']:.2f}")
    print(f"  Female - Mean: {female_stats[feature]['Mean']:.2f}, Std Dev: {female_stats[feature]['Std Dev']:.2f}, Variance: {female_stats[feature]['Variance']:.2f}")


# 4. Analysis and EDA

# 4.1. What age group is most vulnerable or has a large number of patients with a higher risk of heart attack?
print(data['target'].value_counts(normalize=True))
sns.countplot(x='target', data=data)
plt.xticks([0, 1], ['less chance', 'more chance'])
plt.title('Chances of Heart Disease')
plt.figure(figsize=(15, 6))

data['age'].hist(bins=20)
plt.title('Number of People Having Heart Disease Age Wise')
plt.xlabel('Age')
plt.ylabel('No. of Persons')
plt.show()
# Adding a probability distribution curve to the age histogram
sns.histplot(data['age'], bins=20, kde=True, color='blue', stat='density')
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.title('Age Distribution with Probability Density Curve of Number of People Having Heart Disease')
plt.show()

# 4.2. Are men mostly prone to heart attacks or women?
sns.countplot(x='sex', data=data)
plt.title('Number of Males and Females')
plt.xticks([0, 1], ['Females', 'Males'])
plt.show()

sns.countplot(x='sex', data=data, hue='target')
plt.title('Chances of Heart Disease Gender Wise')
plt.xticks([0, 1], ['Females', 'Males'])
plt.legend(labels=['Less Chance', 'High Chance'])
plt.show()

# 4.3. What chest pain types pose a severe risk of a heart attack?
data['cp'].value_counts().plot(kind='bar')
plt.xticks([0, 1, 2, 3], ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"], rotation=0)
plt.xlabel('Chest Pain Type')
plt.ylabel('No. of Persons')
plt.show()
sns.countplot(x='cp', hue='target', data=data)  # formatting the plot for a better view
plt.title('Relation between types of chest pain and number of people having high or low chances of heart attack')
plt.xticks([0, 1, 2, 3], ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
plt.legend(labels=['low chance', 'high chance'])
plt.show()

# 4.4. How fasting blood sugar is related to heart attack?
sns.countplot(x='fbs', hue='target', data=data)
plt.legend(labels=['low chance', 'high chance'])
plt.title('Relationship Between Fasting Blood Sugar and Heart Attack')
plt.show()

# 4.5. Due to cholesterol, how many patients are at higher risk?
data['chol'].hist()
plt.xlabel('Serum cholestoral (mg/dl)')
plt.ylabel('No. of Person')
plt.title('Distribution of Serum Cholesterol Levels in Patients')
plt.show()

# 4.6. Due to cholesterol, how many patients are at higher risk?
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.hist(data['chol'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Serum Cholesterol Levels in Patients')
plt.xlabel('Serum Cholesterol (mg/dl)')
plt.ylabel('Number of Persons')
plt.show()
# Adding probability curve
sns.histplot(data['chol'], kde=True, color='orange', stat='density')
plt.xlabel('Serum Cholestoral (mg/dl)')
plt.ylabel('Probability Density')
plt.title('Cholesterol Distribution with Probability Density Curve')
plt.show()

# 4.7. EDA on Cholesterol Levels vs Risk of Heart Disease
# Now for a visual on some key health metrics.

male_health_data = male_data[['age', 'sex', 'chol', 'target']]
female_health_data = female_data[['age', 'sex', 'chol', 'target']]
female_health_data.sort_values(by='chol', ascending=False)
male_health_data.sort_values(by='chol', ascending=False)
combined_chol_data = pd.concat([male_health_data, female_health_data],
                               ignore_index=True)  # Concatenating the two DataFrames along the rows vertically
combined_chol_data.reset_index(drop=True, inplace=True)
print(combined_chol_data)

# Creating a pair plot for Cholesterol Comparisons
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Histogram 1
sns.histplot(data=combined_chol_data, x="chol", hue="sex", multiple="stack", ax=ax1, palette='YlGnBu')
ax1.set_title('Cholesterol vs Participant Gender')
ax1.set_xlabel('Cholesterol Value  mg/dL')
ax1.set_ylabel('Number of Participants')
legend1 = ax1.legend(title="Sex", labels=["Male", "Female"])
# Histogram 2
sns.histplot(data=combined_chol_data, x="chol", hue="target", multiple="stack", ax=ax2, palette='YlGnBu')
ax2.set_title('Cholesterol vs Heart Disease Diagnosis')
ax2.set_xlabel('Cholesterol Value  mg/dL')
ax2.set_ylabel('Number of Participants')
legend2 = ax2.legend(title="Heart Disease", labels=["No", "Yes"])
plt.tight_layout()
plt.show()



# 5. Additional Analysis

# 5.1. What percentage of participants actually have heart disease?
heart_percent = data["target"].value_counts(normalize=True).to_dict()
heart_percent["Heart Disease"] = heart_percent.pop(1)
heart_percent["Normal"] = heart_percent.pop(0)

plt.figure(figsize=(4, 2), dpi=200)
labels = heart_percent.keys()
sizes = heart_percent.values()
colors = ["#c1d8c1", "#6283af"]
explode = (0, 0.1)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, colors=colors, startangle=90,
        textprops={'fontsize': 5})
plt.title("Heart Disease Percentage", size=6, fontweight="bold")
plt.show()

# 5.2. Correlation Heatmap
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1, center=0,
            cbar_kws={"shrink": 0.75})
plt.title('Correlation Heatmap of Updated Features')
plt.show()

# 5.3 Fitting a Normal Distribution to the 'age' variable
mu, std = norm.fit(data['age'])
plt.hist(data['age'], bins=20, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title('Fitting a Normal Distribution to the age variable with Fit results: mu = %.2f,  std = %.2f' % (mu, std))
plt.show()

# Setting the confidence level
confidence_level = 0.95

# Calculating confidence interval for mean
def calculate_ci_for_mean(data, confidence_level):
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

# Calculating confidence interval for count (proportion)
def calculate_ci_for_count(count, total, confidence_level):
    p = count / total
    margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * np.sqrt((p * (1 - p)) / total)
    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error
    return lower_bound, upper_bound

# Displaying basic measures and averages with confidence intervals
for index, row in basic_measures_df.iterrows():
    basic_measure = row['Basic Measures']
    value = row['Values']

    if 'Number' in basic_measure or 'Average' in basic_measure:
        if 'Number' in basic_measure:
            count_data = data.shape[0]
            count_ci = calculate_ci_for_count(int(value.split()[0]), count_data, confidence_level)
            print(f"{basic_measure}: {value} (Confidence Interval: {count_ci})")
        else:
            column_name = basic_measure.split()[-2].lower()
            avg_data = data[column_name]
            avg_ci = calculate_ci_for_mean(avg_data, confidence_level)
            print(f"{basic_measure}: {value} (Confidence Interval: {avg_ci})")

# Plotting confidence intervals for age
sns.histplot(data['age'], bins=20, kde=True, color='blue', stat='density')
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.title('Age Distribution with Probability Density Curve of Number of People Having Heart Disease w. CI')

# Adding confidence interval lines for mean age
mean_age_ci = calculate_ci_for_mean(data['age'], confidence_level)
plt.axvline(mean_age_ci[0], color='red', linestyle='dashed', linewidth=2, label='Mean CI (95%)')
plt.axvline(mean_age_ci[1], color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.show()

# Calculating confidence interval for cholesterol
cholesterol_ci = calculate_ci_for_mean(data['chol'], confidence_level)
print(f"Average cholesterol level: {oa_chol:.0f} mg/dL (Confidence Interval: {cholesterol_ci})")

# Plotting confidence intervals for cholesterol
plt.figure(figsize=(10, 6))
plt.hist(data['chol'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Serum Cholesterol Levels in Patients with Confidence Interval')
plt.xlabel('Serum Cholesterol (mg/dl)')
plt.ylabel('Number of Persons')

# Adding confidence interval lines for mean cholesterol
plt.axvline(cholesterol_ci[0], color='red', linestyle='dashed', linewidth=2, label='Mean CI (95%)')
plt.axvline(cholesterol_ci[1], color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.show()

# Adding probability curve with confidence interval for cholesterol
sns.histplot(data['chol'], kde=True, color='orange', stat='density')
plt.xlabel('Serum Cholestoral (mg/dl)')
plt.ylabel('Probability Density')
plt.title('Cholesterol Distribution with Probability Density Curve and Confidence Interval')
plt.axvline(cholesterol_ci[0], color='red', linestyle='dashed', linewidth=2, label='Mean CI (95%)')
plt.axvline(cholesterol_ci[1], color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.show()

# Calculating confidence interval for Male Age
age_ci = calculate_ci_for_mean(data['age'], confidence_level)
print(f"Average age of Male participants: {male_age_avg:.0f} years (Confidence Interval: {age_ci})")

# Plotting confidence intervals for age
plt.figure(figsize=(10, 6))
plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Male Age in Patients with Confidence Interval')
plt.xlabel('Male Age')
plt.ylabel('Number of Persons')

# Adding confidence interval lines for mean age
plt.axvline(age_ci[0], color='red', linestyle='dashed', linewidth=2, label='Mean CI (95%)')
plt.axvline(age_ci[1], color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.show()

# Adding probability curve with confidence interval for age
sns.histplot(data['age'], kde=True, color='orange', stat='density')
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.title('Age Distribution with Probability Density Curve and Confidence Interval')
plt.axvline(age_ci[0], color='red', linestyle='dashed', linewidth=2, label='Mean CI (95%)')
plt.axvline(age_ci[1], color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.show()
