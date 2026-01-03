import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run_decision_tree_lab():
    # ---------------------------------------------------------
    # 1. Data Generation (Salaries Dataset)
    # ---------------------------------------------------------
    data = {
        'company': ['google', 'google', 'google', 'google', 'google', 'google',
                    'abc pharma', 'abc pharma', 'abc pharma', 'abc pharma',
                    'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook'],
        'job': ['sales executive', 'sales executive', 'business manager', 'business manager', 'computer programmer', 'computer programmer',
                'sales executive', 'computer programmer', 'business manager', 'business manager',
                'sales executive', 'sales executive', 'business manager', 'business manager', 'computer programmer', 'computer programmer'],
        'degree': ['bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters',
                   'masters', 'bachelors', 'bachelors', 'masters',
                   'bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters'],
        'salary_more_then_100k': [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    print("📊 Raw Data:")
    print(df.head())
    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Preprocessing (Label Encoding)
    # ---------------------------------------------------------
    # Decision Trees in sklearn usually require numerical input
    inputs = df.drop('salary_more_then_100k', axis='columns')
    target = df['salary_more_then_100k']

    le_company = LabelEncoder()
    le_job = LabelEncoder()
    le_degree = LabelEncoder()

    inputs['company_n'] = le_company.fit_transform(inputs['company'])
    inputs['job_n'] = le_job.fit_transform(inputs['job'])
    inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

    inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
    
    print("🔢 Encoded Inputs:")
    print(inputs_n.head())

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    # Criterion='gini' is default, but we can set 'entropy' explicitly
    model = tree.DecisionTreeClassifier(criterion='gini')
    model.fit(inputs_n, target)

    print("-" * 30)
    print(f"✅ Model Trained. Accuracy on Train Set: {model.score(inputs_n, target):.4f}")

    # ---------------------------------------------------------
    # 4. Prediction Logic
    # ---------------------------------------------------------
    # Example: Google (2), Business Manager (0), Bachelors (0)
    # Note: You'd need to check the mapping of LabelEncoder to be sure of indices
    prediction = model.predict([[2, 0, 0]])
    print(f"🔮 Prediction for [Google, Bus. Mgr, Bach]: {'>100k' if prediction[0]==1 else '<=100k'}")

    # ---------------------------------------------------------
    # 5. Tree Visualization (The "White Box" Proof)
    # ---------------------------------------------------------
    print("🎨 Generating Decision Tree Diagram...")
    plt.figure(figsize=(12, 8))
    tree.plot_tree(model, 
                   feature_names=['company', 'job', 'degree'], 
                   class_names=['<=100k', '>100k'], 
                   filled=True)
    plt.title("Decision Tree Logic Flow")
    plt.show()

if __name__ == "__main__":
    run_decision_tree_lab()
