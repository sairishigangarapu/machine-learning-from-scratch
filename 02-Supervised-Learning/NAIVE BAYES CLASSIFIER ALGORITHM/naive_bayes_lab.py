import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def run_gaussian_nb_titanic():
    print("\n🔹 PART 1: Gaussian NB (Numeric Data - Titanic)")
    
    # 1. Load & Clean Data (Simulating a clean subset for demo)
    data = {
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male'],
        'Age': [22, 38, 26, 35, 35, None, 54, 2],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 51.86, 21.07],
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    # Preprocessing
    df['Age'] = df['Age'].fillna(df['Age'].mean()) # Handle Missing Values
    
    # One-Hot Encode 'Sex' manually for simplicity
    df['is_female'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
    
    X = df[['Pclass', 'is_female', 'Age', 'Fare']]
    y = df['Survived']
    
    # Train
    model = GaussianNB()
    model.fit(X, y)
    
    print(f"✅ Model Trained. Prediction for [3, male, 22, 7.25]: {model.predict([[3, 0, 22, 7.25]])[0]}")


def run_multinomial_nb_spam():
    print("\n🔹 PART 2: Multinomial NB (Text Data - Spam Filter)")
    
    # 1. Dataset
    data = {
        'Message': [
            "Win a $1000 Walmart gift card now!",
            "Meeting tomorrow at 10am?",
            "Free entry in 2 a wkly comp to win FA Cup final tkts",
            "Can we reschedule the call?",
            "URGENT! You have won a 1 week FREE membership"
        ],
        'Category': ['spam', 'ham', 'spam', 'ham', 'spam']
    }
    df = pd.DataFrame(data)
    
    # 2. Encode Labels
    df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.2, random_state=42)
    
    # 3. Create Pipeline (Vectorizer -> Classifier)
    # The Pipeline handles the "Bag of Words" conversion automatically
    clf = Pipeline([
        ('vectorizer', CountVectorizer()), # Convert text to numbers
        ('nb', MultinomialNB())            # Apply Naive Bayes
    ])
    
    # 4. Train
    clf.fit(X_train, y_train)
    
    # 5. Test
    new_emails = [
        "Hey, are we still on for lunch?",
        "YOU WON! Click here to claim your prize."
    ]
    preds = clf.predict(new_emails)
    
    print("📧 Testing New Emails:")
    for email, pred in zip(new_emails, preds):
        label = "SPAM 🚨" if pred == 1 else "Ham ✅"
        print(f" - '{email}' -> {label}")

if __name__ == "__main__":
    run_gaussian_nb_titanic()
    run_multinomial_nb_spam()
