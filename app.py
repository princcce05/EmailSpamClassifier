import streamlit as st
import joblib

# Load model and vectorizer using joblib
model = joblib.load('spam_classifier_model.pkl')  # or 'spam_classifier_model.pkl' if you didn't rename
vectorizer = joblib.load('vectorizer.pkl')

st.title('📧 Email Spam Classifier')

# Input from user
email_text = st.text_area('Enter Email Text')

if st.button('Predict'):
    # Step 1: Vectorize user input
    input_data = vectorizer.transform([email_text])  # ✅ this is now 2D and correct

    # Step 2: Predict
    prediction = model.predict(input_data)

    # Step 3: Show result
    result = 'Spam ❌' if prediction[0] == 0 else 'Not Spam ✅'
    st.success(f'This email is: **{result}**')


st.markdown("---")
st.subheader("📈 Model Info")
st.write("Training Accuracy: 96.76912721561588%")

