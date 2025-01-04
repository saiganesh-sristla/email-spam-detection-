'''
Download dependencies
pip install plyer
pip install numpy
pip install pandas
pip install scikit-learn
pip install imapclient
pip install email
'''

#Machine Learning model 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv(r'C:\Users\saiga\OneDrive\Desktop\Machine-Learning\Spam detection Desktop notifier\mail_data.csv') 

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# checking the number of rows and columns in the dataframe
mail_data.shape

# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']

#Splitting the data into training data & test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

#Feature Extraction

# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)

print(X_train_features)

#Logistic Regression

model = LogisticRegression()

# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

#Evaluating the trained model
# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data : ', accuracy_on_training_data)

# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on test data : ', accuracy_on_test_data)

# Gmail connectivity

import imaplib
import email
from plyer import notification
import time

username = "your username"  
password = "your password"  
imap_host = 'imap.gmail.com'
imap_port = 993

# Connect to Gmail server
server = imaplib.IMAP4_SSL(imap_host, imap_port)
server.login(username, password)
server.select('inbox')

# Fetch the last 10 emails
type, data = server.search(None, 'ALL')
mail_ids = data[0]
id_list = mail_ids.split()  
latest_emails = id_list[-10:] 

for num in latest_emails:
    typ, data = server.fetch(num, '(RFC822)')
    raw_email = data[0][1]
    
    # Parse the email
    email_message = email.message_from_bytes(raw_email)

    # Extract details from the email
    email_subject = email_message['subject']
    email_from = email_message['from']
    print(f"From: {email_from}\n")
    print(f"Subject: {email_subject}\n")
    print("Body:")

    email_text = ""
    # Loop over the email parts to get the body text
    for part in email_message.walk():
        if part.get_content_type() == 'text/plain' and part.get('Content-Disposition') is None:
            email_text = part.get_payload(decode=True).decode('utf-8', errors='replace')  
            print(email_text)

    # Truncate email body if it's too long
    email_text = email_text[:64] + '...' if len(email_text) > 64 else email_text  

    # Prediction using the spam detector model
    input_mail = [email_text]
    input_data_features = feature_extraction.transform(input_mail)  
    prediction = model.predict(input_data_features)

    # Determine spam or ham
    if prediction[0] == 1:
        status = 'Ham mail'
    else:
        status = 'Spam mail'

    # Ensure that the notification message is not too long
    notification_message = f"{status}: {email_subject[:30]}"  
    
    # Truncate notification message if too long
    if len(notification_message) > 64:
        notification_message = notification_message[:64]  

    # Send notification
    notification.notify(title="Email Notification", message=notification_message, toast=False)
    time.sleep(5)  
