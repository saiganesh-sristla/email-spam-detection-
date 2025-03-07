import random

ham_messages = [
    "Hey, are we still on for tonight?",
    "Meeting rescheduled to 2 PM. Let me know if that works.",
    "The client presentation is tomorrow. Be prepared!",
    "Mom said to call her when you’re free.",
    "Reminder: Doctor’s appointment at 4 PM.",
    "Lunch at our usual spot?",
    "Can you email me the latest report?",
    "Flight is delayed by an hour. Keep me updated.",
    "Great job on the project! Let’s discuss the next steps.",
    "Dad’s birthday is coming up. Any gift ideas?",
    "Let’s grab coffee next week.",
    "Your subscription has been renewed successfully.",
    "Invoice #4523 is attached. Let me know if you have questions.",
]

spam_messages = [
    "CLAIM YOUR $1000 NOW! Click here: http://bit.ly/winbig",
    "V1AGRA, C1ALIS for 80% OFF! Limited time deal: http://cheapmeds.com",
    "¡Gana $1000 ahora! Haz clic aquí: http://ganadinero.com",
    "Recieve FREE gift card! Sign up: http://freetreats.net",
    "Your Netflix account is locked! Verify now: http://netflix-verify.com",
    "Work from home & earn $5000/week! No experience needed!",
    "Your package delivery failed! Reschedule here: http://delivery-alert.com",
    "Your invoice #INV789: http://track-order.com",
    "Hot singles in your area! Chat now: http://findlove.com",
    "Congratulations! You've been selected for an exclusive deal!",
    "Suspicious login detected! Secure your account: http://secureme.net",
    "Your refund is available! Claim now: http://tax-return.com",
    "Earn money online fast! No skills needed! http://quickcash.com",
]

# Generate dataset
data = []
for _ in range(2500):  # Half ham, half spam
    data.append(("ham", random.choice(ham_messages)))
    data.append(("spam", random.choice(spam_messages)))

# Shuffle to randomize order
random.shuffle(data)

# Save to CSV
import pandas as pd
df = pd.DataFrame(data, columns=["Category", "Message"])
df.to_csv("spam_dataset.csv", index=False)

print("Generated dataset saved as 'spam_dataset.csv'")
