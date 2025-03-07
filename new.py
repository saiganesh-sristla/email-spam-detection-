import numpy as np
import pandas as pd
import pickle
import os
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
import imaplib
import email
from email.header import decode_header
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

class SpamFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Email Spam Detection System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.model = None
        self.vectorizer = None
        self.is_monitoring = False
        self.monitoring_thread = None
        self.email_data = []
        self.username = "your username"
        self.password = "your password"
        self.imap_host = "imap.gmail.com"
        self.imap_port = 993
        self.setup_ui()
        
    def setup_ui(self):
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.monitoring_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Setup")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.monitoring_tab, text="Email Monitoring")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Set up each tab
        self.setup_setup_tab()
        self.setup_training_tab()
        self.setup_monitoring_tab()
        self.setup_analysis_tab()
        self.setup_settings_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_setup_tab(self):
        frame = ttk.LabelFrame(self.setup_tab, text="Email Account Setup")
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ttk.Label(frame, text="Gmail Username:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.username_entry = ttk.Entry(frame, width=40)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(frame, text="App Password:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.password_entry = ttk.Entry(frame, width=40, show="*")
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(frame, text="Note: Use App Password generated in your Google Account").grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        help_btn = ttk.Button(frame, text="How to get App Password?", command=self.open_app_password_help)
        help_btn.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        
        test_conn_btn = ttk.Button(frame, text="Test Connection", command=self.test_connection)
        test_conn_btn.grid(row=3, column=1, padx=10, pady=10)
        
        # Instructions frame
        instr_frame = ttk.LabelFrame(self.setup_tab, text="Instructions")
        instr_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        instructions = """
        Welcome to the Advanced Email Spam Detection System!

        Getting Started:
        1. Enter your Gmail username and app password
        2. Load your dataset or use the default one
        3. Train your model with desired parameters
        4. Start monitoring your emails for spam

        Features:
        - Real-time email monitoring
        - Multiple machine learning models
        - Interactive data visualization
        - Customizable notification settings
        - Email statistics and analysis
        """
        
        instr_text = scrolledtext.ScrolledText(instr_frame, wrap=tk.WORD, width=70, height=10)
        instr_text.pack(padx=10, pady=10, fill="both", expand=True)
        instr_text.insert(tk.END, instructions)
        instr_text.configure(state="disabled")
        
    def setup_training_tab(self):
        # Dataset frame
        dataset_frame = ttk.LabelFrame(self.training_tab, text="Dataset")
        dataset_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        ttk.Label(dataset_frame, text="Dataset Path:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.dataset_path = ttk.Entry(dataset_frame, width=50)
        self.dataset_path.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        self.dataset_path.insert(0, "mail_data.csv")
        
        browse_btn = ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset)
        browse_btn.grid(row=0, column=2, padx=10, pady=10)
        
        load_btn = ttk.Button(dataset_frame, text="Load Dataset", command=self.load_dataset)
        load_btn.grid(row=1, column=0, padx=10, pady=10)
        
        self.dataset_info = ttk.Label(dataset_frame, text="Dataset not loaded")
        self.dataset_info.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky=tk.W)
        
        # Model configuration frame
        model_frame = ttk.LabelFrame(self.training_tab, text="Model Configuration")
        model_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        ttk.Label(model_frame, text="Vectorizer:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.vectorizer_var = tk.StringVar(value="TF-IDF")
        vectorizer_combo = ttk.Combobox(model_frame, textvariable=self.vectorizer_var, 
                                        values=["TF-IDF", "Count Vectorizer"], state="readonly", width=15)
        vectorizer_combo.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        
        ttk.Label(model_frame, text="Algorithm:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.algorithm_var = tk.StringVar(value="Logistic Regression")
        algorithm_combo = ttk.Combobox(model_frame, textvariable=self.algorithm_var, 
                                     values=["Logistic Regression", "Random Forest", "SVM"], state="readonly", width=15)
        algorithm_combo.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        ttk.Label(model_frame, text="Test Size:").grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)
        self.test_size_var = tk.StringVar(value="0.2")
        test_size_combo = ttk.Combobox(model_frame, textvariable=self.test_size_var, 
                                     values=["0.1", "0.2", "0.3"], state="readonly", width=15)
        test_size_combo.grid(row=0, column=3, padx=10, pady=10, sticky=tk.W)
        
        # Training buttons
        train_btn = ttk.Button(model_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, padx=10, pady=10)
        
        save_model_btn = ttk.Button(model_frame, text="Save Model", command=self.save_model)
        save_model_btn.grid(row=2, column=1, padx=10, pady=10)
        
        load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        load_model_btn.grid(row=2, column=2, padx=10, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.training_tab, text="Training Results")
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=70, height=10)
        self.results_text.pack(padx=10, pady=10, fill="both", expand=True)
        
    def setup_monitoring_tab(self):
        # Control frame
        control_frame = ttk.Frame(self.monitoring_tab)
        control_frame.pack(fill="x", padx=20, pady=10)
        
        self.monitor_btn = ttk.Button(control_frame, text="Start Monitoring", command=self.toggle_monitoring)
        self.monitor_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Refresh Interval (seconds):").pack(side=tk.LEFT, padx=5, pady=5)
        self.refresh_var = tk.StringVar(value="60")
        refresh_entry = ttk.Entry(control_frame, textvariable=self.refresh_var, width=5)
        refresh_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        clear_btn = ttk.Button(control_frame, text="Clear List", command=self.clear_email_list)
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Email list frame
        email_frame = ttk.LabelFrame(self.monitoring_tab, text="Email Monitoring")
        email_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Treeview for emails
        columns = ("Status", "From", "Subject", "Date", "Confidence")
        self.email_tree = ttk.Treeview(email_frame, columns=columns, show="headings")
        
        # Set column headings
        self.email_tree.heading("Status", text="Status")
        self.email_tree.heading("From", text="From")
        self.email_tree.heading("Subject", text="Subject")
        self.email_tree.heading("Date", text="Date")
        self.email_tree.heading("Confidence", text="Confidence")
        
        # Set column widths
        self.email_tree.column("Status", width=100)
        self.email_tree.column("From", width=200)
        self.email_tree.column("Subject", width=300)
        self.email_tree.column("Date", width=150)
        self.email_tree.column("Confidence", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(email_frame, orient=tk.VERTICAL, command=self.email_tree.yview)
        self.email_tree.configure(yscroll=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.email_tree.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        
        # Bind double-click event to view email details
        self.email_tree.bind("<Double-1>", self.view_email_details)
        
        # Email preview frame
        preview_frame = ttk.LabelFrame(self.monitoring_tab, text="Email Preview")
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, width=70, height=10)
        self.preview_text.pack(padx=10, pady=10, fill="both", expand=True)
        
    def setup_analysis_tab(self):
        # Controls frame
        controls_frame = ttk.Frame(self.analysis_tab)
        controls_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(controls_frame, text="Analysis Type:").pack(side=tk.LEFT, padx=5, pady=5)
        self.analysis_var = tk.StringVar(value="Spam vs Ham Distribution")
        analysis_combo = ttk.Combobox(controls_frame, textvariable=self.analysis_var, 
                                    values=["Spam vs Ham Distribution", "Detection Accuracy Over Time", 
                                            "Top Spam Keywords", "Email Volume by Hour"], state="readonly", width=25)
        analysis_combo.pack(side=tk.LEFT, padx=5, pady=5)
        
        update_btn = ttk.Button(controls_frame, text="Update Analysis", command=self.update_analysis)
        update_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Split the analysis tab into two parts
        paned = ttk.PanedWindow(self.analysis_tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Graph frame
        graph_frame = ttk.LabelFrame(paned, text="Visualization")
        paned.add(graph_frame, weight=2)
        
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(paned, text="Statistics")
        paned.add(stats_frame, weight=1)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, width=30, height=15)
        self.stats_text.pack(padx=10, pady=10, fill="both", expand=True)
        
    def setup_settings_tab(self):
        # Notification settings
        notif_frame = ttk.LabelFrame(self.settings_tab, text="Notification Settings")
        notif_frame.pack(fill="x", padx=20, pady=10)
        
        self.enable_notif_var = tk.BooleanVar(value=True)
        enable_notif = ttk.Checkbutton(notif_frame, text="Enable Desktop Notifications", variable=self.enable_notif_var)
        enable_notif.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        
        self.notif_spam_only_var = tk.BooleanVar(value=True)
        notif_spam_only = ttk.Checkbutton(notif_frame, text="Notify for Spam Emails Only", variable=self.notif_spam_only_var)
        notif_spam_only.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        
        ttk.Label(notif_frame, text="Notification Duration (seconds):").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.notif_duration_var = tk.StringVar(value="5")
        notif_duration = ttk.Entry(notif_frame, textvariable=self.notif_duration_var, width=5)
        notif_duration.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        # Filter settings
        filter_frame = ttk.LabelFrame(self.settings_tab, text="Filter Settings")
        filter_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(filter_frame, text="Spam Confidence Threshold:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.spam_threshold_var = tk.StringVar(value="0.5")
        spam_threshold = ttk.Scale(filter_frame, from_=0, to=1, orient="horizontal", 
                              variable=self.spam_threshold_var, length=200)
        spam_threshold.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        ttk.Label(filter_frame, textvariable=self.spam_threshold_var).grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)
        
        # Action settings
        action_frame = ttk.LabelFrame(self.settings_tab, text="Action Settings")
        action_frame.pack(fill="x", padx=20, pady=10)
        
        self.auto_move_spam_var = tk.BooleanVar(value=False)
        auto_move_spam = ttk.Checkbutton(action_frame, text="Automatically Move Spam to Junk Folder", 
                                      variable=self.auto_move_spam_var)
        auto_move_spam.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky=tk.W)
        
        self.mark_spam_read_var = tk.BooleanVar(value=False)
        mark_spam_read = ttk.Checkbutton(action_frame, text="Mark Spam as Read", variable=self.mark_spam_read_var)
        mark_spam_read.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        
        # Save settings button
        save_settings_btn = ttk.Button(self.settings_tab, text="Save Settings", command=self.save_settings)
        save_settings_btn.pack(padx=20, pady=20)
        
    # Functionality methods
    def open_app_password_help(self):
        webbrowser.open("https://support.google.com/accounts/answer/185833")
    
    def test_connection(self):
        self.username = self.username_entry.get()
        self.password = self.password_entry.get()
        
        if not self.username or not self.password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        try:
            server = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
            server.login(self.username, self.password)
            server.select('inbox')
            messagebox.showinfo("Success", "Connection successful!")
            server.logout()
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
    
    def browse_dataset(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.dataset_path.delete(0, tk.END)
            self.dataset_path.insert(0, filename)
    
    def load_dataset(self):
        path = self.dataset_path.get()
        if not path:
            messagebox.showerror("Error", "Please specify dataset path")
            return
            
        try:
            self.status_var.set("Loading dataset...")
            self.raw_mail_data = pd.read_csv(path)
            self.mail_data = self.raw_mail_data.where((pd.notnull(self.raw_mail_data)), '')
            
            # Display dataset info
            spam_count = len(self.mail_data[self.mail_data['Category'] == 'spam'])
            ham_count = len(self.mail_data[self.mail_data['Category'] == 'ham'])
            self.dataset_info.config(text=f"Loaded {len(self.mail_data)} emails: {spam_count} spam, {ham_count} ham")
            
            self.status_var.set("Dataset loaded successfully")
            
            # Update analysis tab
            self.update_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.status_var.set("Error loading dataset")
    
    def train_model(self):
        if not hasattr(self, 'mail_data'):
            messagebox.showerror("Error", "Please load dataset first")
            return
            
        try:
            self.status_var.set("Training model...")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training model...\n\n")
            
            # Label encoding
            self.mail_data.loc[self.mail_data['Category'] == 'spam', 'Category'] = 0
            self.mail_data.loc[self.mail_data['Category'] == 'ham', 'Category'] = 1
            
            # Feature extraction
            X = self.mail_data['Message']
            Y = self.mail_data['Category'].astype('int')
            
            # Train-test split
            test_size = float(self.test_size_var.get())
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=3)
            
            # Feature extraction setup
            if self.vectorizer_var.get() == "TF-IDF":
                self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
            else:
                self.vectorizer = CountVectorizer(min_df=1, stop_words='english', lowercase=True)
                
            X_train_features = self.vectorizer.fit_transform(X_train)
            X_test_features = self.vectorizer.transform(X_test)
            
            # Model selection and training
            if self.algorithm_var.get() == "Logistic Regression":
                self.model = LogisticRegression(max_iter=1000)
            elif self.algorithm_var.get() == "Random Forest":
                self.model = RandomForestClassifier(n_estimators=100)
            else:  # SVM
                self.model = SVC(probability=True)
                
            self.model.fit(X_train_features, Y_train)
            
            # Model evaluation
            train_prediction = self.model.predict(X_train_features)
            test_prediction = self.model.predict(X_test_features)
            
            train_accuracy = accuracy_score(Y_train, train_prediction)
            test_accuracy = accuracy_score(Y_test, test_prediction)
            
            precision = precision_score(Y_test, test_prediction)
            recall = recall_score(Y_test, test_prediction)
            f1 = f1_score(Y_test, test_prediction)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_features, Y_train, cv=5)
            
            # Display results
            self.results_text.insert(tk.END, f"Model: {self.algorithm_var.get()}\n")
            self.results_text.insert(tk.END, f"Vectorizer: {self.vectorizer_var.get()}\n\n")
            self.results_text.insert(tk.END, f"Training Accuracy: {train_accuracy:.4f}\n")
            self.results_text.insert(tk.END, f"Testing Accuracy: {test_accuracy:.4f}\n\n")
            self.results_text.insert(tk.END, f"Precision: {precision:.4f}\n")
            self.results_text.insert(tk.END, f"Recall: {recall:.4f}\n")
            self.results_text.insert(tk.END, f"F1 Score: {f1:.4f}\n\n")
            self.results_text.insert(tk.END, f"Cross-Validation Scores: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})\n")
            
            # Confusion matrix
            cm = confusion_matrix(Y_test, test_prediction)
            self.results_text.insert(tk.END, f"\nConfusion Matrix:\n")
            self.results_text.insert(tk.END, f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
            self.results_text.insert(tk.END, f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
            
            self.status_var.set("Model trained successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Error training model")
    
    def save_model(self):
        if not self.model or not self.vectorizer:
            messagebox.showerror("Error", "No trained model to save")
            return
            
        try:
            # Create models directory if it doesn't exist
            if not os.path.exists("models"):
                os.makedirs("models")
                
            # Save model and vectorizer
            with open("models/spam_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
                
            with open("models/vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
                
            messagebox.showinfo("Success", "Model saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        try:
            # Load model and vectorizer
            with open("models/spam_model.pkl", "rb") as f:
                self.model = pickle.load(f)
                
            with open("models/vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
                
            messagebox.showinfo("Success", "Model loaded successfully")
            self.status_var.set("Model loaded successfully")
            
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found. Train a model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def toggle_monitoring(self):
        if not self.model or not self.vectorizer:
            messagebox.showerror("Error", "Please train or load a model first")
            return
            
        if self.username == "" or self.password == "":
            messagebox.showerror("Error", "Please set up email credentials first")
            return
            
        if self.is_monitoring:
            # Stop monitoring
            self.is_monitoring = False
            self.monitor_btn.config(text="Start Monitoring")
            self.status_var.set("Email monitoring stopped")
        else:
            # Start monitoring
            self.is_monitoring = True
            self.monitor_btn.config(text="Stop Monitoring")
            self.status_var.set("Email monitoring started")
            
            # Start monitoring in a separate thread
            self.monitoring_thread = threading.Thread(target=self.monitor_emails)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def monitor_emails(self):
        while self.is_monitoring:
            try:
                # Connect to Gmail
                server = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
                server.login(self.username, self.password)
                server.select('inbox')
                
                # Search for unseen emails
                status, messages = server.search(None, 'UNSEEN')
                
                if messages[0]:
                    email_ids = messages[0].split()
                    
                    # Process the latest 10 emails only
                    for email_id in email_ids[-10:]:
                        self.process_email(server, email_id)
                        
                server.logout()
                
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
                self.status_var.set(f"Monitoring error: {str(e)}")
                time.sleep(5)
                
            # Wait for the specified refresh interval
            time.sleep(int(self.refresh_var.get()))
            
            if not self.is_monitoring:
                break
    
   
            # Predict if spam
    def process_email(self, server, email_id):
        try:
            # Fetch email
            status, data = server.fetch(email_id, '(RFC822)')
            raw_email = data[0][1]
            email_message = email.message_from_bytes(raw_email)
            
            # Extract email details
            subject = decode_header(email_message['subject'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            
            from_addr = decode_header(email_message['from'])[0][0]
            if isinstance(from_addr, bytes):
                from_addr = from_addr.decode()
                
            date = email_message['date']
            
            # Extract email body
            body = ""
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        try:
                            body = part.get_payload(decode=True).decode()
                        except:
                            body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        break
            else:
                try:
                    body = email_message.get_payload(decode=True).decode()
                except:
                    body = email_message.get_payload(decode=True).decode('utf-8', errors='replace')
            
            # Clean the body text
            body = re.sub(r'\s+', ' ', body).strip()
            
            # Predict if spam
            features = self.vectorizer.transform([f"{subject} {body}"])
            prediction = self.model.predict(features)[0]
            
            # Get probability scores
            probability = self.model.predict_proba(features)[0]
            confidence = probability[1] if prediction == 1 else probability[0]
            
            # Determine status
            status = "Ham" if prediction == 1 else "Spam"
            
            # Apply actions based on settings
            if status == "Spam" and self.auto_move_spam_var.get():
                # Move to spam folder
                server.copy(email_id, '[Gmail]/Spam')
                server.store(email_id, '+FLAGS', '\\Deleted')
                
            if status == "Spam" and self.mark_spam_read_var.get():
                # Mark as read
                server.store(email_id, '+FLAGS', '\\Seen')
                
            # Display notification if enabled
            if self.enable_notif_var.get():
                if status == "Spam" or not self.notif_spam_only_var.get():
                    self.show_notification(subject, from_addr, status)
            
            # Add to treeview
            confidence_pct = f"{confidence*100:.1f}%"
            item_id = self.email_tree.insert("", "end", values=(status, from_addr, subject, date, confidence_pct))
            
            # Store email data for preview
            self.email_data.append({
                "id": item_id,
                "status": status,
                "from": from_addr,
                "subject": subject,
                "date": date,
                "confidence": confidence,
                "body": body
            })
            
            # Update email count statistics
            self.update_email_stats(status)
            
            return True
        except Exception as e:
            print(f"Error processing email: {str(e)}")
            return False
            
    def show_notification(self, subject, sender, status):
        # Simple notification using messagebox
        # In a real app, you might use a platform-specific notification system
        duration = int(self.notif_duration_var.get())
        
        # Create a top-level window for notification
        notif = tk.Toplevel(self.root)
        notif.title("New Email")
        notif.geometry("300x100+50+50")
        notif.attributes("-topmost", True)
        
        # Notification content
        frame = ttk.Frame(notif, padding=10)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text=f"New {status} Email", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"From: {sender}").pack(anchor="w")
        ttk.Label(frame, text=f"Subject: {subject}", wraplength=280).pack(anchor="w")
        
        # Auto-close after duration
        notif.after(duration * 1000, notif.destroy)
        
    def view_email_details(self, event):
        selected_item = self.email_tree.selection()[0]
        
        # Find the corresponding email in self.email_data
        email_data = None
        for email in self.email_data:
            if email["id"] == selected_item:
                email_data = email
                break
                
        if email_data:
            # Display email preview
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, f"From: {email_data['from']}\n")
            self.preview_text.insert(tk.END, f"Subject: {email_data['subject']}\n")
            self.preview_text.insert(tk.END, f"Date: {email_data['date']}\n")
            self.preview_text.insert(tk.END, f"Status: {email_data['status']} (Confidence: {email_data['confidence']*100:.1f}%)\n")
            self.preview_text.insert(tk.END, "\n" + "-"*50 + "\n\n")
            self.preview_text.insert(tk.END, email_data['body'])
            
    def clear_email_list(self):
        # Clear the email treeview and data
        for item in self.email_tree.get_children():
            self.email_tree.delete(item)
        self.email_data = []
        self.preview_text.delete(1.0, tk.END)
        
    def update_email_stats(self, status):
        # Update email statistics
        if not hasattr(self, 'email_stats'):
            self.email_stats = {
                "total": 0,
                "spam": 0,
                "ham": 0,
                "by_hour": {},
                "last_updated": time.time()
            }
            
        # Update counts
        self.email_stats["total"] += 1
        if status == "Spam":
            self.email_stats["spam"] += 1
        else:
            self.email_stats["ham"] += 1
            
        # Update hourly stats
        hour = time.strftime("%H")
        if hour not in self.email_stats["by_hour"]:
            self.email_stats["by_hour"][hour] = 0
        self.email_stats["by_hour"][hour] += 1
        
        # Update last updated time
        self.email_stats["last_updated"] = time.time()
        
    def update_analysis(self):
        analysis_type = self.analysis_var.get()
        
        # Clear previous plot
        self.ax.clear()
        
        if analysis_type == "Spam vs Ham Distribution":
            self.plot_spam_ham_distribution()
        elif analysis_type == "Detection Accuracy Over Time":
            self.plot_accuracy_over_time()
        elif analysis_type == "Top Spam Keywords":
            self.plot_top_spam_keywords()
        elif analysis_type == "Email Volume by Hour":
            self.plot_email_volume_by_hour()
            
        # Update the canvas
        self.canvas.draw()
        
    def plot_spam_ham_distribution(self):
        # Plot spam vs ham distribution
        if hasattr(self, 'mail_data'):
            # For training data
            spam_count = len(self.raw_mail_data[self.raw_mail_data['Category'] == 'spam'])
            ham_count = len(self.raw_mail_data[self.raw_mail_data['Category'] == 'ham'])
            
            labels = ['Spam', 'Ham']
            counts = [spam_count, ham_count]
            colors = ['#ff9999', '#66b3ff']
            
            self.ax.bar(labels, counts, color=colors)
            self.ax.set_title('Spam vs Ham Distribution in Training Data')
            self.ax.set_ylabel('Count')
            
            # Add counts on top of bars
            for i, count in enumerate(counts):
                self.ax.text(i, count + 0.1, str(count), ha='center')
                
            # Update stats
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Total Emails: {spam_count + ham_count}\n")
            self.stats_text.insert(tk.END, f"Spam Emails: {spam_count} ({spam_count/(spam_count + ham_count)*100:.1f}%)\n")
            self.stats_text.insert(tk.END, f"Ham Emails: {ham_count} ({ham_count/(spam_count + ham_count)*100:.1f}%)\n")
            
        elif hasattr(self, 'email_stats'):
            # For monitored emails
            labels = ['Spam', 'Ham']
            counts = [self.email_stats["spam"], self.email_stats["ham"]]
            colors = ['#ff9999', '#66b3ff']
            
            self.ax.bar(labels, counts, color=colors)
            self.ax.set_title('Spam vs Ham Distribution in Monitored Emails')
            self.ax.set_ylabel('Count')
            
            # Add counts on top of bars
            for i, count in enumerate(counts):
                self.ax.text(i, count + 0.1, str(count), ha='center')
                
            # Update stats
            self.stats_text.delete(1.0, tk.END)
            total = self.email_stats["total"]
            if total > 0:
                self.stats_text.insert(tk.END, f"Total Emails: {total}\n")
                self.stats_text.insert(tk.END, f"Spam Emails: {self.email_stats['spam']} ({self.email_stats['spam']/total*100:.1f}%)\n")
                self.stats_text.insert(tk.END, f"Ham Emails: {self.email_stats['ham']} ({self.email_stats['ham']/total*100:.1f}%)\n")
                self.stats_text.insert(tk.END, f"\nLast Updated: {time.strftime('%H:%M:%S')}")
        else:
            self.ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            
    def plot_accuracy_over_time(self):
        # This would normally plot accuracy over time from a history log
        # For demo purposes, we'll just create some mock data
        
        if hasattr(self, 'model'):
            # Mock data for demonstration
            dates = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
            accuracy = [0.92, 0.93, 0.95, 0.94, 0.96]
            precision = [0.90, 0.92, 0.94, 0.93, 0.95]
            recall = [0.89, 0.91, 0.93, 0.92, 0.94]
            
            self.ax.plot(dates, accuracy, marker='o', label='Accuracy')
            self.ax.plot(dates, precision, marker='s', label='Precision')
            self.ax.plot(dates, recall, marker='^', label='Recall')
            
            self.ax.set_title('Model Performance Over Time')
            self.ax.set_xlabel('Month')
            self.ax.set_ylabel('Score')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Update stats
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Model Performance Metrics:\n\n")
            self.stats_text.insert(tk.END, f"Current Accuracy: {accuracy[-1]:.4f}\n")
            self.stats_text.insert(tk.END, f"Current Precision: {precision[-1]:.4f}\n")
            self.stats_text.insert(tk.END, f"Current Recall: {recall[-1]:.4f}\n\n")
            self.stats_text.insert(tk.END, f"Accuracy Improvement: {(accuracy[-1] - accuracy[0])*100:.2f}%\n")
            
        else:
            self.ax.text(0.5, 0.5, 'No model trained', ha='center', va='center')
            
    def plot_top_spam_keywords(self):
        if hasattr(self, 'vectorizer') and hasattr(self, 'model'):
            # Get feature names and their coefficients
            try:
                feature_names = self.vectorizer.get_feature_names_out()
                
                # For logistic regression and SVM
                if hasattr(self.model, 'coef_'):
                    coefficients = self.model.coef_[0]
                # For random forest
                elif hasattr(self.model, 'feature_importances_'):
                    coefficients = self.model.feature_importances_
                else:
                    raise AttributeError("Model doesn't provide feature importance")
                    
                # Create DataFrame of features and importance
                features_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coefficients
                })
                
                # Sort by absolute importance
                features_df['abs_importance'] = abs(features_df['importance'])
                features_df = features_df.sort_values('abs_importance', ascending=False)
                
                # Get top 10 features
                top_features = features_df.head(10)
                
                # Plot
                colors = ['#ff9999' if x < 0 else '#66b3ff' for x in top_features['importance']]
                self.ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], color=colors)
                self.ax.set_title('Top 10 Important Features for Spam Detection')
                self.ax.set_xlabel('Importance')
                
                # Update stats
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, "Top Spam Keywords:\n\n")
                
                for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance'])):
                    self.stats_text.insert(tk.END, f"{i+1}. {feature}: {importance:.4f}\n")
                    
            except Exception as e:
                print(f"Error in feature importance analysis: {str(e)}")
                self.ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        else:
            self.ax.text(0.5, 0.5, 'No model trained', ha='center', va='center')
            
    def plot_email_volume_by_hour(self):
        if hasattr(self, 'email_stats') and 'by_hour' in self.email_stats:
            # Sort hours
            hours = sorted(list(self.email_stats['by_hour'].keys()))
            counts = [self.email_stats['by_hour'][hour] for hour in hours]
            
            # Plot
            self.ax.bar(hours, counts, color='#66b3ff')
            self.ax.set_title('Email Volume by Hour')
            self.ax.set_xlabel('Hour of Day')
            self.ax.set_ylabel('Number of Emails')
            
            # Add grid
            self.ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Update stats
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Email Volume Statistics:\n\n")
            
            if counts:
                peak_hour = hours[counts.index(max(counts))]
                self.stats_text.insert(tk.END, f"Peak Hour: {peak_hour}:00 ({max(counts)} emails)\n")
                self.stats_text.insert(tk.END, f"Average Emails per Hour: {sum(counts)/len(counts):.1f}\n")
                self.stats_text.insert(tk.END, f"Total Hours Monitored: {len(hours)}\n")
            else:
                self.stats_text.insert(tk.END, "No hourly data available yet")
                
        else:
            self.ax.text(0.5, 0.5, 'No email data available', ha='center', va='center')
            
    def save_settings(self):
        # Save user settings
        settings = {
            "enable_notifications": self.enable_notif_var.get(),
            "notify_spam_only": self.notif_spam_only_var.get(),
            "notification_duration": self.notif_duration_var.get(),
            "spam_threshold": self.spam_threshold_var.get(),
            "auto_move_spam": self.auto_move_spam_var.get(),
            "mark_spam_read": self.mark_spam_read_var.get(),
            "refresh_interval": self.refresh_var.get()
        }
        
        try:
            with open("settings.pkl", "wb") as f:
                pickle.dump(settings, f)
                
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def load_settings(self):
        try:
            with open("settings.pkl", "rb") as f:
                settings = pickle.load(f)
                
            # Apply settings
            self.enable_notif_var.set(settings.get("enable_notifications", True))
            self.notif_spam_only_var.set(settings.get("notify_spam_only", True))
            self.notif_duration_var.set(settings.get("notification_duration", "5"))
            self.spam_threshold_var.set(settings.get("spam_threshold", "0.5"))
            self.auto_move_spam_var.set(settings.get("auto_move_spam", False))
            self.mark_spam_read_var.set(settings.get("mark_spam_read", False))
            self.refresh_var.set(settings.get("refresh_interval", "60"))
            
        except FileNotFoundError:
            # Default settings
            pass
        except Exception as e:
            print(f"Error loading settings: {str(e)}")

# Main function
def main():
    root = tk.Tk()
    app = SpamFilterApp(root)
    app.load_settings()  # Load saved settings
    root.mainloop()

if __name__ == "__main__":
    main()