from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, Response
import sqlite3
import os
import uuid
import qrcode
from datetime import datetime, timedelta
import csv
import io
import pickle
import numpy as np
import requests
import random
import warnings
import re

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, auth
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[WARNING] Firebase Admin SDK not installed. Install with: pip install firebase-admin")

# Try to use config.py if available, otherwise use direct configuration
try:
    from config import get_config
    config = get_config()
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    config = None

# Initialize Flask app with frontend folder paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)  # Go up one level to project root
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# Use config if available, otherwise use defaults
if USE_CONFIG and config:
    app = Flask(__name__, 
                template_folder=config.TEMPLATE_FOLDER,
                static_folder=config.STATIC_FOLDER)
    app.secret_key = config.SECRET_KEY
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    DB_PATH = config.DB_PATH
    FIREBASE_CREDENTIALS_PATH = config.FIREBASE_CREDENTIALS_PATH
    FIREBASE_WEB_API_KEY = config.FIREBASE_WEB_API_KEY
    OTP_CODE_LENGTH = config.OTP_CODE_LENGTH
    OTP_EXPIRY_MINUTES = config.OTP_EXPIRY_MINUTES
    UPLOAD_FOLDER = config.UPLOAD_FOLDER
    QR_FOLDER = config.QR_FOLDER
    MODEL_PATH = config.MODEL_PATH
    EMERGENCY_MODEL_PATH = config.EMERGENCY_MODEL_PATH
else:
    # Fallback to direct configuration (backward compatibility)
    app = Flask(__name__, 
                template_folder=os.path.join(FRONTEND_DIR, 'templates'),
                static_folder=os.path.join(FRONTEND_DIR, 'static'))
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
    DB_PATH = os.path.join(BACKEND_DIR, 'health_system.db')
    FIREBASE_CREDENTIALS_PATH = os.path.join(BACKEND_DIR, 'pkl', 'swasthya-sampark-firebase-adminsdk-fbsvc-121be5c997.json')
    if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
        FIREBASE_CREDENTIALS_PATH = os.path.join(BACKEND_DIR, 'firebase_service_account.json')
    FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY', 'BC5Hbsevk0B2jrRVwGVm0iMK0mq-2DaefIRLd_0aueAUz6LABC5jApBBqkfvLw6vTB3PWAwsCgsdkvvC2QlMa_c')
    OTP_CODE_LENGTH = int(os.environ.get('OTP_CODE_LENGTH', 6))
    OTP_EXPIRY_MINUTES = int(os.environ.get('OTP_EXPIRY_MINUTES', 10))
    UPLOAD_FOLDER = os.path.join(BACKEND_DIR, 'uploads')
    QR_FOLDER = os.path.join(FRONTEND_DIR, 'static', 'qr')
    MODEL_PATH = os.path.join(BACKEND_DIR, 'pkl', 'svm_health_risk_model.pkl')
    EMERGENCY_MODEL_PATH = os.path.join(BACKEND_DIR, 'pkl', 'Logistic_regression_prediction.pkl')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Firebase Configuration
FIREBASE_INITIALIZED = False

# Initialize Firebase Admin SDK
if FIREBASE_AVAILABLE:
    try:
        # Try service account JSON first (preferred for Admin SDK)
        if os.path.exists(FIREBASE_CREDENTIALS_PATH):
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
            FIREBASE_INITIALIZED = True
            print("[OK] Firebase Admin SDK initialized successfully (Service Account)")
        # If no service account, use web API key for client-side operations
        elif FIREBASE_WEB_API_KEY:
            # Web API Key is configured - enable Firebase features
            # Note: Admin SDK requires service account for full features
            # Web API key enables client-side Firebase operations and validation
            FIREBASE_INITIALIZED = True
            print(f"[OK] Firebase Web API Key configured: {FIREBASE_WEB_API_KEY[:30]}...")
            print("[INFO] Web API Key enabled for client-side Firebase operations")
            print("[INFO] For full Admin SDK features, add service account JSON file")
        else:
            print(f"[WARNING] Firebase credentials file not found at {FIREBASE_CREDENTIALS_PATH}")
            print("[INFO] OTP verification will use database storage only")
            print("[INFO] To enable Firebase:")
            print("  1. Download service account key and save as 'firebase_service_account.json'")
            print("  2. Or set FIREBASE_WEB_API_KEY environment variable")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Firebase Admin SDK: {e}")
        print("[INFO] OTP verification will use database storage only")
else:
    print("[WARNING] Firebase Admin SDK not available. OTP verification will use database storage only")
# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QR_FOLDER, exist_ok=True)

# Load health risk prediction model (SVM)
health_risk_model = None
model_scaler = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            # Handle both old format (just model) and new format (dict with model and scaler)
            if isinstance(model_data, dict):
                health_risk_model = model_data.get('model')
                model_scaler = model_data.get('scaler')
                print(f"[OK] Health risk model loaded successfully from {MODEL_PATH}")
                print(f"[OK] Model includes scaler for feature normalization")
            else:
                # Old format - just the model
                health_risk_model = model_data
                model_scaler = None
                print(f"[OK] Health risk model loaded (legacy format, no scaler)")
    else:
        print(f"[WARNING] Health risk model not found at {MODEL_PATH}")
        print(f"[INFO] Run 'python train_model.py' to train a new model")
except Exception as e:
    print(f"[ERROR] Error loading health risk model: {e}")
    print(f"[INFO] Continuing without AI risk prediction. Emergency triggers will be based on treatment status only.")

# Load emergency priority prediction model (Logistic Regression)
emergency_model = None
emergency_scaler = None
try:
    if os.path.exists(EMERGENCY_MODEL_PATH):
        # Try multiple loading methods including joblib
        loaded = False
        
        # Method 1: Try joblib (common for scikit-learn models)
        try:
            import joblib
            emergency_data = joblib.load(EMERGENCY_MODEL_PATH)
            if isinstance(emergency_data, dict):
                emergency_model = emergency_data.get('model') or emergency_data.get('logistic_model') or emergency_data.get('classifier')
                emergency_scaler = emergency_data.get('scaler')
            elif hasattr(emergency_data, 'predict'):
                # Direct model object (LogisticRegression)
                emergency_model = emergency_data
                emergency_scaler = None
            
            if emergency_model and hasattr(emergency_model, 'predict'):
                print(f"[OK] Emergency prediction model loaded successfully using joblib")
                print(f"[OK] Model type: {type(emergency_model).__name__}")
                if hasattr(emergency_model, 'n_features_in_'):
                    print(f"[OK] Model expects {emergency_model.n_features_in_} features")
                if emergency_scaler:
                    print(f"[OK] Emergency model includes scaler for feature normalization")
                else:
                    print(f"[INFO] No scaler found - features will be used as-is")
                loaded = True
        except ImportError:
            print(f"[INFO] joblib not available, trying other methods...")
        except Exception as e:
            print(f"[INFO] joblib loading failed: {e}, trying other methods...")
        
        # Method 2: Try pickle with different protocols and encodings
        if not loaded:
            loading_methods = [
                ('standard', lambda f: pickle.load(f)),
                ('latin1', lambda f: pickle.load(f, encoding='latin1')),
                ('bytes', lambda f: pickle.load(f, encoding='bytes')),
                ('protocol4', lambda f: pickle.load(f, fix_imports=True)),
            ]
            
            for method_name, load_func in loading_methods:
                try:
                    with open(EMERGENCY_MODEL_PATH, 'rb') as f:
                        emergency_data = load_func(f)
                        # Handle different formats
                        if isinstance(emergency_data, dict):
                            emergency_model = emergency_data.get('model') or emergency_data.get('logistic_model') or emergency_data.get('classifier')
                            emergency_scaler = emergency_data.get('scaler')
                            if emergency_model and hasattr(emergency_model, 'predict'):
                                print(f"[OK] Emergency prediction model loaded successfully ({method_name})")
                                if emergency_scaler:
                                    print(f"[OK] Emergency model includes scaler for feature normalization")
                                loaded = True
                                break
                        else:
                            # Assume it's the model directly
                            if hasattr(emergency_data, 'predict'):
                                emergency_model = emergency_data
                                emergency_scaler = None
                                print(f"[OK] Emergency prediction model loaded (direct format, {method_name})")
                                loaded = True
                                break
                except Exception as e:
                    continue  # Try next method
        
        # Method 3: Try with dill (more compatible pickle alternative)
        if not loaded:
            try:
                import dill
                with open(EMERGENCY_MODEL_PATH, 'rb') as f:
                    emergency_data = dill.load(f)
                    if isinstance(emergency_data, dict):
                        emergency_model = emergency_data.get('model') or emergency_data.get('logistic_model') or emergency_data.get('classifier')
                        emergency_scaler = emergency_data.get('scaler')
                    elif hasattr(emergency_data, 'predict'):
                        emergency_model = emergency_data
                        emergency_scaler = None
                    
                    if emergency_model and hasattr(emergency_model, 'predict'):
                        print(f"[OK] Emergency prediction model loaded successfully using dill")
                        if emergency_scaler:
                            print(f"[OK] Emergency model includes scaler for feature normalization")
                        loaded = True
            except ImportError:
                pass  # dill not available
            except Exception as e:
                pass  # dill failed
        
        if not loaded:
            print(f"[WARNING] Could not load emergency model with any method")
            print(f"[INFO] Emergency section will use rule-based priority prediction")
            print(f"[INFO] To fix: The model file may need to be re-saved with Python 3.x compatible pickle protocol")
    else:
        print(f"[WARNING] Emergency prediction model not found at {EMERGENCY_MODEL_PATH}")
        print(f"[INFO] Emergency section will use rule-based priority prediction")
except Exception as e:
    print(f"[ERROR] Error loading emergency prediction model: {e}")
    print(f"[INFO] Emergency section will use rule-based priority prediction")


# Helper function to get DB connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize database with basic schema
def init_db():
    """Initialize database and create all tables if they don't exist"""
    # Ensure database directory exists
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"[INFO] Created database directory: {db_dir}")
    
    print(f"[INFO] Initializing database at: {DB_PATH}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        print("[INFO] Creating database tables...")
        
        # Hospitals
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS hospitals (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL,
                   reg_no TEXT UNIQUE NOT NULL,
                   email TEXT UNIQUE NOT NULL,
                   password TEXT NOT NULL,
                   state TEXT,
                   district TEXT
               )'''
        )
        print("[OK] Created/verified hospitals table")

        # Doctors
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS doctors (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   hospital_id INTEGER NOT NULL,
                   name TEXT NOT NULL,
                   email TEXT UNIQUE NOT NULL,
                   password TEXT NOT NULL,
                   specialization TEXT,
                   FOREIGN KEY(hospital_id) REFERENCES hospitals(id)
               )'''
        )
        print("[OK] Created/verified doctors table")

        # Users / Patients
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS users (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL,
                   email TEXT UNIQUE NOT NULL,
                   password TEXT NOT NULL,
                   phone TEXT,
                   address TEXT,
                   health_id TEXT UNIQUE NOT NULL,
                   age INTEGER,
                   gender TEXT
               )'''
        )
        print("[OK] Created/verified users table")

        # Medical records (per consultation)
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS records (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_id INTEGER NOT NULL,
                   doctor_id INTEGER NOT NULL,
                   date TEXT NOT NULL,
                   symptoms TEXT,
                   diagnosis TEXT,
                   medicines TEXT,
                   dosage TEXT,
                   treatment_status TEXT,
                   consultation_duration INTEGER,
                   prescription_text TEXT,
                   prescription_filename TEXT,
                   blood_report_filename TEXT,
                   report_filename TEXT,
                   created_at TEXT NOT NULL,
                   risk_level TEXT,
                   risk_score REAL,
                   FOREIGN KEY(user_id) REFERENCES users(id),
                   FOREIGN KEY(doctor_id) REFERENCES doctors(id)
               )'''
        )
        print("[OK] Created/verified records table")

        # Emergency requests
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS emergencies (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_id INTEGER,
                   name TEXT,
                   phone TEXT,
                   location TEXT NOT NULL,
                   status TEXT NOT NULL,
                   requested_at TEXT NOT NULL,
                   response_time_minutes INTEGER
               )'''
        )
        print("[OK] Created/verified emergencies table")

        # OTP storage for password reset and login
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS otp_codes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   phone TEXT NOT NULL,
                   code TEXT NOT NULL,
                   role TEXT NOT NULL,
                   identifier TEXT NOT NULL,
                   purpose TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   expires_at TEXT NOT NULL,
                   verified INTEGER DEFAULT 0
               )'''
        )
        print("[OK] Created/verified otp_codes table")

        conn.commit()
        
        conn.commit()
        
        # Add missing columns if they don't exist
        # Hospitals table
        try:
            cur.execute('ALTER TABLE hospitals ADD COLUMN state TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Emergencies table - ML prediction columns
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN priority TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN severity TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN prediction_score REAL')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN symptoms TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN age INTEGER')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Emergency additional fields for ML model
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN state TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN zone TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN day TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN time_slot TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN emergency_type TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE emergencies ADD COLUMN weather TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE hospitals ADD COLUMN district TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Users table
        try:
            cur.execute('ALTER TABLE users ADD COLUMN age INTEGER')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE users ADD COLUMN gender TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Doctors table - add phone
        try:
            cur.execute('ALTER TABLE doctors ADD COLUMN phone TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Hospitals table - add phone
        try:
            cur.execute('ALTER TABLE hospitals ADD COLUMN phone TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Records table
        try:
            cur.execute('ALTER TABLE records ADD COLUMN dosage TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN treatment_status TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN consultation_duration INTEGER')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN prescription_text TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN prescription_filename TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN blood_report_filename TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN blood_report_file TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN prescription_file TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Add risk prediction columns
        try:
            cur.execute('ALTER TABLE records ADD COLUMN risk_level TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cur.execute('ALTER TABLE records ADD COLUMN risk_score REAL')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Add health metrics columns for risk prediction
        health_metrics_columns = [
            ('systolic_bp', 'INTEGER'),
            ('diastolic_bp', 'INTEGER'),
            ('bmi', 'REAL'),
            ('cholesterol', 'REAL'),
            ('glucose', 'REAL'),
            ('smoking', 'TEXT'),
            ('alcohol', 'TEXT'),
            ('physical_activity', 'TEXT'),
            ('family_history', 'TEXT'),
        ]
        
        for col_name, col_type in health_metrics_columns:
            try:
                cur.execute(f'ALTER TABLE records ADD COLUMN {col_name} {col_type}')
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        conn.close()
        print("[OK] Database initialization completed successfully")
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        # Try to close connection if it was opened
        try:
            if 'conn' in locals():
                conn.close()
        except:
            pass
        raise  # Re-raise to ensure we know about the error


# Utility: generate unique health ID
def generate_health_id():
    # Short UUID-based ID, e.g., H-3F9C-82B1
    raw = uuid.uuid4().hex[:8].upper()
    return f"H-{raw[:4]}-{raw[4:]}"


# Utility: create QR code for health ID
def generate_health_qr(health_id):
    qr_path = os.path.join(QR_FOLDER, f"{health_id}.png")
    if not os.path.exists(qr_path):
        img = qrcode.make(health_id)
        img.save(qr_path)
    return f"qr/{health_id}.png"


# Health Risk Prediction Function
def predict_health_risk(user_data, symptoms, diagnosis, treatment_status, medicines, health_metrics=None):
    """
    Predict health risk using SVM model with rule-based fallback.
    Returns: (risk_level, risk_score, should_trigger_emergency)
    risk_level: 'Low', 'Medium', 'High', 'Critical'
    risk_score: probability/confidence score
    should_trigger_emergency: boolean
    
    Args:
        user_data: User record from database
        symptoms: Symptoms text
        diagnosis: Diagnosis text
        treatment_status: Treatment status
        medicines: Medicines text
        health_metrics: Dict with health metrics (age, gender, systolic_bp, etc.)
    """
    # Rule-based fallback if model is not available
    use_model = health_risk_model is not None
    
    # Default health_metrics if not provided
    if health_metrics is None:
        health_metrics = {}
    
    try:
        # Extract features from available data
        # Priority: health_metrics > user_data > defaults
        
        # Age: from health_metrics, then user_data, then default
        age = health_metrics.get('age')
        if age is None and user_data:
            if hasattr(user_data, 'get'):
                age = user_data.get('age')
            else:
                age = user_data['age'] if 'age' in user_data.keys() else None
        if age is None:
            age = 40
        
        # Ensure age is a number
        try:
            age = int(age) if age else 40
        except (ValueError, TypeError):
            age = 40
        
        age_normalized = min(age / 100.0, 1.0)  # Normalize age to 0-1
        
        # Extract health metrics
        systolic_bp = health_metrics.get('systolic_bp')
        diastolic_bp = health_metrics.get('diastolic_bp')
        bmi = health_metrics.get('bmi')
        cholesterol = health_metrics.get('cholesterol')
        glucose = health_metrics.get('glucose')
        smoking = health_metrics.get('smoking', '')
        alcohol = health_metrics.get('alcohol', '')
        physical_activity = health_metrics.get('physical_activity', '')
        family_history = health_metrics.get('family_history', '')
        
        # Normalize BP (normal: 120/80, high: >140/90)
        bp_normalized = 0.5  # Default
        if systolic_bp and diastolic_bp:
            if systolic_bp > 140 or diastolic_bp > 90:
                bp_normalized = 1.0  # High BP
            elif systolic_bp < 90 or diastolic_bp < 60:
                bp_normalized = 0.8  # Low BP
            else:
                bp_normalized = 0.3  # Normal BP
        
        # Normalize BMI (normal: 18.5-24.9, overweight: 25-29.9, obese: >30)
        bmi_normalized = 0.5  # Default
        if bmi:
            if bmi < 18.5:
                bmi_normalized = 0.6  # Underweight
            elif bmi > 30:
                bmi_normalized = 1.0  # Obese
            elif bmi > 25:
                bmi_normalized = 0.7  # Overweight
            else:
                bmi_normalized = 0.3  # Normal
        
        # Normalize cholesterol (normal: <200, high: >240)
        cholesterol_normalized = 0.5  # Default
        if cholesterol:
            if cholesterol > 240:
                cholesterol_normalized = 1.0  # High
            elif cholesterol > 200:
                cholesterol_normalized = 0.7  # Borderline
            else:
                cholesterol_normalized = 0.3  # Normal
        
        # Normalize glucose (normal: <100, prediabetic: 100-125, diabetic: >125)
        glucose_normalized = 0.5  # Default
        if glucose:
            if glucose > 125:
                glucose_normalized = 1.0  # High (diabetic)
            elif glucose > 100:
                glucose_normalized = 0.7  # Borderline
            else:
                glucose_normalized = 0.3  # Normal
        
        # Encode lifestyle factors using new encoding scheme
        # Smoking: 0 = Non-Smoker, 1 = Smoker
        try:
            smoking_risk = int(smoking) if smoking and smoking.isdigit() else 0
        except (ValueError, TypeError):
            smoking_risk = 0
        
        # Alcohol: 0 = Habit Absent, 1 = Habit Present
        try:
            alcohol_risk = int(alcohol) if alcohol and alcohol.isdigit() else 0
        except (ValueError, TypeError):
            alcohol_risk = 0
        
        # Physical Activity: 0-5 scale (0 = No activity/Sedentary, 5 = 5+ hours/week)
        # For model: normalize to 0-1, but higher activity = lower risk (inverted)
        try:
            activity_value = int(physical_activity) if physical_activity and physical_activity.isdigit() else 0
            # Invert: 0 (no activity) = high risk (1.0), 5 (active) = low risk (0.0)
            activity_risk = 1.0 - (activity_value / 5.0) if activity_value <= 5 else 0.0
        except (ValueError, TypeError):
            activity_risk = 0.5  # Default medium risk
        
        # Family History: 0 = No family history, 1 = Family history present
        try:
            family_history_risk = int(family_history) if family_history and family_history.isdigit() else 0
        except (ValueError, TypeError):
            family_history_risk = 0
        
        # Encode symptoms (simple keyword-based severity)
        symptoms_lower = (symptoms or '').lower()
        symptom_severity = 0.0
        critical_keywords = ['chest pain', 'difficulty breathing', 'unconscious', 'severe', 'emergency', 'critical', 'heart attack', 'stroke']
        high_keywords = ['pain', 'fever', 'bleeding', 'dizziness', 'nausea', 'vomiting']
        medium_keywords = ['cough', 'headache', 'fatigue', 'weakness']
        
        if any(keyword in symptoms_lower for keyword in critical_keywords):
            symptom_severity = 1.0
        elif any(keyword in symptoms_lower for keyword in high_keywords):
            symptom_severity = 0.7
        elif any(keyword in symptoms_lower for keyword in medium_keywords):
            symptom_severity = 0.4
        else:
            symptom_severity = 0.1
        
        # Encode diagnosis severity
        diagnosis_lower = (diagnosis or '').lower()
        diagnosis_severity = 0.0
        critical_diagnosis = ['heart attack', 'stroke', 'severe', 'critical', 'emergency', 'cardiac', 'respiratory failure']
        high_diagnosis = ['hypertension', 'diabetes', 'infection', 'fracture', 'injury']
        medium_diagnosis = ['checkup', 'routine', 'follow-up']
        
        if any(keyword in diagnosis_lower for keyword in critical_diagnosis):
            diagnosis_severity = 1.0
        elif any(keyword in diagnosis_lower for keyword in high_diagnosis):
            diagnosis_severity = 0.6
        elif any(keyword in diagnosis_lower for keyword in medium_diagnosis):
            diagnosis_severity = 0.2
        else:
            diagnosis_severity = 0.3
        
        # Encode treatment status
        status_encoding = {
            'Under Observation': 0.8,
            'Stable': 0.4,
            'Recovered': 0.1,
            'Critical': 1.0,
            'Emergency': 1.0
        }
        treatment_severity = status_encoding.get(treatment_status, 0.5)
        
        # Medicine count (more medicines might indicate complexity)
        medicine_count = len([m.strip() for m in (medicines or '').split(',') if m.strip()]) if medicines else 0
        medicine_complexity = min(medicine_count / 5.0, 1.0)  # Normalize to 0-1
        
        # Use model if available, otherwise use rule-based assessment
        if use_model:
            # Prepare feature vector with health metrics
            features = np.array([[
                age_normalized,
                symptom_severity,
                diagnosis_severity,
                treatment_severity,
                medicine_complexity,
                bp_normalized,
                bmi_normalized,
                cholesterol_normalized,
                glucose_normalized,
                smoking_risk,
                alcohol_risk,
                activity_risk,
                family_history_risk
            ]])
            
            # Scale features if scaler is available
            if model_scaler is not None:
                features_scaled = model_scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            prediction = health_risk_model.predict(features_scaled)[0]
            
            # Get prediction probability if available
            try:
                probabilities = health_risk_model.predict_proba(features_scaled)[0]
                risk_score = float(max(probabilities))
            except:
                risk_score = 0.5
        else:
            # Rule-based assessment when model is not available
            # Calculate composite risk score with health metrics
            base_risk = (symptom_severity * 0.2 + diagnosis_severity * 0.2 + 
                        treatment_severity * 0.2 + medicine_complexity * 0.1)
            
            # Add health metrics to risk calculation
            # smoking_risk, alcohol_risk, family_history_risk are now 0 or 1 (binary)
            # activity_risk is inverted (0 = no activity = high risk 1.0, 5 = active = low risk 0.0)
            health_risk = (bp_normalized * 0.1 + bmi_normalized * 0.1 + 
                          cholesterol_normalized * 0.05 + glucose_normalized * 0.05 +
                          float(smoking_risk) * 0.05 + float(alcohol_risk) * 0.03 + 
                          activity_risk * 0.03 + float(family_history_risk) * 0.02)
            
            risk_score = min(base_risk + health_risk, 1.0)  # Cap at 1.0
            prediction = None  # Will use risk_score for determination
        
        # Determine risk level and emergency trigger
        # Handle different prediction formats
        risk_level = 'Low'
        should_emergency = False
        
        if prediction is not None:
            # Use model prediction
            if isinstance(prediction, (int, np.integer, np.int64, np.int32)):
                pred_value = int(prediction)
                if pred_value >= 2:
                    risk_level = 'Critical'
                    should_emergency = True
                elif pred_value == 1:
                    risk_level = 'High'
                    should_emergency = True
                else:
                    risk_level = 'Low' if risk_score < 0.4 else 'Medium'
                    should_emergency = False
            elif isinstance(prediction, (str, np.str_)):
                # Handle string predictions
                pred_str = str(prediction).lower()
                if 'critical' in pred_str:
                    risk_level = 'Critical'
                    should_emergency = True
                elif 'high' in pred_str:
                    risk_level = 'High'
                    should_emergency = True
                elif 'medium' in pred_str or 'moderate' in pred_str:
                    risk_level = 'Medium'
                    should_emergency = False
                else:
                    risk_level = 'Low'
                    should_emergency = False
            else:
                # Fallback: use risk_score to determine level
                if risk_score >= 0.8:
                    risk_level = 'Critical'
                    should_emergency = True
                elif risk_score >= 0.6:
                    risk_level = 'High'
                    should_emergency = True
                elif risk_score >= 0.4:
                    risk_level = 'Medium'
                    should_emergency = False
                else:
                    risk_level = 'Low'
                    should_emergency = False
        else:
            # Rule-based assessment (when model not available)
            if risk_score >= 0.8:
                risk_level = 'Critical'
                should_emergency = True
            elif risk_score >= 0.6:
                risk_level = 'High'
                should_emergency = True
            elif risk_score >= 0.4:
                risk_level = 'Medium'
                should_emergency = False
            else:
                risk_level = 'Low'
                should_emergency = False
        
        # Override: If treatment status is critical/emergency, always trigger
        if treatment_status in ['Critical', 'Emergency']:
            risk_level = 'Critical'
            should_emergency = True
        
        # Override: If symptoms or diagnosis contain critical keywords, escalate
        if symptom_severity >= 0.9 or diagnosis_severity >= 0.9:
            if risk_level != 'Critical':
                risk_level = 'High'
            should_emergency = True
        
        return risk_level, risk_score, should_emergency
        
    except Exception as e:
        print(f"[ERROR] Error in health risk prediction: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: use rule-based assessment if model fails
        if treatment_status in ['Critical', 'Emergency']:
            return 'Critical', 0.9, True
        elif treatment_status == 'Under Observation':
            return 'Medium', 0.5, False
        return None, None, False


# Role helpers
def login_user(role, user_id):
    session.clear()
    session['role'] = role
    session['user_id'] = user_id


def current_role():
    return session.get('role')


# OTP Helper Functions
def normalize_phone(phone):
    """Normalize phone number by removing spaces, dashes, and other non-digit characters"""
    if not phone:
        return None
    # Remove all non-digit characters except leading +
    normalized = ''.join(c for c in phone if c.isdigit())
    return normalized


def send_otp(phone, role, identifier, purpose='login'):
    """Send OTP via SMS using Firebase Admin SDK"""
    try:
        # Generate OTP code (6 digits)
        otp_code = str(random.randint(100000, 999999))
        
        # Verify phone number format using Firebase (if available)
        if FIREBASE_INITIALIZED:
            try:
                # Firebase Admin SDK can verify phone number format
                # Note: Firebase Auth sends OTP via their service, but we're managing our own OTP
                # This is for phone number validation
                phone_formatted = f"+{phone}" if not phone.startswith('+') else phone
                
                # Store verification session info (optional - for Firebase integration)
                # Firebase would handle actual SMS sending, but we're using our own system
                print(f"[FIREBASE] Phone number validated: {phone_formatted}")
            except Exception as e:
                print(f"[FIREBASE WARNING] Phone validation error: {e}")
        
        # Store OTP in database
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Invalidate previous OTPs for this phone/role/purpose
        cur.execute('''
            UPDATE otp_codes SET verified = 1 
            WHERE phone = ? AND role = ? AND purpose = ? AND verified = 0
        ''', (phone, role, purpose))
        
        # Insert new OTP
        expires_at = (datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
        cur.execute('''
            INSERT INTO otp_codes (phone, code, role, identifier, purpose, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (phone, otp_code, role, identifier, purpose, datetime.utcnow().isoformat(), expires_at))
        
        conn.commit()
        conn.close()
        
        # In production, integrate with Firebase Cloud Messaging or SMS service
        # For now, we'll use a simple approach - in production, use Firebase Auth phone verification
        # or integrate with Firebase Cloud Functions to send SMS
        
        # Log OTP for debugging (visible in Render logs)
        print(f"[OTP DEBUG] OTP for {phone} ({role}): {otp_code}")
        print(f"[OTP INFO] OTP stored in database. Expires at: {expires_at}")
        
        # Show OTP in response message (until SMS service is integrated)
        # TODO: Once SMS is integrated, remove OTP from message and only show "OTP sent to your phone"
        message = f"OTP sent successfully. Your OTP code is: {otp_code} (expires in {OTP_EXPIRY_MINUTES} minutes). Check Render logs if not visible."
        
        # TODO: Integrate with Firebase Cloud Functions or SMS service to actually send SMS
        # For now, OTP is generated and stored - integrate SMS sending service here
        
        return True, message
        
    except Exception as e:
        print(f"[OTP ERROR] {str(e)}")
        return False, f"Error sending OTP: {str(e)}"


def predict_emergency_priority(symptoms, age=None, location=None, state=None, zone=None, 
                              day=None, time_slot=None, emergency_type=None, weather=None, user_data=None):
    """
    Predict emergency priority using Logistic Regression model
    
    Args:
        symptoms: Text description of symptoms
        age: Patient age (optional)
        location: Emergency location (optional)
        state: State name (optional)
        zone: Zone type - Urban, Rural, Highway (optional)
        day: Day of week - Monday, Tuesday, etc. (optional)
        time_slot: Time slot - Morning, Afternoon, Evening, Night (optional)
        emergency_type: Emergency type - EMS, Traffic, Fire (optional)
        weather: Weather condition - Rain, Heatwave, Fog, Clear (optional)
        user_data: User data dict (optional, for additional context)
    
    Returns:
        (priority, severity, prediction_score)
        - priority: 'Low', 'Medium', 'High', 'Critical'
        - severity: 'Mild', 'Moderate', 'Severe', 'Critical'
        - prediction_score: Probability score (0-1)
    """
    if emergency_model is None:
        # Fallback to rule-based prediction
        return predict_emergency_priority_rulebased(symptoms, age, location, state, zone, 
                                                     day, time_slot, emergency_type, weather, user_data)
    
    try:
        # Extract features from symptoms and other inputs
        # This is a simplified feature extraction - adjust based on your model's requirements
        symptom_text = str(symptoms).lower() if symptoms else ''
        
        # Feature extraction (adjust based on your model's actual features)
        # Common emergency features:
        features = []
        
        # Age normalization (0-1 scale, assuming 0-100 range)
        age_normalized = (age / 100.0) if age else 0.5
        
        # Symptom severity keywords
        critical_keywords = ['chest pain', 'difficulty breathing', 'unconscious', 'severe', 'emergency', 
                            'critical', 'heart attack', 'stroke', 'bleeding', 'trauma', 'accident']
        high_keywords = ['pain', 'fever', 'vomiting', 'dizziness', 'nausea', 'weakness']
        moderate_keywords = ['discomfort', 'mild', 'ache', 'tired']
        
        symptom_severity = 0.0
        if any(keyword in symptom_text for keyword in critical_keywords):
            symptom_severity = 1.0
        elif any(keyword in symptom_text for keyword in high_keywords):
            symptom_severity = 0.7
        elif any(keyword in symptom_text for keyword in moderate_keywords):
            symptom_severity = 0.4
        else:
            symptom_severity = 0.2
        
        # States list (35 states/UTs)
        states_list = ['All India', 'Uttar Pradesh', 'Maharashtra', 'West Bengal', 'Jharkhand',
                       'Madhya Pradesh', 'Bihar', 'Rajasthan', 'Tamil Nadu', 'Orissa', 'Assam',
                       'Karnataka', 'Andhra Pradesh', 'Haryana', 'Chhatisgarh', 'Jammu and Kashmir',
                       'Telangana', 'Uttarakhand', 'Himachal Pradesh', 'Gujarat', 'Kerala',
                       'Arunachal Pradesh', 'Delhi', 'Nagaland', 'Mizoram', 'Meghalaya',
                       'Tripura', 'Manipur', 'Goa', 'Andaman and Nicobar Island', 'Ladakh',
                       'Sikkim', 'Puducherry', 'Dadra and Nagar Haveli and Daman and Diu', 'Chandigarh']
        
        # Zone options (3)
        zone_options = ['Urban', 'Rural', 'Highway']
        
        # Day options (7)
        day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Time slot options (4)
        time_slot_options = ['Morning', 'Afternoon', 'Evening', 'Night']
        
        # Emergency type options (3)
        emergency_type_options = ['EMS', 'Traffic', 'Fire']
        
        # Weather options (4)
        weather_options = ['Rain', 'Heatwave', 'Fog', 'Clear']
        
        # Build feature vector with one-hot encoding for categorical variables
        # The model expects 50 features, which likely includes:
        # - Age (1 feature)
        # - Symptom severity (1 feature)
        # - State one-hot (35 features)
        # - Zone one-hot (3 features)
        # - Day one-hot (7 features)
        # - Time slot one-hot (4 features)
        # - Emergency type one-hot (3 features)
        # - Weather one-hot (4 features)
        # Total: 1 + 1 + 35 + 3 + 7 + 4 + 3 + 4 = 58 features (but model expects 50)
        # Let's build it step by step and pad/truncate as needed
        
        features_list = []
        
        # 1. Age (normalized)
        features_list.append(age_normalized)
        
        # 2. Symptom severity
        features_list.append(symptom_severity)
        
        # 3. State one-hot encoding (35 features)
        state_onehot = [0] * len(states_list)
        if state and state in states_list:
            state_onehot[states_list.index(state)] = 1
        features_list.extend(state_onehot)
        
        # 4. Zone one-hot encoding (3 features)
        zone_onehot = [0] * len(zone_options)
        if zone and zone in zone_options:
            zone_onehot[zone_options.index(zone)] = 1
        features_list.extend(zone_onehot)
        
        # 5. Day one-hot encoding (7 features)
        day_onehot = [0] * len(day_options)
        if day and day in day_options:
            day_onehot[day_options.index(day)] = 1
        features_list.extend(day_onehot)
        
        # 6. Time slot one-hot encoding (4 features)
        time_slot_onehot = [0] * len(time_slot_options)
        if time_slot and time_slot in time_slot_options:
            time_slot_onehot[time_slot_options.index(time_slot)] = 1
        features_list.extend(time_slot_onehot)
        
        # 7. Emergency type one-hot encoding (3 features)
        emergency_type_onehot = [0] * len(emergency_type_options)
        if emergency_type and emergency_type in emergency_type_options:
            emergency_type_onehot[emergency_type_options.index(emergency_type)] = 1
        features_list.extend(emergency_type_onehot)
        
        # 8. Weather one-hot encoding (4 features)
        weather_onehot = [0] * len(weather_options)
        if weather and weather in weather_options:
            weather_onehot[weather_options.index(weather)] = 1
        features_list.extend(weather_onehot)
        
        # Total features: 1 + 1 + 35 + 3 + 7 + 4 + 3 + 4 = 58
        base_features = features_list
        
        # Build feature vector (adjust based on your model's expected features)
        if hasattr(emergency_model, 'n_features_in_'):
            n_features = emergency_model.n_features_in_
            if len(base_features) < n_features:
                features = np.array([base_features + [0.0] * (n_features - len(base_features))])
            elif len(base_features) > n_features:
                features = np.array([base_features[:n_features]])
            else:
                features = np.array([base_features])
        else:
            # Default: use all features we have
            features = np.array([base_features])
        
        # Scale features if scaler is available
        if emergency_scaler:
            features = emergency_scaler.transform(features)
        
        # Make prediction (suppress feature name warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            if hasattr(emergency_model, 'predict_proba'):
                # Get probability scores
                probabilities = emergency_model.predict_proba(features)[0]
                prediction_score = float(max(probabilities))
                prediction = emergency_model.predict(features)[0]
            else:
                # Binary or single output
                prediction = emergency_model.predict(features)[0]
                prediction_score = 0.8 if prediction > 0 else 0.2
        
        # Convert prediction to priority levels
        # Model classes are: ['High', 'Low', 'Medium'] - no 'Critical'
        # Map model output to our priority system
        if isinstance(prediction, (str, np.str_)):
            pred_str = str(prediction)
            if pred_str == 'High':
                priority = 'High'
                severity = 'Severe'
            elif pred_str == 'Medium':
                priority = 'Medium'
                severity = 'Moderate'
            elif pred_str == 'Low':
                priority = 'Low'
                severity = 'Mild'
            else:
                # Fallback
                priority = 'Medium'
                severity = 'Moderate'
        elif isinstance(prediction, (int, np.integer, np.int64, np.int32)):
            # If model returns index, map based on classes
            if hasattr(emergency_model, 'classes_'):
                class_name = emergency_model.classes_[prediction]
                if class_name == 'High':
                    priority = 'High'
                    severity = 'Severe'
                elif class_name == 'Medium':
                    priority = 'Medium'
                    severity = 'Moderate'
                else:
                    priority = 'Low'
                    severity = 'Mild'
            else:
                # Fallback mapping
                if prediction >= 2:
                    priority = 'High'
                    severity = 'Severe'
                elif prediction == 1:
                    priority = 'Medium'
                    severity = 'Moderate'
                else:
                    priority = 'Low'
                    severity = 'Mild'
        else:
            # Fallback
            priority = 'Medium'
            severity = 'Moderate'
        
        # If prediction score is very high and priority is High, consider it Critical
        if priority == 'High' and prediction_score >= 0.85:
            priority = 'Critical'
            severity = 'Critical'
        
        return priority, severity, float(prediction_score)
        
    except Exception as e:
        print(f"[ERROR] Emergency prediction error: {e}")
        # Fallback to rule-based
        return predict_emergency_priority_rulebased(symptoms, age, location, state, zone, 
                                                day, time_slot, emergency_type, weather, user_data)


def predict_emergency_priority_rulebased(symptoms, age=None, location=None, state=None, zone=None,
                                         day=None, time_slot=None, emergency_type=None, weather=None, user_data=None):
    """Rule-based fallback for emergency priority prediction"""
    symptom_text = str(symptoms).lower() if symptoms else ''
    
    critical_keywords = ['chest pain', 'difficulty breathing', 'unconscious', 'severe', 'emergency', 
                        'critical', 'heart attack', 'stroke', 'bleeding', 'trauma', 'accident']
    high_keywords = ['pain', 'fever', 'vomiting', 'dizziness', 'nausea', 'weakness']
    
    # Base priority from symptoms
    base_priority = 'Low'
    base_severity = 'Mild'
    base_score = 0.35
    
    if any(keyword in symptom_text for keyword in critical_keywords):
        base_priority = 'Critical'
        base_severity = 'Critical'
        base_score = 0.95
    elif any(keyword in symptom_text for keyword in high_keywords):
        base_priority = 'High'
        base_severity = 'Severe'
        base_score = 0.75
    elif symptoms and len(symptoms) > 20:
        base_priority = 'Medium'
        base_severity = 'Moderate'
        base_score = 0.55
    
    # Adjust based on emergency type
    if emergency_type == 'Fire':
        if base_priority == 'Low':
            base_priority = 'High'
            base_severity = 'Severe'
            base_score = 0.80
        elif base_priority == 'Medium':
            base_priority = 'High'
            base_severity = 'Severe'
            base_score = 0.85
    elif emergency_type == 'Traffic':
        if base_priority == 'Low':
            base_priority = 'Medium'
            base_severity = 'Moderate'
            base_score = 0.60
    
    # Adjust based on weather
    if weather == 'Fog':
        base_score = min(base_score + 0.1, 1.0)
    elif weather == 'Rain':
        base_score = min(base_score + 0.05, 1.0)
    
    # Adjust based on zone
    if zone == 'Highway':
        base_score = min(base_score + 0.1, 1.0)
    elif zone == 'Rural':
        base_score = min(base_score + 0.05, 1.0)
    
    # Update priority based on adjusted score
    if base_score >= 0.8:
        base_priority = 'Critical'
        base_severity = 'Critical'
    elif base_score >= 0.6:
        base_priority = 'High'
        base_severity = 'Severe'
    elif base_score >= 0.4:
        base_priority = 'Medium'
        base_severity = 'Moderate'
    else:
        base_priority = 'Low'
        base_severity = 'Mild'
    
    return base_priority, base_severity, base_score


def verify_otp(phone, code, role, identifier, purpose='login'):
    """Verify OTP code using Firebase Admin SDK and database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Find valid OTP in database
        cur.execute('''
            SELECT * FROM otp_codes 
            WHERE phone = ? AND code = ? AND role = ? AND identifier = ? 
            AND purpose = ? AND verified = 0 AND expires_at > ?
        ''', (phone, code, role, identifier, purpose, datetime.utcnow().isoformat()))
        
        otp_record = cur.fetchone()
        
        if otp_record:
            # Verify with Firebase (if available)
            if FIREBASE_INITIALIZED:
                try:
                    # Format phone number for Firebase
                    phone_formatted = f"+{phone}" if not phone.startswith('+') else phone
                    
                    # Firebase verification
                    if FIREBASE_AVAILABLE and FIREBASE_WEB_API_KEY:
                        # Use Firebase Web API Key for verification
                        # In production, you can use Firebase Authentication REST API
                        # to verify phone number authentication tokens
                        print(f"[FIREBASE] OTP verification using Web API Key for {phone_formatted}")
                        
                        # Optional: Verify Firebase ID token if using Firebase Auth
                        # token = request.headers.get('Authorization', '').replace('Bearer ', '')
                        # decoded_token = auth.verify_id_token(token)
                    else:
                        print(f"[FIREBASE] OTP verification for {phone_formatted}")
                except Exception as e:
                    print(f"[FIREBASE WARNING] Verification error: {e}")
                    # Continue with database verification even if Firebase check fails
            
            # Mark OTP as verified in database
            cur.execute('UPDATE otp_codes SET verified = 1 WHERE id = ?', (otp_record['id'],))
            conn.commit()
            conn.close()
            return True, "OTP verified successfully"
        else:
            conn.close()
            return False, "Invalid or expired OTP"
    except Exception as e:
        return False, f"Error verifying OTP: {str(e)}"


def current_user_id():
    return session.get('user_id')


# Decorator-like helpers (simple checks in routes for beginner friendliness)


@app.route('/')
def index():
    return render_template('index.html')


# -----------------
# Hospital auth
# -----------------


@app.route('/hospital/register', methods=['GET', 'POST'])
def hospital_register():
    if request.method == 'POST':
        name = request.form['name']
        reg_no = request.form['reg_no']
        email = request.form['email']
        password = request.form['password']
        state = request.form.get('state')
        district = request.form.get('district')

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                'INSERT INTO hospitals (name, reg_no, email, password, state, district) VALUES (?, ?, ?, ?, ?, ?)',
                (name, reg_no, email, password, state, district),
            )
            conn.commit()
            flash('Hospital registered successfully. Please login.', 'success')
            return redirect(url_for('hospital_login'))
        except sqlite3.IntegrityError:
            flash('Hospital with this email or registration number already exists.', 'danger')
        finally:
            conn.close()

    return render_template('hospital_register.html')


@app.route('/hospital/login', methods=['GET', 'POST'])
def hospital_login():
    if request.method == 'POST':
        login_type = request.form.get('login_type', 'password')
        print(f"[DEBUG] Hospital login - login_type: {login_type}, form data: {dict(request.form)}")
        
        if login_type == 'otp':
            # OTP login
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            otp_code = request.form.get('otp_code', '').strip()
            
            if not email or not phone:
                flash('Email and phone number are required', 'danger')
                return render_template('hospital_login.html', login_type='otp')
            
            # Normalize phone number
            normalized_phone = normalize_phone(phone)
            if not normalized_phone:
                flash('Invalid phone number format', 'danger')
                return render_template('hospital_login.html', login_type='otp', email=email, phone=phone)
            
            if not otp_code:
                # Send OTP
                conn = get_db_connection()
                cur = conn.cursor()
                
                # First check by email only
                cur.execute('SELECT * FROM hospitals WHERE email = ?', (email,))
                hospital = cur.fetchone()
                
                if not hospital:
                    conn.close()
                    flash('Email not found', 'danger')
                    return render_template('hospital_login.html', login_type='otp', email=email, phone=phone)
                
                # Check if phone matches (normalize both)
                hospital_phone = normalize_phone(hospital['phone']) if hospital['phone'] else None
                
                if not hospital_phone:
                    conn.close()
                    flash('Phone number not registered. Please use password login or update your profile.', 'danger')
                    return render_template('hospital_login.html', login_type='otp', email=email, phone=phone)
                
                if hospital_phone != normalized_phone:
                    conn.close()
                    flash(f'Phone number does not match. Registered phone ends with: ...{hospital_phone[-4:]}', 'danger')
                    return render_template('hospital_login.html', login_type='otp', email=email, phone=phone)
                
                conn.close()
                
                # Use normalized phone for OTP
                success, message = send_otp(normalized_phone, 'hospital', email, 'login')
                if success:
                    flash(message, 'success')  # Show OTP in message
                    # Extract OTP code from message for display
                    import re
                    otp_match = re.search(r'(\d{6})', message)
                    otp_code_display = otp_match.group(1) if otp_match else None
                    return render_template('hospital_login.html', login_type='otp', email=email, phone=phone, otp_sent=True, otp_code=otp_code_display, otp_message=message)
                else:
                    flash(message, 'danger')
                return render_template('hospital_login.html', login_type='otp', email=email, phone=phone)
            else:
                # Verify OTP (use normalized phone)
                normalized_phone = normalize_phone(phone)
                success, message = verify_otp(normalized_phone, otp_code, 'hospital', email, 'login')
                if success:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute('SELECT * FROM hospitals WHERE email = ?', (email,))
                    hospital = cur.fetchone()
                    conn.close()
                    
                    if hospital:
                        login_user('hospital', hospital['id'])
                        flash('Logged in successfully', 'success')
                        return redirect(url_for('hospital_dashboard'))
                else:
                    flash(message, 'danger')
                    return render_template('hospital_login.html', login_type='otp', email=email, phone=phone, otp_sent=True)
        else:
            # Password login
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()

            if not email or not password:
                flash('Email and password are required', 'danger')
                return render_template('hospital_login.html', login_type='password')

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM hospitals WHERE email = ? AND password = ?', (email, password))
            hospital = cur.fetchone()
            conn.close()

            if hospital:
                login_user('hospital', hospital['id'])
                flash('Logged in successfully', 'success')
                return redirect(url_for('hospital_dashboard'))
            else:
                flash('Invalid email or password', 'danger')
                return render_template('hospital_login.html', login_type='password', email=email)

    return render_template('hospital_login.html')


@app.route('/hospital/profile', methods=['GET', 'POST'])
def hospital_profile():
    if current_role() != 'hospital':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))
    
    hospital_id = current_user_id()
    conn = get_db_connection()
    cur = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        state = request.form.get('state', '').strip() or None
        district = request.form.get('district', '').strip() or None
        
        # Check if email is already taken by another hospital
        cur.execute('SELECT id FROM hospitals WHERE email = ? AND id != ?', (email, hospital_id))
        if cur.fetchone():
            conn.close()
            flash('Email already exists. Please use a different email.', 'danger')
            cur.execute('SELECT * FROM hospitals WHERE id = ?', (hospital_id,))
            hospital = cur.fetchone()
            conn.close()
            return render_template('hospital_profile.html', hospital=hospital)
        
        # Update hospital profile
        try:
            cur.execute('''
                UPDATE hospitals 
                SET name = ?, email = ?, phone = ?, state = ?, district = ?
                WHERE id = ?
            ''', (name, email, phone, state, district, hospital_id))
            conn.commit()
            flash('Profile updated successfully', 'success')
            conn.close()
            return redirect(url_for('hospital_profile'))
        except Exception as e:
            conn.rollback()
            conn.close()
            flash(f'Error updating profile: {str(e)}', 'danger')
            return redirect(url_for('hospital_profile'))
    
    # GET request - show current profile
    cur.execute('SELECT * FROM hospitals WHERE id = ?', (hospital_id,))
    hospital = cur.fetchone()
    conn.close()
    
    return render_template('hospital_profile.html', hospital=hospital)


@app.route('/hospital/forgot_password', methods=['GET', 'POST'])
def hospital_forgot_password():
    if request.method == 'POST':
        step = request.form.get('step', 'request')
        
        if step == 'request':
            # Request OTP
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            
            if not email or not phone:
                flash('Email and phone number are required', 'danger')
                return render_template('forgot_password.html', role='hospital')
            
            # Normalize phone number
            normalized_phone = normalize_phone(phone)
            if not normalized_phone:
                flash('Invalid phone number format', 'danger')
                return render_template('forgot_password.html', role='hospital')
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM hospitals WHERE email = ?', (email,))
            hospital = cur.fetchone()
            
            if not hospital:
                conn.close()
                flash('Email not found', 'danger')
                return render_template('forgot_password.html', role='hospital')
            
            # Check if phone matches (normalize both)
            hospital_phone = normalize_phone(hospital['phone']) if hospital['phone'] else None
            
            if not hospital_phone:
                conn.close()
                flash('Phone number not registered. Please update your profile or use password login.', 'danger')
                return render_template('forgot_password.html', role='hospital')
            
            if hospital_phone != normalized_phone:
                conn.close()
                flash(f'Phone number does not match. Registered phone ends with: ...{hospital_phone[-4:]}', 'danger')
                return render_template('forgot_password.html', role='hospital')
            
            conn.close()
            
            success, message = send_otp(normalized_phone, 'hospital', email, 'reset')
            if success:
                flash(message, 'success')  # Show OTP in message
                return render_template('forgot_password.html', role='hospital', step='verify', email=email, phone=phone)
            else:
                flash(message, 'danger')
        
        elif step == 'verify':
            # Verify OTP and reset password
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            otp_code = request.form.get('otp_code', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            
            if not otp_code or not new_password or not confirm_password:
                flash('All fields are required', 'danger')
                return render_template('forgot_password.html', role='hospital', step='verify', email=email, phone=phone)
            
            if new_password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('forgot_password.html', role='hospital', step='verify', email=email, phone=phone)
            
            # Verify OTP (use normalized phone)
            normalized_phone = normalize_phone(phone)
            success, message = verify_otp(normalized_phone, otp_code, 'hospital', email, 'reset')
            if success:
                # Update password
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute('UPDATE hospitals SET password = ? WHERE email = ?', (new_password, email))
                conn.commit()
                conn.close()
                flash('Password reset successfully. Please login.', 'success')
                return redirect(url_for('hospital_login'))
            else:
                flash(message, 'danger')
                return render_template('forgot_password.html', role='hospital', step='verify', email=email, phone=phone)
    
    return render_template('forgot_password.html', role='hospital')


@app.route('/hospital/delete_doctor/<int:doctor_id>', methods=['POST'])
def delete_doctor(doctor_id):
    if current_role() != 'hospital':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    hospital_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure this doctor belongs to the logged-in hospital
    cur.execute('SELECT * FROM doctors WHERE id = ? AND hospital_id = ?', (doctor_id, hospital_id))
    doctor = cur.fetchone()
    if not doctor:
        conn.close()
        flash('Doctor not found for this hospital.', 'warning')
        return redirect(url_for('hospital_dashboard'))

    # Prevent deletion if doctor has existing medical records
    cur.execute('SELECT COUNT(*) AS c FROM records WHERE doctor_id = ?', (doctor_id,))
    count = cur.fetchone()['c']
    if count > 0:
        conn.close()
        flash('Cannot delete doctor with existing medical records.', 'warning')
        return redirect(url_for('hospital_dashboard'))

    cur.execute('DELETE FROM doctors WHERE id = ? AND hospital_id = ?', (doctor_id, hospital_id))
    conn.commit()
    conn.close()

    flash('Doctor deleted successfully.', 'success')
    return redirect(url_for('hospital_dashboard'))


@app.route('/hospital/logout')
def hospital_logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))


@app.route('/hospital/dashboard')
def hospital_dashboard():
    if current_role() != 'hospital':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    hospital_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Load basic hospital info (for location display)
    # Be defensive in case the existing DB was created before state/district columns were added.
    try:
        cur.execute('SELECT state, district FROM hospitals WHERE id = ?', (hospital_id,))
        hospital = cur.fetchone()
    except sqlite3.OperationalError:
        # Fallback: no such columns in this DB; keep hospital minimal so template can still render.
        hospital = {'state': None, 'district': None}

    # Count doctors for this hospital
    cur.execute('SELECT COUNT(*) AS c FROM doctors WHERE hospital_id = ?', (hospital_id,))
    doctors_count = cur.fetchone()['c']

    # Count distinct patients treated by this hospital's doctors
    cur.execute(
        '''SELECT COUNT(DISTINCT r.user_id) AS c
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE d.hospital_id = ?''',
        (hospital_id,),
    )
    patients_count = cur.fetchone()['c']

    # Count emergency cases (demo: all emergencies)
    cur.execute('SELECT COUNT(*) AS c FROM emergencies')
    emergency_count = cur.fetchone()['c']

    # Get doctor list
    cur.execute('SELECT * FROM doctors WHERE hospital_id = ?', (hospital_id,))
    doctors = cur.fetchall()

    # Get patients treated by this hospital's doctors with visit count and last visit
    cur.execute(
        '''SELECT u.id AS user_id, u.name, u.health_id, COUNT(r.id) AS visits,
                  MAX(r.date) AS last_visit
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           JOIN users u ON r.user_id = u.id
           WHERE d.hospital_id = ?
           GROUP BY u.id, u.name, u.health_id
           ORDER BY last_visit DESC''',
        (hospital_id,),
    )
    patients = cur.fetchall()

    # Get all medical records for this hospital's doctors (Visit History)
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization, u.name as patient_name
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           JOIN users u ON r.user_id = u.id
           WHERE d.hospital_id = ?
           ORDER BY r.date DESC
           LIMIT 50''',
        (hospital_id,),
    )
    records = cur.fetchall()

    conn.close()

    return render_template(
        'hospital_dashboard.html',
        hospital=hospital,
        doctors_count=doctors_count,
        patients_count=patients_count,
        emergency_count=emergency_count,
        doctors=doctors,
        patients=patients,
        records=records,
    )


@app.route('/hospital/patient/<int:user_id>')
def hospital_patient_detail(user_id):
    if current_role() != 'hospital':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    hospital_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure patient exists
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        flash('Patient not found.', 'warning')
        return redirect(url_for('hospital_dashboard'))

    # Fetch records for this patient but only with doctors from this hospital
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE r.user_id = ? AND d.hospital_id = ?
           ORDER BY r.date DESC''',
        (user_id, hospital_id),
    )
    records = cur.fetchall()

    conn.close()

    # Optionally, we could hide patients that have no records with this hospital
    print_mode = request.args.get('print') == '1'
    return render_template('hospital_patient_detail.html', patient=patient, records=records, print_mode=print_mode)


@app.route('/hospital/add_doctor', methods=['POST'])
def add_doctor():
    if current_role() != 'hospital':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    hospital_id = current_user_id()
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    specialization = request.form.get('specialization', '')

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO doctors (hospital_id, name, email, password, specialization) VALUES (?, ?, ?, ?, ?)',
            (hospital_id, name, email, password, specialization),
        )
        conn.commit()
        flash('Doctor added successfully.', 'success')
    except sqlite3.IntegrityError:
        flash('Doctor with this email already exists.', 'danger')
    finally:
        conn.close()

    return redirect(url_for('hospital_dashboard'))


# -----------------
# Doctor auth & dashboard
# -----------------


@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        login_type = request.form.get('login_type', 'password')
        print(f"[DEBUG] Doctor login - login_type: {login_type}, form data: {dict(request.form)}")
        
        if login_type == 'otp':
            # OTP login
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            otp_code = request.form.get('otp_code', '').strip()
            
            if not email or not phone:
                flash('Email and phone number are required', 'danger')
                return render_template('doctor_login.html', login_type='otp')
            
            # Normalize phone number
            normalized_phone = normalize_phone(phone)
            if not normalized_phone:
                flash('Invalid phone number format', 'danger')
                return render_template('doctor_login.html', login_type='otp', email=email, phone=phone)
            
            if not otp_code:
                # Send OTP
                conn = get_db_connection()
                cur = conn.cursor()
                
                # First check by email only
                cur.execute('SELECT * FROM doctors WHERE email = ?', (email,))
                doctor = cur.fetchone()
                
                if not doctor:
                    conn.close()
                    flash('Email not found', 'danger')
                    return render_template('doctor_login.html', login_type='otp', email=email, phone=phone)
                
                # Check if phone matches (normalize both)
                doctor_phone = normalize_phone(doctor['phone']) if doctor['phone'] else None
                
                if not doctor_phone:
                    conn.close()
                    flash('Phone number not registered. Please use password login or contact your hospital admin.', 'danger')
                    return render_template('doctor_login.html', login_type='otp', email=email, phone=phone)
                
                if doctor_phone != normalized_phone:
                    conn.close()
                    flash(f'Phone number does not match. Registered phone ends with: ...{doctor_phone[-4:]}', 'danger')
                    return render_template('doctor_login.html', login_type='otp', email=email, phone=phone)
                
                conn.close()
                
                # Use normalized phone for OTP
                success, message = send_otp(normalized_phone, 'doctor', email, 'login')
                if success:
                    flash(message, 'success')  # Show OTP in message
                    # Extract OTP code from message for display
                    otp_match = re.search(r'(\d{6})', message)
                    otp_code_display = otp_match.group(1) if otp_match else None
                    return render_template('doctor_login.html', login_type='otp', email=email, phone=phone, otp_sent=True, otp_code=otp_code_display, otp_message=message)
                else:
                    flash(message, 'danger')
                return render_template('doctor_login.html', login_type='otp', email=email, phone=phone)
            else:
                # Verify OTP (use normalized phone)
                normalized_phone = normalize_phone(phone)
                success, message = verify_otp(normalized_phone, otp_code, 'doctor', email, 'login')
                if success:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute('SELECT * FROM doctors WHERE email = ?', (email,))
                    doctor = cur.fetchone()
                    conn.close()
                    
                    if doctor:
                        login_user('doctor', doctor['id'])
                        flash('Logged in successfully', 'success')
                        return redirect(url_for('doctor_dashboard'))
                else:
                    flash(message, 'danger')
                    return render_template('doctor_login.html', login_type='otp', email=email, phone=phone, otp_sent=True)
        else:
            # Password login
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()

            if not email or not password:
                flash('Email and password are required', 'danger')
                return render_template('doctor_login.html', login_type='password')

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM doctors WHERE email = ? AND password = ?', (email, password))
            doctor = cur.fetchone()
            conn.close()

            if doctor:
                login_user('doctor', doctor['id'])
                flash('Logged in successfully', 'success')
                return redirect(url_for('doctor_dashboard'))
            else:
                flash('Invalid email or password', 'danger')
                return render_template('doctor_login.html', login_type='password', email=email)

    return render_template('doctor_login.html')


@app.route('/doctor/forgot_password', methods=['GET', 'POST'])
def doctor_forgot_password():
    if request.method == 'POST':
        step = request.form.get('step', 'request')
        
        if step == 'request':
            # Request OTP
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            
            if not email or not phone:
                flash('Email and phone number are required', 'danger')
                return render_template('forgot_password.html', role='doctor')
            
            # Normalize phone number
            normalized_phone = normalize_phone(phone)
            if not normalized_phone:
                flash('Invalid phone number format', 'danger')
                return render_template('forgot_password.html', role='doctor')
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM doctors WHERE email = ?', (email,))
            doctor = cur.fetchone()
            
            if not doctor:
                conn.close()
                flash('Email not found', 'danger')
                return render_template('forgot_password.html', role='doctor')
            
            # Check if phone matches (normalize both)
            doctor_phone = normalize_phone(doctor['phone']) if doctor['phone'] else None
            
            if not doctor_phone:
                conn.close()
                flash('Phone number not registered. Please contact your hospital admin.', 'danger')
                return render_template('forgot_password.html', role='doctor')
            
            if doctor_phone != normalized_phone:
                conn.close()
                flash(f'Phone number does not match. Registered phone ends with: ...{doctor_phone[-4:]}', 'danger')
                return render_template('forgot_password.html', role='doctor')
            
            conn.close()
            
            success, message = send_otp(normalized_phone, 'doctor', email, 'reset')
            if success:
                flash(message, 'success')  # Show OTP in message
                return render_template('forgot_password.html', role='doctor', step='verify', email=email, phone=phone)
            else:
                flash(message, 'danger')
        
        elif step == 'verify':
            # Verify OTP and reset password
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            otp_code = request.form.get('otp_code', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            
            if not otp_code or not new_password or not confirm_password:
                flash('All fields are required', 'danger')
                return render_template('forgot_password.html', role='doctor', step='verify', email=email, phone=phone)
            
            if new_password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('forgot_password.html', role='doctor', step='verify', email=email, phone=phone)
            
            # Verify OTP (use normalized phone)
            normalized_phone = normalize_phone(phone)
            success, message = verify_otp(normalized_phone, otp_code, 'doctor', email, 'reset')
            if success:
                # Update password
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute('UPDATE doctors SET password = ? WHERE email = ?', (new_password, email))
                conn.commit()
                conn.close()
                flash('Password reset successfully. Please login.', 'success')
                return redirect(url_for('doctor_login'))
            else:
                flash(message, 'danger')
                return render_template('forgot_password.html', role='doctor', step='verify', email=email, phone=phone)
    
    return render_template('forgot_password.html', role='doctor')


@app.route('/doctor/logout')
def doctor_logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))


@app.route('/doctor/profile', methods=['GET', 'POST'])
def doctor_profile():
    if current_role() != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))
    
    doctor_id = current_user_id()
    conn = get_db_connection()
    cur = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        specialization = request.form.get('specialization', '').strip() or None
        
        # Check if email is already taken by another doctor
        cur.execute('SELECT id FROM doctors WHERE email = ? AND id != ?', (email, doctor_id))
        if cur.fetchone():
            conn.close()
            flash('Email already exists. Please use a different email.', 'danger')
            cur.execute('SELECT * FROM doctors WHERE id = ?', (doctor_id,))
            doctor = cur.fetchone()
            conn.close()
            return render_template('doctor_profile.html', doctor=doctor)
        
        # Update doctor profile
        try:
            cur.execute('''
                UPDATE doctors 
                SET name = ?, email = ?, phone = ?, specialization = ?
                WHERE id = ?
            ''', (name, email, phone, specialization, doctor_id))
            conn.commit()
            flash('Profile updated successfully', 'success')
            conn.close()
            return redirect(url_for('doctor_profile'))
        except Exception as e:
            conn.rollback()
            conn.close()
            flash(f'Error updating profile: {str(e)}', 'danger')
            return redirect(url_for('doctor_profile'))
    
    # GET request - show current profile
    cur.execute('SELECT * FROM doctors WHERE id = ?', (doctor_id,))
    doctor = cur.fetchone()
    conn.close()
    
    return render_template('doctor_profile.html', doctor=doctor)


@app.route('/doctor/dashboard', methods=['GET', 'POST'])
def doctor_dashboard():
    if current_role() != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    conn = get_db_connection()
    cur = conn.cursor()

    doctor_id = current_user_id()

    patient = None
    records = []

    # Calculate statistics for this doctor
    cur.execute(
        '''SELECT COUNT(DISTINCT user_id) as unique_patients FROM records WHERE doctor_id = ?''',
        (doctor_id,)
    )
    unique_patients_result = cur.fetchone()
    unique_patients_count = unique_patients_result['unique_patients'] if unique_patients_result else 0

    cur.execute(
        '''SELECT COUNT(*) as total_records FROM records WHERE doctor_id = ?''',
        (doctor_id,)
    )
    total_records_result = cur.fetchone()
    total_records_count = total_records_result['total_records'] if total_records_result else 0

    cur.execute(
        '''SELECT COUNT(*) as recovered FROM records WHERE doctor_id = ? AND treatment_status = 'Recovered' ''',
        (doctor_id,)
    )
    recovered_result = cur.fetchone()
    recovered_count = recovered_result['recovered'] if recovered_result else 0

    cur.execute(
        '''SELECT COUNT(*) as observation FROM records WHERE doctor_id = ? AND treatment_status = 'Under Observation' ''',
        (doctor_id,)
    )
    observation_result = cur.fetchone()
    observation_count = observation_result['observation'] if observation_result else 0

    # Handle patient search by Health ID (ABHA-like)
    search_health_id = None
    if request.method == 'POST' and 'search_health_id' in request.form:
        search_health_id = request.form['search_health_id'].strip()
        cur.execute('SELECT * FROM users WHERE health_id = ?', (search_health_id,))
        patient = cur.fetchone()
        if patient:
            cur.execute(
                '''SELECT r.*, d.id as doctor_id, d.name as doctor_name, d.specialization as doctor_specialization,
                          h.id as hospital_id, h.name as hospital_name, h.reg_no as hospital_reg_no
                   FROM records r
                   JOIN doctors d ON r.doctor_id = d.id
                   JOIN hospitals h ON d.hospital_id = h.id
                   WHERE r.user_id = ?
                   ORDER BY r.date DESC''',
                (patient['id'],),
            )
            records = cur.fetchall()
        else:
            flash('No patient found with that Health ID.', 'warning')

    conn.close()

    return render_template(
        'doctor_dashboard.html',
        patient=patient,
        records=records,
        search_health_id=search_health_id,
        doctor_id=doctor_id,
        unique_patients_count=unique_patients_count,
        total_records_count=total_records_count,
        recovered_count=recovered_count,
        observation_count=observation_count,
        current_user_id=doctor_id,
    )


@app.route('/doctor/patient/<int:user_id>/export/csv')
def doctor_patient_export_csv(user_id):
    """Export all visits for a given patient with the current doctor as CSV."""
    if current_role() != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    doctor_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure patient exists
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        flash('Patient not found.', 'warning')
        return redirect(url_for('doctor_dashboard'))

    # Records for this patient with this doctor
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE r.user_id = ? AND r.doctor_id = ?
           ORDER BY r.date DESC''',
        (user_id, doctor_id),
    )
    records = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Date',
        'Doctor Name',
        'Doctor Specialization',
        'Diagnosis',
        'Medicines',
        'Dosage',
        'Treatment Status',
        'Prescription Text',
    ])

    for r in records:
        # Use .keys() checks to be safe with older rows that may miss new columns
        dosage_val = r['dosage'] if 'dosage' in r.keys() and r['dosage'] is not None else ''
        prescription_val = (
            r['prescription_text']
            if 'prescription_text' in r.keys() and r['prescription_text'] is not None
            else ''
        )
        status_val = (
            r['treatment_status']
            if 'treatment_status' in r.keys() and r['treatment_status'] is not None
            else ''
        )

        writer.writerow([
            r['date'],
            r['doctor_name'],
            r['doctor_specialization'],
            r['diagnosis'] or '',
            r['medicines'] or '',
            dosage_val,
            status_val,
            prescription_val,
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"doctor_patient_{user_id}_visits.csv"
    response = Response(csv_data, mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


@app.route('/doctor/add_record/<int:user_id>', methods=['POST'])
def add_record(user_id):
    if current_role() != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    doctor_id = current_user_id()

    date = request.form['date']
    symptoms = request.form.get('symptoms', '')
    diagnosis = request.form.get('diagnosis', '')
    medicines = request.form.get('medicines', '')
    dosage = request.form.get('dosage', '')
    treatment_status = request.form.get('treatment_status', '')

    # Consultation duration stored in minutes; validate numeric input
    consultation_duration_raw = request.form.get('consultation_duration', '').strip()
    consultation_duration = int(consultation_duration_raw) if consultation_duration_raw.isdigit() else None

    prescription_text = request.form.get('prescription_text', '')
    
    # Health metrics for risk prediction
    age = request.form.get('age', '').strip()
    age = int(age) if age and age.isdigit() else None
    
    gender = request.form.get('gender', '').strip() or None
    
    systolic_bp = request.form.get('systolic_bp', '').strip()
    systolic_bp = int(systolic_bp) if systolic_bp and systolic_bp.isdigit() else None
    
    diastolic_bp = request.form.get('diastolic_bp', '').strip()
    diastolic_bp = int(diastolic_bp) if diastolic_bp and diastolic_bp.isdigit() else None
    
    bmi = request.form.get('bmi', '').strip()
    bmi = float(bmi) if bmi and bmi.replace('.', '').isdigit() else None
    
    cholesterol = request.form.get('cholesterol', '').strip()
    cholesterol = float(cholesterol) if cholesterol and cholesterol.replace('.', '').isdigit() else None
    
    glucose = request.form.get('glucose', '').strip()
    glucose = float(glucose) if glucose and glucose.replace('.', '').isdigit() else None
    
    smoking = request.form.get('smoking', '').strip() or None
    alcohol = request.form.get('alcohol', '').strip() or None
    physical_activity = request.form.get('physical_activity', '').strip() or None
    family_history = request.form.get('family_history', '').strip() or None

    # File uploads: blood report and optional prescription file
    blood_report_file = request.files.get('blood_report_file')
    prescription_file = request.files.get('prescription_file')

    blood_report_filename = None
    if blood_report_file and blood_report_file.filename:
        blood_report_filename = f"blood_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{blood_report_file.filename}"
        blood_report_path = os.path.join(app.config['UPLOAD_FOLDER'], blood_report_filename)
        blood_report_file.save(blood_report_path)

    prescription_filename = None
    if prescription_file and prescription_file.filename:
        prescription_filename = f"presc_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{prescription_file.filename}"
        prescription_path = os.path.join(app.config['UPLOAD_FOLDER'], prescription_filename)
        prescription_file.save(prescription_path)

    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get user data for health risk prediction
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user_data = cur.fetchone()
    
    # Create health metrics dict for prediction
    health_metrics = {
        'age': age,
        'gender': gender,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'bmi': bmi,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'smoking': smoking,
        'alcohol': alcohol,
        'physical_activity': physical_activity,
        'family_history': family_history,
    }
    
    # Predict health risk using AI model with health metrics
    risk_level, risk_score, should_trigger_emergency = predict_health_risk(
        user_data, symptoms, diagnosis, treatment_status, medicines, health_metrics
    )
    
    # Insert medical record with risk prediction and health metrics
    # Note: report_filename is kept for backward compatibility, using blood_report_filename value
    cur.execute(
        '''INSERT INTO records
           (user_id, doctor_id, date, symptoms, diagnosis, medicines, dosage, treatment_status,
            consultation_duration, prescription_text, prescription_filename, blood_report_filename,
            report_filename, created_at, risk_level, risk_score,
            systolic_bp, diastolic_bp, bmi, cholesterol, glucose, smoking, alcohol, physical_activity, family_history)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            user_id,
            doctor_id,
            date,
            symptoms,
            diagnosis,
            medicines,
            dosage,
            treatment_status,
            consultation_duration,
            prescription_text,
            prescription_filename,
            blood_report_filename,
            blood_report_filename,  # report_filename for backward compatibility
            datetime.utcnow().isoformat(),
            risk_level,
            risk_score if risk_score is not None else None,
            systolic_bp,
            diastolic_bp,
            bmi,
            cholesterol,
            glucose,
            smoking,
            alcohol,
            physical_activity,
            family_history,
        ),
    )
    conn.commit()
    
    # If high risk predicted, automatically create emergency record
    emergency_created = False
    if should_trigger_emergency and user_data:
        try:
            # Get patient location (use address or default)
            # Handle both dict and Row objects
            if hasattr(user_data, 'get'):
                patient_location = user_data.get('address', 'Location not specified')
                patient_name = user_data.get('name', 'Patient')
                patient_phone = user_data.get('phone', 'Not provided')
            else:
                patient_location = user_data['address'] if 'address' in user_data.keys() and user_data['address'] else 'Location not specified'
                patient_name = user_data['name'] if 'name' in user_data.keys() and user_data['name'] else 'Patient'
                patient_phone = user_data['phone'] if 'phone' in user_data.keys() and user_data['phone'] else 'Not provided'
            
            cur.execute(
                '''INSERT INTO emergencies (user_id, name, phone, location, status, requested_at, response_time_minutes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (
                    user_id,
                    patient_name,
                    patient_phone,
                    f"{patient_location} (Auto-triggered by AI Risk Assessment)",
                    'Ambulance Dispatched',
                    datetime.utcnow().isoformat(),
                    10,  # Faster response for AI-detected emergencies
                ),
            )
            conn.commit()
            emergency_created = True
        except Exception as e:
            print(f"Error creating emergency record: {e}")
    
    conn.close()
    
    # Flash messages based on risk assessment
    if emergency_created:
        flash(f'⚠️ HIGH RISK DETECTED! Record added. Emergency ambulance automatically dispatched. Risk Level: {risk_level} (Score: {risk_score:.2f})', 'danger')
    elif risk_level:
        if risk_level in ['High', 'Critical']:
            flash(f'⚠️ Record added. High risk detected (Level: {risk_level}, Score: {risk_score:.2f}). Please monitor patient closely.', 'warning')
        else:
            flash(f'Record added successfully. Risk Assessment: {risk_level} (Score: {risk_score:.2f})', 'success')
    else:
        flash('Record added successfully.', 'success')
    
    return redirect(url_for('doctor_dashboard'))


# -----------------
# User auth & dashboard
# -----------------


@app.route('/user/register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        age_raw = request.form.get('age', '').strip()
        phone = request.form['phone']
        address = request.form['address']

        # Basic numeric validation for age
        age = int(age_raw) if age_raw.isdigit() else None

        health_id = generate_health_id()

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                '''INSERT INTO users (name, email, password, phone, address, health_id, age)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (name, email, password, phone, address, health_id, age),
            )
            conn.commit()
            user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            conn.close()
            flash('User with this email already exists.', 'danger')
            return render_template('user_register.html')

        conn.close()

        login_user('user', user_id)
        flash('Registration successful.', 'success')
        return redirect(url_for('user_dashboard'))

    return render_template('user_register.html')


@app.route('/user/login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        login_type = request.form.get('login_type', 'password')
        print(f"[DEBUG] User login - login_type: {login_type}, form data: {dict(request.form)}")
        
        if login_type == 'otp':
            # OTP login
            identifier = request.form.get('identifier', '').strip()
            phone = request.form.get('phone', '').strip()
            otp_code = request.form.get('otp_code', '').strip()
            
            if not identifier or not phone:
                flash('Health ID and phone number are required', 'danger')
                return render_template('user_login.html', login_type='otp')
            
            # Normalize phone number
            normalized_phone = normalize_phone(phone)
            if not normalized_phone:
                flash('Invalid phone number format', 'danger')
                return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone)
            
            if not otp_code:
                # Send OTP
                conn = get_db_connection()
                cur = conn.cursor()
                
                # First check by health_id only
                cur.execute('SELECT * FROM users WHERE health_id = ?', (identifier,))
                user = cur.fetchone()
                
                if not user:
                    conn.close()
                    flash('Health ID not found', 'danger')
                    return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone)
                
                # Check if phone matches (normalize both)
                user_phone = normalize_phone(user['phone']) if user['phone'] else None
                
                if not user_phone:
                    conn.close()
                    flash('Phone number not registered. Please use password login or contact support.', 'danger')
                    return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone)
                
                if user_phone != normalized_phone:
                    conn.close()
                    flash(f'Phone number does not match. Registered phone ends with: ...{user_phone[-4:]}', 'danger')
                    return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone)
                
                conn.close()
                
                # Use normalized phone for OTP
                success, message = send_otp(normalized_phone, 'user', identifier, 'login')
                if success:
                    flash(message, 'success')  # Show OTP in message
                    # Extract OTP code from message for display
                    otp_match = re.search(r'(\d{6})', message)
                    otp_code_display = otp_match.group(1) if otp_match else None
                    return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone, otp_sent=True, otp_code=otp_code_display, otp_message=message)
                else:
                    flash(message, 'danger')
                return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone)
            else:
                # Verify OTP (use normalized phone)
                normalized_phone = normalize_phone(phone)
                success, message = verify_otp(normalized_phone, otp_code, 'user', identifier, 'login')
                if success:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute('SELECT * FROM users WHERE health_id = ?', (identifier,))
                    user = cur.fetchone()
                    conn.close()
                    
                    if user:
                        login_user('user', user['id'])
                        flash('Logged in successfully', 'success')
                        return redirect(url_for('user_dashboard'))
                else:
                    flash(message, 'danger')
                    return render_template('user_login.html', login_type='otp', identifier=identifier, phone=phone, otp_sent=True)
        else:
            # Password login
            identifier = request.form.get('identifier', '').strip()
            password = request.form.get('password', '').strip()

            if not identifier or not password:
                flash('Health ID and password are required', 'danger')
                return render_template('user_login.html', login_type='password')

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM users WHERE health_id = ? AND password = ?', (identifier, password))
            user = cur.fetchone()
            conn.close()

            if user:
                login_user('user', user['id'])
                flash('Logged in successfully', 'success')
                return redirect(url_for('user_dashboard'))
            else:
                flash('Invalid Health ID or password', 'danger')
                return render_template('user_login.html', login_type='password', identifier=identifier)

    return render_template('user_login.html')


@app.route('/user/forgot_password', methods=['GET', 'POST'])
def user_forgot_password():
    if request.method == 'POST':
        step = request.form.get('step', 'request')
        
        if step == 'request':
            # Request OTP
            identifier = request.form.get('identifier', '').strip()
            phone = request.form.get('phone', '').strip()
            
            if not identifier or not phone:
                flash('Health ID and phone number are required', 'danger')
                return render_template('forgot_password.html', role='user')
            
            # Normalize phone number
            normalized_phone = normalize_phone(phone)
            if not normalized_phone:
                flash('Invalid phone number format', 'danger')
                return render_template('forgot_password.html', role='user')
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM users WHERE health_id = ?', (identifier,))
            user = cur.fetchone()
            
            if not user:
                conn.close()
                flash('Health ID not found', 'danger')
                return render_template('forgot_password.html', role='user')
            
            # Check if phone matches (normalize both)
            user_phone = normalize_phone(user['phone']) if user['phone'] else None
            
            if not user_phone:
                conn.close()
                flash('Phone number not registered. Please contact support.', 'danger')
                return render_template('forgot_password.html', role='user')
            
            if user_phone != normalized_phone:
                conn.close()
                flash(f'Phone number does not match. Registered phone ends with: ...{user_phone[-4:]}', 'danger')
                return render_template('forgot_password.html', role='user')
            
            conn.close()
            
            success, message = send_otp(normalized_phone, 'user', identifier, 'reset')
            if success:
                flash(message, 'success')  # Show OTP in message
                return render_template('forgot_password.html', role='user', step='verify', identifier=identifier, phone=phone)
            else:
                flash(message, 'danger')
        
        elif step == 'verify':
            # Verify OTP and reset password
            identifier = request.form.get('identifier', '').strip()
            phone = request.form.get('phone', '').strip()
            otp_code = request.form.get('otp_code', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            
            if not otp_code or not new_password or not confirm_password:
                flash('All fields are required', 'danger')
                return render_template('forgot_password.html', role='user', step='verify', identifier=identifier, phone=phone)
            
            if new_password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('forgot_password.html', role='user', step='verify', identifier=identifier, phone=phone)
            
            # Verify OTP
            success, message = verify_otp(phone, otp_code, 'user', identifier, 'reset')
            if success:
                # Update password
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute('UPDATE users SET password = ? WHERE health_id = ? AND phone = ?', (new_password, identifier, phone))
                conn.commit()
                conn.close()
                flash('Password reset successfully. Please login.', 'success')
                return redirect(url_for('user_login'))
            else:
                flash(message, 'danger')
                return render_template('forgot_password.html', role='user', step='verify', identifier=identifier, phone=phone)
    
    return render_template('forgot_password.html', role='user')


@app.route('/user/profile', methods=['GET', 'POST'])
def user_profile():
    if current_role() != 'user':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))
    
    user_id = current_user_id()
    conn = get_db_connection()
    cur = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        age_raw = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip() or None
        
        age = int(age_raw) if age_raw and age_raw.isdigit() else None
        
        # Check if email is already taken by another user
        cur.execute('SELECT id FROM users WHERE email = ? AND id != ?', (email, user_id))
        if cur.fetchone():
            conn.close()
            flash('Email already exists. Please use a different email.', 'danger')
            cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = cur.fetchone()
            conn.close()
            return render_template('user_profile.html', user=user)
        
        # Update user profile
        try:
            cur.execute('''
                UPDATE users 
                SET name = ?, email = ?, phone = ?, address = ?, age = ?, gender = ?
                WHERE id = ?
            ''', (name, email, phone, address, age, gender, user_id))
            conn.commit()
            flash('Profile updated successfully', 'success')
            conn.close()
            return redirect(url_for('user_profile'))
        except Exception as e:
            conn.rollback()
            conn.close()
            flash(f'Error updating profile: {str(e)}', 'danger')
            return redirect(url_for('user_profile'))
    
    # GET request - show current profile
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cur.fetchone()
    conn.close()
    
    return render_template('user_profile.html', user=user)


@app.route('/user/logout')
def user_logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))


@app.route('/user/dashboard')
def user_dashboard():
    if current_role() != 'user':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    user_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cur.fetchone()

    # Full medical records with doctor info
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE r.user_id = ?
           ORDER BY r.date DESC''',
        (user_id,),
    )
    records = cur.fetchall()

    # Distinct hospitals that have treated this user
    cur.execute(
        '''SELECT DISTINCT h.id AS hospital_id, h.name
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           JOIN hospitals h ON d.hospital_id = h.id
           WHERE r.user_id = ?
           ORDER BY h.name''',
        (user_id,),
    )
    hospitals = cur.fetchall()

    conn.close()

    # Ensure QR exists
    qr_rel_path = generate_health_qr(user['health_id'])

    return render_template('user_dashboard.html', user=user, records=records, hospitals=hospitals, qr_path=qr_rel_path)


@app.route('/user/hospital/<int:hospital_id>')
def user_hospital_detail(hospital_id):
    if current_role() != 'user':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    user_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Load hospital
    cur.execute('SELECT * FROM hospitals WHERE id = ?', (hospital_id,))
    hospital = cur.fetchone()
    if not hospital:
        conn.close()
        flash('Hospital not found.', 'warning')
        return redirect(url_for('user_dashboard'))

    # Load this user's records at this hospital
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE r.user_id = ? AND d.hospital_id = ?
           ORDER BY r.date DESC''',
        (user_id, hospital_id),
    )
    records = cur.fetchall()

    conn.close()

    print_mode = request.args.get('print') == '1'

    return render_template('user_hospital_detail.html', hospital=hospital, records=records, print_mode=print_mode)


@app.route('/user/hospital/<int:hospital_id>/export/csv')
def user_hospital_export_csv(hospital_id):
    """Export this user's visits at a specific hospital as CSV (Excel-compatible)."""
    if current_role() != 'user':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    user_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure hospital exists
    cur.execute('SELECT * FROM hospitals WHERE id = ?', (hospital_id,))
    hospital = cur.fetchone()
    if not hospital:
        conn.close()
        flash('Hospital not found.', 'warning')
        return redirect(url_for('user_dashboard'))

    # Same filter as user_hospital_detail
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE r.user_id = ? AND d.hospital_id = ?
           ORDER BY r.date DESC''',
        (user_id, hospital_id),
    )
    records = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Date',
        'Doctor Name',
        'Doctor Specialization',
        'Diagnosis',
        'Medicines',
        'Dosage',
        'Treatment Status',
        'Prescription Text',
    ])

    for r in records:
        # Safely handle legacy rows that might not have new columns
        dosage_val = r['dosage'] if 'dosage' in r.keys() and r['dosage'] is not None else ''
        prescription_val = (
            r['prescription_text']
            if 'prescription_text' in r.keys() and r['prescription_text'] is not None
            else ''
        )
        status_val = (
            r['treatment_status']
            if 'treatment_status' in r.keys() and r['treatment_status'] is not None
            else ''
        )

        writer.writerow([
            r['date'],
            r['doctor_name'],
            r['doctor_specialization'],
            r['diagnosis'] or '',
            r['medicines'] or '',
            dosage_val,
            status_val,
            prescription_val,
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"user_hospital_{hospital_id}_visits.csv"
    response = Response(csv_data, mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


@app.route('/reports/<filename>')
def download_report(filename):
    """Keep download endpoint for backwards compatibility (forced attachment)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/reports/view/<filename>')
def view_report(filename):
    """Serve report inline so browser can preview PDF/images without download dialog."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


@app.route('/hospital/patient/<int:user_id>/export/csv')
def hospital_patient_export_csv(user_id):
    """Export all visits for a patient at this hospital as CSV (Excel-compatible)."""
    if current_role() != 'hospital':
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))

    hospital_id = current_user_id()

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure patient exists
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        flash('Patient not found.', 'warning')
        return redirect(url_for('hospital_dashboard'))

    # Same records as hospital_patient_detail
    cur.execute(
        '''SELECT r.*, d.name as doctor_name, d.specialization as doctor_specialization
           FROM records r
           JOIN doctors d ON r.doctor_id = d.id
           WHERE r.user_id = ? AND d.hospital_id = ?
           ORDER BY r.date DESC''',
        (user_id, hospital_id),
    )
    records = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        'Date',
        'Doctor Name',
        'Doctor Specialization',
        'Diagnosis',
        'Medicines',
        'Dosage',
        'Treatment Status',
        'Prescription Text',
    ])

    for r in records:
        # Safely handle legacy rows that might not have new columns
        dosage_val = r['dosage'] if 'dosage' in r.keys() and r['dosage'] is not None else ''
        prescription_val = (
            r['prescription_text']
            if 'prescription_text' in r.keys() and r['prescription_text'] is not None
            else ''
        )
        status_val = (
            r['treatment_status']
            if 'treatment_status' in r.keys() and r['treatment_status'] is not None
            else ''
        )

        writer.writerow([
            r['date'],
            r['doctor_name'],
            r['doctor_specialization'],
            r['diagnosis'] or '',
            r['medicines'] or '',
            dosage_val,
            status_val,
            prescription_val,
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"patient_{user_id}_visits.csv"
    response = Response(csv_data, mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


# -----------------
# Emergency module
# -----------------


@app.route('/emergency', methods=['GET', 'POST'])
def emergency():
    user_id = current_user_id() if current_role() == 'user' else None

    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        location = request.form['location']
        symptoms = request.form.get('symptoms', '')
        age = request.form.get('age')
        age = int(age) if age and age.isdigit() else None
        
        # New fields for ML model
        state = request.form.get('state')
        zone = request.form.get('zone')
        day = request.form.get('day')
        time_slot = request.form.get('time_slot')
        emergency_type = request.form.get('emergency_type')
        weather = request.form.get('weather')
        
        # Auto-detect day if not provided
        if not day:
            day = datetime.now().strftime('%A')
        
        # Auto-detect time slot if not provided
        if not time_slot:
            current_hour = datetime.now().hour
            if 5 <= current_hour < 12:
                time_slot = 'Morning'
            elif 12 <= current_hour < 17:
                time_slot = 'Afternoon'
            elif 17 <= current_hour < 21:
                time_slot = 'Evening'
            else:
                time_slot = 'Night'

        # Get user data if logged in
        user_data = None
        if user_id:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user_row = cur.fetchone()
            if user_row:
                # Convert Row to dict
                user_data = dict(user_row)
            conn.close()

        # Predict emergency priority using ML model
        priority, severity, prediction_score = predict_emergency_priority(
            symptoms=symptoms,
            age=age,
            location=location,
            state=state,
            zone=zone,
            day=day,
            time_slot=time_slot,
            emergency_type=emergency_type,
            weather=weather,
            user_data=user_data
        )

        # Calculate response time based on priority
        response_time_map = {
            'Critical': 5,
            'High': 10,
            'Medium': 15,
            'Low': 20
        }
        response_time = response_time_map.get(priority, 15)

        # Determine status based on priority
        status_map = {
            'Critical': 'Ambulance Dispatched - Critical Priority',
            'High': 'Ambulance Dispatched - High Priority',
            'Medium': 'Ambulance Dispatched - Medium Priority',
            'Low': 'Ambulance Dispatched - Low Priority'
        }
        status = status_map.get(priority, 'Ambulance Dispatched')

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            '''INSERT INTO emergencies (user_id, name, phone, location, status, requested_at, 
               response_time_minutes, priority, severity, prediction_score, symptoms, age,
               state, zone, day, time_slot, emergency_type, weather)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                user_id,
                name,
                phone,
                location,
                status,
                datetime.utcnow().isoformat(),
                response_time,
                priority,
                severity,
                prediction_score,
                symptoms,
                age,
                state,
                zone,
                day,
                time_slot,
                emergency_type,
                weather
            ),
        )
        conn.commit()

        # Get emergency ID for result page
        emergency_id = cur.lastrowid

        # Calculate active requests (pending emergencies, excluding current one)
        cur.execute('SELECT COUNT(*) AS c FROM emergencies WHERE id != ? AND status LIKE ?', 
                   (emergency_id, '%Dispatched%'))
        active_requests = cur.fetchone()['c']
        
        # Calculate available ambulances (simulated: total ambulances - active requests)
        # In a real system, this would come from ambulance tracking system
        total_ambulances = 10  # Simulated total ambulances
        available_ambulances = max(0, total_ambulances - active_requests - 1)  # -1 for current dispatch
        
        # Get prediction probabilities for all classes
        prediction_probabilities = {}
        if emergency_model and hasattr(emergency_model, 'predict_proba'):
            try:
                # Re-extract features for probability calculation
                symptom_text = str(symptoms).lower() if symptoms else ''
                age_normalized = (age / 100.0) if age else 0.5
                
                # Symptom severity
                critical_keywords = ['chest pain', 'difficulty breathing', 'unconscious', 'severe', 'emergency', 
                                    'critical', 'heart attack', 'stroke', 'bleeding', 'trauma', 'accident']
                high_keywords = ['pain', 'fever', 'vomiting', 'dizziness', 'nausea', 'weakness']
                moderate_keywords = ['discomfort', 'mild', 'ache', 'tired']
                
                symptom_severity = 0.0
                if any(keyword in symptom_text for keyword in critical_keywords):
                    symptom_severity = 1.0
                elif any(keyword in symptom_text for keyword in high_keywords):
                    symptom_severity = 0.7
                elif any(keyword in symptom_text for keyword in moderate_keywords):
                    symptom_severity = 0.4
                else:
                    symptom_severity = 0.2
                
                # Build features (same as in predict_emergency_priority)
                states_list = ['All India', 'Uttar Pradesh', 'Maharashtra', 'West Bengal', 'Jharkhand',
                               'Madhya Pradesh', 'Bihar', 'Rajasthan', 'Tamil Nadu', 'Orissa', 'Assam',
                               'Karnataka', 'Andhra Pradesh', 'Haryana', 'Chhatisgarh', 'Jammu and Kashmir',
                               'Telangana', 'Uttarakhand', 'Himachal Pradesh', 'Gujarat', 'Kerala',
                               'Arunachal Pradesh', 'Delhi', 'Nagaland', 'Mizoram', 'Meghalaya',
                               'Tripura', 'Manipur', 'Goa', 'Andaman and Nicobar Island', 'Ladakh',
                               'Sikkim', 'Puducherry', 'Dadra and Nagar Haveli and Daman and Diu', 'Chandigarh']
                zone_options = ['Urban', 'Rural', 'Highway']
                day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                time_slot_options = ['Morning', 'Afternoon', 'Evening', 'Night']
                emergency_type_options = ['EMS', 'Traffic', 'Fire']
                weather_options = ['Rain', 'Heatwave', 'Fog', 'Clear']
                
                features_list = [age_normalized, symptom_severity]
                
                # State one-hot
                state_onehot = [0] * len(states_list)
                if state and state in states_list:
                    state_onehot[states_list.index(state)] = 1
                features_list.extend(state_onehot)
                
                # Zone one-hot
                zone_onehot = [0] * len(zone_options)
                if zone and zone in zone_options:
                    zone_onehot[zone_options.index(zone)] = 1
                features_list.extend(zone_onehot)
                
                # Day one-hot
                day_onehot = [0] * len(day_options)
                if day and day in day_options:
                    day_onehot[day_options.index(day)] = 1
                features_list.extend(day_onehot)
                
                # Time slot one-hot
                time_slot_onehot = [0] * len(time_slot_options)
                if time_slot and time_slot in time_slot_options:
                    time_slot_onehot[time_slot_options.index(time_slot)] = 1
                features_list.extend(time_slot_onehot)
                
                # Emergency type one-hot
                emergency_type_onehot = [0] * len(emergency_type_options)
                if emergency_type and emergency_type in emergency_type_options:
                    emergency_type_onehot[emergency_type_options.index(emergency_type)] = 1
                features_list.extend(emergency_type_onehot)
                
                # Weather one-hot
                weather_onehot = [0] * len(weather_options)
                if weather and weather in weather_options:
                    weather_onehot[weather_options.index(weather)] = 1
                features_list.extend(weather_onehot)
                
                # Truncate to model's expected features
                n_features = emergency_model.n_features_in_
                if len(features_list) > n_features:
                    features_list = features_list[:n_features]
                elif len(features_list) < n_features:
                    features_list = features_list + [0.0] * (n_features - len(features_list))
                
                features = np.array([features_list])
                
                # Get probabilities
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    probabilities = emergency_model.predict_proba(features)[0]
                    classes = emergency_model.classes_
                    
                    # Map probabilities to priority levels
                    for i, class_name in enumerate(classes):
                        if class_name == 'Low':
                            prediction_probabilities['Low'] = float(probabilities[i]) * 100
                        elif class_name == 'Medium':
                            prediction_probabilities['Medium'] = float(probabilities[i]) * 100
                        elif class_name == 'High':
                            prediction_probabilities['High'] = float(probabilities[i]) * 100
            except Exception as e:
                print(f"[WARNING] Could not get prediction probabilities: {e}")
                # Fallback probabilities
                if priority == 'Low':
                    prediction_probabilities = {'Low': 75.0, 'Medium': 20.0, 'High': 5.0}
                elif priority == 'Medium':
                    prediction_probabilities = {'Low': 20.0, 'Medium': 60.0, 'High': 20.0}
                else:
                    prediction_probabilities = {'Low': 10.0, 'Medium': 30.0, 'High': 60.0}
        else:
            # Fallback probabilities
            if priority == 'Low':
                prediction_probabilities = {'Low': 75.0, 'Medium': 20.0, 'High': 5.0}
            elif priority == 'Medium':
                prediction_probabilities = {'Low': 20.0, 'Medium': 60.0, 'High': 20.0}
            else:
                prediction_probabilities = {'Low': 10.0, 'Medium': 30.0, 'High': 60.0}
        
        # Calculate area demand forecast
        demand_factors = []
        if day and day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            demand_factors.append('Weekday')
        if time_slot in ['Morning', 'Afternoon']:
            demand_factors.append(f'{time_slot.lower()} time slot')
        if weather == 'Clear':
            demand_factors.append('Clear weather conditions')
        elif weather in ['Rain', 'Fog']:
            demand_factors.append(f'{weather.lower()} conditions')
        
        # Historical frequency (simulated)
        cur.execute('SELECT COUNT(*) AS c FROM emergencies WHERE state = ?', (state or 'All India',))
        historical_count = cur.fetchone()['c']
        if historical_count < 5:
            demand_factors.append('Low historical emergency frequency')
        elif historical_count < 20:
            demand_factors.append('Moderate historical emergency frequency')
        else:
            demand_factors.append('High historical emergency frequency')
        
        if available_ambulances >= 3:
            demand_factors.append('Adequate ambulance availability')
        else:
            demand_factors.append('Limited ambulance availability')
        
        # Determine demand level
        if available_ambulances >= 4 and historical_count < 10:
            demand_level = 'LOW'
        elif available_ambulances >= 2 and historical_count < 20:
            demand_level = 'MEDIUM'
        else:
            demand_level = 'HIGH'
        
        # Determine life threat risk
        symptom_text = str(symptoms).lower() if symptoms else ''
        life_threat_keywords = ['chest pain', 'heart attack', 'stroke', 'unconscious', 'bleeding', 
                               'difficulty breathing', 'trauma', 'accident', 'severe']
        life_threat_risk = 'Yes' if any(keyword in symptom_text for keyword in life_threat_keywords) else 'No'
        
        # Determine dispatch type
        dispatch_type_map = {
            'Critical': 'Emergency Priority',
            'High': 'High Priority',
            'Medium': 'Standard Priority',
            'Low': 'Non-Emergency Priority'
        }
        dispatch_type = dispatch_type_map.get(priority, 'Standard Priority')
        
        # Decision logic
        decision_logic = []
        if severity in ['Mild', 'Moderate']:
            decision_logic.append('Patient condition is non-critical')
        else:
            decision_logic.append('Patient condition requires immediate attention')
        
        if demand_level == 'LOW':
            decision_logic.append('Area demand currently low')
        elif demand_level == 'HIGH':
            decision_logic.append('Area demand currently high')
        else:
            decision_logic.append('Area demand at moderate level')
        
        if active_requests == 0:
            decision_logic.append('No competing high-priority emergencies')
        else:
            decision_logic.append(f'{active_requests} other active emergency requests')
        
        # Simple analytics: total emergencies, avg response time
        cur.execute('SELECT COUNT(*) AS c, AVG(response_time_minutes) AS avg_rt FROM emergencies')
        stats = cur.fetchone()

        # Get the emergency record with predictions
        cur.execute('SELECT * FROM emergencies WHERE id = ?', (emergency_id,))
        emergency_record = cur.fetchone()

        conn.close()

        flash(f'Emergency request submitted. Ambulance is on the way! Priority: {priority}', 'success')
        return render_template('emergency_result.html', 
                             stats=stats, 
                             emergency=emergency_record,
                             priority=priority,
                             severity=severity,
                             prediction_score=prediction_score,
                             zone=zone,
                             day=day,
                             time_slot=time_slot,
                             emergency_type=emergency_type,
                             weather=weather,
                             active_requests=active_requests,
                             available_ambulances=available_ambulances,
                             demand_level=demand_level,
                             demand_factors=demand_factors,
                             life_threat_risk=life_threat_risk,
                             dispatch_type=dispatch_type,
                             decision_logic=decision_logic,
                             prediction_probabilities=prediction_probabilities,
                             response_time=response_time)

    return render_template('emergency.html')


# -------------
# Simple analytics for hospital dashboard emergency stats
# -------------


@app.route('/analytics/ambulance')
def ambulance_analytics():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) AS c, AVG(response_time_minutes) AS avg_rt FROM emergencies')
    stats = cur.fetchone()
    conn.close()
    return {
        'total_emergencies': stats['c'] if stats else 0,
        'avg_response_time': round(stats['avg_rt'], 1) if stats and stats['avg_rt'] else None,
    }


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
