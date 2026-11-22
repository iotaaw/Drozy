# DROWZY: DRIVER DROWSINESS DETECTION WITH NLP-BASED FATIGUE TEXT ANALYSIS

> **A Hybrid Artificial Intelligence System Integrating Computer Vision and Natural Language Processing for Driver Safety**

**Academic Project Report & Implementation**  
- **Candidate:** B. Sylesh (Roll No: `124cs0072`)  
- **Department:** Department of Computer Science & Engineering  
- **Institution:** Indian Institute of Information Technology Design & Manufacturing (IIITDM), Kurnool  
- **Academic Year:** 2026–2027  

---

## 1. Abstract

Driver fatigue and microsleeps are leading contributors to highway vehicular accidents worldwide. While conventional driver-monitoring systems depend solely on reactive vehicle trajectory tracking or intrusive wearable biometric sensors, modern Advanced Driver Assistance Systems (ADAS) benefit immensely from multi-modal artificial intelligence.

**DROWZY** is an integrated, dual-modality safety framework combining:
1. **Real-Time Computer Vision (CV):** An edge-compatible facial landmark tracking pipeline that continuously monitors the driver's eye state, calculates the **Eye Aspect Ratio (EAR)**, and triggers instant visual and acoustic alarms when prolonged eye closure ($\ge 48$ consecutive frames at $\text{EAR} < 0.30$) is detected.
2. **Natural Language Processing (NLP):** A machine learning text analysis engine that processes natural language driver utterances or speech transcripts into five cognitive/safety states (**FATIGUE**, **ALERT**, **REST_REQUEST**, **STRESS**, **NEUTRAL**) to provide proactive safety advice before involuntary microsleep occurs.

The system demonstrates that accessible, non-intrusive software techniques can provide robust driver protection on commodity personal computing hardware without requiring expensive graphics cards or wearable sensors.

---

## 2. Problem Statement

Driving under exhaustion or continuous physical stress leads to involuntary **microsleep episodes**—brief lapses of consciousness lasting from a fraction of a second up to several seconds. At highway velocities (e.g., $100\text{ km/h}$), a $2$-second microsleep results in traveling over $55\text{ meters}$ without steering control or hazard reaction capability.

Conventional solutions present notable limitations:
- **Wearable Sensors (EEG/ECG/Wristbands):** Intrusive, cause skin irritation, restrict natural head motion, and face strong resistance from drivers.
- **Vehicle Telemetry (Steering/Lane Departure):** Reactive rather than proactive; warnings trigger only after the vehicle has already begun swerving or drifting across lanes.
- **Cognitive State Invisibility:** Traditional optical systems cannot register early mental exhaustion, stress, or a driver's explicit desire to rest before physical eyelid closure begins.

**The Solution:** DROWZY unites optical monitoring of physical symptoms with natural language comprehension of cognitive self-reports.

---

## 3. Project Objectives

1. **Live Video Feed Acquisition:** Continuously capture digital video frames from a connected webcam (`cv2.VideoCapture(0)`) at standard frame rates.
2. **Facial Landmark Localization:** Rapidly localize the driver's face using dlib's Histogram of Oriented Gradients (HOG) + Linear SVM and map 68 anatomical landmark points.
3. **Eye Aspect Ratio (EAR) Computation:** Compute vertical and horizontal Euclidean distances across eye landmarks to measure eyelid openness in real time.
4. **Blink vs. Microsleep Discrimination:** Implement a temporal persistence threshold (`48` consecutive frames at $\text{EAR} < 0.30$) to systematically ignore natural blinks ($100\text{–}300\text{ ms}$) while identifying true microsleeps ($\ge 1.6\text{ s}$).
5. **Immediate Dual Alarming:** Overlay prominent `"DROWSINESS DETECTED"` warning text and emit a $2500\text{ Hz}$ acoustic alarm via `winsound.Beep`.
6. **NLP Fatigue Classification:** Provide a natural language text analyzer trained on a custom balanced dataset ($300$ samples) classifying driver statements into $5$ operational categories.
7. **Educational NLP Pipeline:** Implement clean, modular text normalization, tokenization, stopword filtering, lemmatization, and TF-IDF feature extraction inside a Scikit-Learn pipeline.
8. **Interactive Multi-Module GUI:** Provide both a standalone OpenCV desktop runner (`main.py`) and an integrated Streamlit interface (`app.py`).

---

## 4. System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    DROWZY HYBRID SYSTEM ARCHITECTURE                               |
+----------------------------------------------------------------------------------------------------+
                                                  |
           +--------------------------------------+--------------------------------------+
           |                                                                             |
           v                                                                             v
+------------------------------------+                         +------------------------------------+
|   PILLAR 1: COMPUTER VISION (CV)   |                         |  PILLAR 2: NATURAL LANGUAGE (NLP)  |
+------------------------------------+                         +------------------------------------+
| 1. Video Capture (Webcam Index 0)  |                         | 1. Driver Text / Voice Transcript  |
| 2. Preprocessing (450px, Grayscale)|                         | 2. Text Preprocessing:             |
| 3. Face Detection (dlib HOG + SVM) |                         |    - Lowercase & Punctuation Clean |
| 4. 68 Facial Landmarks (Kazemi ERT)|                         |    - Tokenization & Stopwords      |
| 5. Eye Landmark Extraction:        |                         |    - Morphological Lemmatization   |
|    - Left Eye:  [42:48]            |                         | 3. TF-IDF Feature Extraction       |
|    - Right Eye: [36:42]            |                         |    - Unigram & Bigram Features     |
| 6. Eye Aspect Ratio (EAR) Engine:  |                         |    - Sublinear TF Scaling          |
|    - EAR = (||p2-p6||+||p3-p5||)/  |                         | 4. Multinomial Logistic Regression |
|            (2 * ||p1-p4||)         |                         | 5. Class Probability Estimation    |
| 7. Threshold Evaluation:           |                         | 6. Categorization (5 Classes):     |
|    - EAR < 0.30 for >= 48 frames   |                         |    FATIGUE, ALERT, REST_REQUEST,   |
| 8. Emergency Alarming:             |                         |    STRESS, NEUTRAL                 |
|    - "DROWSINESS DETECTED" Banner  |                         | 7. Actionable Safety Guidance      |
|    - winsound.Beep (2500Hz, 1000ms)|                         |    - Rest stop / break advice      |
+------------------------------------+                         +------------------------------------+
           |                                                                             |
           +--------------------------------------+--------------------------------------+
                                                  |
                                                  v
                   +--------------------------------------------------------------+
                   |                 INTEGRATED STREAMLIT DASHBOARD               |
                   |               Interactive Controls & Visual UI               |
                   +--------------------------------------------------------------+
```

---

## 5. Dual-Modality Comparison

| Operational Dimension | Pillar 1: Computer Vision (CV) | Pillar 2: Natural Language Processing (NLP) |
| :--- | :--- | :--- |
| **Input Modality** | Video frames from optical webcam | Driver text inputs or voice transcripts |
| **Monitored State** | Physical eyelid closure, microsleeps | Cognitive exhaustion, stress, rest requests |
| **Temporal Horizon** | Real-time emergency (seconds before crash) | Anticipatory / Preventive (minutes/hours ahead) |
| **Primary Metric** | Eye Aspect Ratio ($\text{EAR} < 0.30$ for $48$ frames) | TF-IDF n-gram vector + Logistic Regression probability |
| **Action / Response** | Loud $2500\text{ Hz}$ alarm + screen text | Contextual guidance, route rest stop suggestion |
| **Sensor Requirement** | Contactless optical camera (Index 0) | Microphonic speech-to-text or dashboard input |

---

## 6. Technologies and Libraries Used

| Technology | Category | Role in DROWZY |
| :--- | :--- | :--- |
| **Python 3.10–3.12** | Programming Language | Core execution runtime across all modules |
| **OpenCV (`cv2`)** | Computer Vision | Video stream capture, frame resizing, color transformation, contour rendering |
| **dlib** | Machine Learning Toolkit | HOG frontal face detector and 68-point facial shape predictor |
| **imutils** | Image Processing Utility | Aspect-ratio resizing and facial landmark index slicing |
| **SciPy (`scipy.spatial`)** | Numerical Computing | Euclidean distance calculation (`dist.euclidean`) |
| **Scikit-Learn** | Machine Learning | `Pipeline`, `TfidfVectorizer`, `LogisticRegression`, and evaluation metrics |
| **NLTK / Morphological Rules**| NLP Library | Word tokenization, English stopwords filtering, and lemmatization |
| **Pandas & NumPy** | Data Engineering | Dataset structuring, CSV manipulation, matrix operations |
| **Streamlit** | Web Interface Framework | Interactive browser GUI uniting CV launcher and NLP analyzer |
| **winsound** | Audio Driver | Native Windows motherboard/speaker acoustic alarm ($2500\text{ Hz}$) |

---

## 7. Computer Vision Methodology

### 7.1 Face & Facial Landmark Detection
- **Detector:** `dlib.get_frontal_face_detector()` implements a Histogram of Oriented Gradients (HOG) feature descriptor paired with a Linear Support Vector Machine (SVM).
- **Landmark Predictor:** `shape_predictor_68_face_landmarks.dat` implements the Ensemble of Regression Trees (ERT) algorithm (Kazemi & Sullivan, 2014) to pinpoint 68 $(x, y)$ coordinates in milliseconds.
- **Eye Region Extraction:**
  - Left Eye: Points `42` to `47` (Python slice `[42:48]`)
  - Right Eye: Points `36` to `41` (Python slice `[36:42]`)

### 7.2 Eye Aspect Ratio (EAR) Formulation
For a single eye with 6 landmark points $(p_1, p_2, p_3, p_4, p_5, p_6)$:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

```python
def eyeAspectRatio(eye):
    # Vertical Euclidean Distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal Distance
    C = dist.euclidean(eye[0], eye[3])
    # Return Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear
```

Both eyes are calculated independently and averaged:
$$\text{ear} = \frac{\text{leftEAR} + \text{rightEAR}}{2.0}$$

### 7.3 Drowsiness Thresholding Logic
- **`earThresh = 0.3`**: If $\text{EAR} < 0.3$, the eyes are classified as closed.
- **`earFrames = 48`**: At $\sim 30\text{ FPS}$, $48$ consecutive frames correspond to $\approx 1.6\text{ seconds}$ of sustained closure.
- **Blink Rejection:** Normal blinks last only $100\text{–}300\text{ ms}$ ($3\text{–}5\text{ frames}$), resetting the counter to $0$ without raising false alarms.
- **Alarm Trigger:** Once `count >= 48`, the system renders `"DROWSINESS DETECTED"` in red text at `(10, 30)` and activates `winsound.Beep(2500, 1000)`.

---

## 8. Natural Language Processing (NLP) Methodology

The NLP module analyzes natural language driver feedback or speech-to-text transcripts to identify fatigue, stress, or rest needs.

### 8.1 NLP Preprocessing Stages
1. **Lowercasing & Normalization:** Converts input text to uniform lowercase and cleans accented or special characters.
2. **Punctuation Cleaning:** Strips non-alphanumeric symbols while retaining intra-word hyphens and single apostrophes.
3. **Word Tokenization:** Segments continuous strings into discrete linguistic tokens using regular-expression boundaries (`\b[a-zA-Z]{2,}\b`).
4. **Stopword Removal:** Filters out high-frequency functional words using the standard NLTK 179-word English stopword corpus.
5. **Lemmatization:** Reduces inflected forms to their dictionary lemma using `WordNetLemmatizer` supplemented by driving/fatigue domain stem maps (e.g., `"drowsiness"` $\rightarrow$ `"drowsy"`, `"exhausted"` $\rightarrow$ `"exhaust"`).

### 8.2 TF-IDF Feature Extraction
Converts preprocessed token sequences into weighted numerical vectors:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

- **N-gram Range:** Unigrams and Bigrams (`ngram_range=(1, 2)`) to capture both single terms (`"sleepy"`) and phrases (`"cannot keep"`, `"pull over"`).
- **Sublinear Scaling:** Applies `sublinear_tf=True` ($1 + \log(\text{TF})$) to prevent high-frequency terms from overshadowing informative rare words.

### 8.3 Classification Algorithm
- **Classifier:** Multinomial Logistic Regression (`LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')`).
- **Scikit-Learn Pipeline:** Encapsulates vectorization and classification inside a unified object to prevent data leakage during cross-validation and testing:
  ```python
  pipeline = Pipeline([
      ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
      ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))
  ])
  ```

---

## 9. Dataset Description

The NLP dataset (`nlp/dataset.csv`) contains **300 curated, balanced samples** across 5 distinct operational categories ($60$ samples per class):

| Category | Samples | Core Semantic Themes & Examples |
| :--- | :---: | :--- |
| **FATIGUE** | 60 | Heavy eyelids, yawning, microsleep sensation (*"I cannot keep my eyes open"*, *"Feeling drowsy and nodding off"*) |
| **ALERT** | 60 | High energy, sharp focus, ready to drive (*"I am wide awake and attentive"*, *"Well rested and fully focused"*) |
| **REST_REQUEST** | 60 | Explicit desire to stop, pull over, coffee break (*"Let us pull over at the next rest area"*, *"Need a 15 minute nap"*) |
| **STRESS** | 60 | Highway anxiety, bad weather frustration, irritation (*"Traffic is giving me a severe headache"*, *"Aggressive drivers are stressing me out"*) |
| **NEUTRAL** | 60 | Objective vehicle telemetry, route navigation (*"The vehicle is cruising at 60 mph"*, *"Taking the next exit toward highway 40"*) |

---

## 10. Actual Model Evaluation Results

The classifier was evaluated using a **stratified 80/20 train/test split** ($240$ training samples, $60$ testing samples) using `random_state=42`. All metrics are derived from actual model execution:

### Summary Performance
- **Overall Accuracy:** `90.00%` ($54 / 60$ correct test predictions)
- **Macro-Averaged Precision:** `90.49%`
- **Macro-Averaged Recall:** `90.00%`
- **Macro-Averaged F1-Score:** `90.12%`

### Detailed Classification Report

| Category | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **ALERT** | `100.00%` | `100.00%` | `100.00%` | 12 |
| **FATIGUE** | `84.62%` | `91.67%` | `88.00%` | 12 |
| **NEUTRAL** | `90.91%` | `83.33%` | `86.96%` | 12 |
| **REST_REQUEST** | `100.00%` | `91.67%` | `95.65%` | 12 |
| **STRESS** | `76.92%` | `83.33%` | `80.00%` | 12 |
| **Overall / Macro Avg** | **`90.49%`** | **`90.00%`** | **`90.12%`** | **60** |

### Confusion Matrix ($5 \times 5$)
```
Actual \ Predicted   ALERT   FATIGUE   NEUTRAL   REST_REQUEST   STRESS
ALERT                 12        0         0            0           0
FATIGUE                0       11         0            0           1
NEUTRAL                0        0        10            0           2
REST_REQUEST           0        1         0           11           0
STRESS                 0        1         1            0          10
```

---

## 11. Safety Recommendations & Guidance Engine

When a text input is analyzed, DROWZY pairs the predicted class with contextual, informational advice:

- **FATIGUE:** *"Fatigue-related indicators detected. Consider stopping at a safe location and resting before continuing your journey."*
- **ALERT:** *"No fatigue indicators were detected in the entered text. Continue to remain attentive and follow safe driving practices."*
- **REST_REQUEST:** *"You indicated that you need rest. Please stop at a safe location and take a break."*
- **STRESS:** *"Stress-related indicators detected. Consider taking a break at a safe location before continuing."*
- **NEUTRAL:** *"No clear fatigue-related indicators were identified in the text."*

> **Safety Disclaimer:** The NLP classifier provides informational assistance based on driver text inputs. It does not provide medical diagnoses and does not override real-time optical warnings from the computer vision system.

---

## 12. Project Directory Structure

```
Drowsiness/
│
├── main.py                               # Standalone OpenCV Webcam Drowsiness Detection
├── app.py                                # Integrated Streamlit Dashboard (CV + NLP)
├── shape_predictor_68_face_landmarks.dat# Pre-trained dlib 68-point landmark model (~99.7 MB)
├── requirements.txt                      # Project library dependencies
├── .gitignore                            # Git exclusion rules
├── README.md                             # Comprehensive project documentation
├── PROJECT_REPORT.md                     # Complete academic project report (Markdown)
│
├── nlp/                                  # Natural Language Processing Subsystem
│   ├── __init__.py                       # Package initializer
│   ├── preprocessing.py                  # Tokenization, cleaning, stopword removal, lemmatization
│   ├── dataset.csv                       # 300 labeled driver sentences across 5 categories
│   ├── train_model.py                    # Scikit-learn TF-IDF + Logistic Regression pipeline trainer
│   ├── fatigue_classifier.py             # Inference classifier and safety recommendation engine
│   ├── evaluation.py                     # Out-of-sample testing and verification script
│   ├── evaluation_results.json           # Actual saved evaluation metrics
│   └── model.pkl                         # Serialized pipeline model artifact
│
├── Drowsiness_Detection_Project_Report.docx # Academic project report (Microsoft Word)
├── Drowsiness_Detection_Project_Report.pdf  # Academic project report (PDF, 23+ pages)
├── Drowsiness_Detection_Presentation.pptx   # College project defense presentation (PowerPoint)
│
├── confusion_matrix.png                  # Visualized test confusion matrix heatmap
├── dataset_distribution.png              # Class sample distribution bar chart
├── system_architecture.png               # High-level hybrid architecture diagram
└── nlp_pipeline_diagram.png              # Text preprocessing & classification pipeline diagram
```

---

## 13. Installation & Setup (Windows PowerShell)

### Step 1: Open Terminal & Navigate to Project Directory
```powershell
cd C:\Users\Ramakrishna\OneDrive\Pictures\Drowsiness
```

### Step 2: Create & Activate Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Pre-trained Model File
Ensure `shape_predictor_68_face_landmarks.dat` is present in the root folder. If missing, download it from the official dlib repository and decompress it.

---

## 14. Execution Instructions

### Option A: Launch Integrated Web Dashboard (Recommended)
Launches the full interactive browser interface containing both the NLP analyzer and CV launcher:
```powershell
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

### Option B: Run Standalone Computer Vision System
Runs the direct OpenCV webcam monitoring loop:
```powershell
python main.py
```
- Focus on the window titled **`Frame`**.
- Keep your eyes open $\rightarrow$ Normal tracking with red eye contours.
- Close eyes for $\ge 48$ frames ($\sim 1.6\text{ s}$) $\rightarrow$ On-screen red `"DROWSINESS DETECTED"` warning and $2500\text{ Hz}$ audio alarm.
- Press **`q`** on your keyboard to exit.

### Option C: Train & Evaluate the NLP Model
Retrain the model on `nlp/dataset.csv` and regenerate metrics:
```powershell
python -m nlp.train_model
```
Run independent evaluation and generalization tests on unseen sentences:
```powershell
python -m nlp.evaluation
```

---

## 15. Sample NLP Inputs and Outputs

| # | Input Driver Utterance | Predicted Class | Confidence | Safety Guidance |
| :-: | :--- | :---: | :---: | :--- |
| 1 | *"I feel like my eyes are closing against my will."* | **FATIGUE** | `32.0%` | Stop at a safe location and rest before continuing. |
| 2 | *"My eyelids are drooping and I am yawning every minute."* | **FATIGUE** | `41.9%` | Fatigue indicators detected. Please take a rest break. |
| 3 | *"I feel refreshed and ready for another hundred kilometers."* | **ALERT** | `46.3%` | No fatigue detected. Continue attentive driving. |
| 4 | *"Let us take a brief stop at the upcoming service area."* | **REST_REQUEST** | `67.4%` | You indicated rest need. Stop safely and take a break. |
| 5 | *"Traffic is unbearable and making me extremely irritable."* | **STRESS** | `47.3%` | Stress indicators detected. Consider taking a breather. |
| 6 | *"The cruise control is holding the car at fifty miles per hour."* | **NEUTRAL** | `46.7%` | No clear fatigue indicators identified. |

---

## 16. Advantages

1. **Dual-Pillar Protection:** Fuses physiological real-time eye tracking with cognitive self-reported text sentiment.
2. **Contactless & Non-Invasive:** Requires zero wearable sensors, adhesive patches, or headbands.
3. **High Temporal Precision:** Distinguishes ordinary $150\text{ ms}$ blinks from dangerous $1.6\text{ s}$ microsleeps.
4. **Lightweight & Efficient:** Executes at $25\text{–}30\text{ FPS}$ on standard CPU hardware without GPU acceleration.
5. **Standardized NLP Pipeline:** Implements true text normalization, tokenization, stopword removal, lemmatization, TF-IDF vectorization, and Logistic Regression without data leakage.
6. **Fully Offline:** Self-contained execution without external cloud API dependencies.

---

## 17. Limitations & Future Scope

### Current Limitations
- **Lighting Sensitivity:** Standard RGB optical cameras suffer under pitch-black nighttime cabin conditions.
- **Eyewear Occlusion:** Dark tinted or polarized sunglasses prevent optical landmark extraction on pupils.
- **Manual Text Input:** Current prototype requires typing or simulated voice transcripts.
- **Synchronous Audio Call:** `winsound.Beep` temporarily pauses video frames for $1.0\text{ s}$ during alarms.

### Future Scope
- **Speech-to-Text Voice Integration:** Connecting offline automatic speech recognition (e.g., Whisper) for hands-free voice analysis.
- **Yawning Detection (Mouth Aspect Ratio - MAR):** Tracking points `48` to `67` to flag repetitive yawning before eye closure.
- **Head Pose Estimation:** Calculating Pitch, Yaw, and Roll to identify head nodding and driver distraction.
- **Infrared (NIR) Night-Vision:** Adding $850\text{ nm}$ IR LEDs for reliable night driving surveillance.
- **Embedded Edge Deployment:** Porting the hybrid architecture to a Raspberry Pi 5 or NVIDIA Jetson Nano.

---

## 18. References

1. **Soukupová, T., & Čech, J. (2016).** *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop (CVWW).
2. **Kazemi, V., & Sullivan, J. (2014).** *One Millisecond Face Alignment with an Ensemble of Regression Trees.* IEEE CVPR, pp. 1867–1874.
3. **Dalal, N., & Triggs, B. (2005).** *Histograms of Oriented Gradients for Human Detection.* IEEE CVPR, Vol. 1, pp. 886–893.
4. **Jurafsky, D., & Martin, J. H. (2023).** *Speech and Language Processing (3rd ed. draft).* Pearson.
5. **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval.* Cambridge University Press.
6. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
7. **Bradski, G. (2000).** *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.
8. **King, D. E. (2009).** *Dlib-ml: A Machine Learning Toolkit.* Journal of Machine Learning Research, 10, pp. 1755–1758.

---

**Academic Submission & Defense Record**  
**B. Sylesh (124cs0072)**  
Department of Computer Science & Engineering  
Indian Institute of Information Technology Design & Manufacturing (IIITDM), Kurnool
