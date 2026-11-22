# DROWZY: DRIVER DROWSINESS DETECTION WITH NLP-BASED FATIGUE TEXT ANALYSIS

> **A Hybrid Artificial Intelligence System Integrating Computer Vision and Natural Language Processing for Driver Safety**

---

## 1. COVER PAGE

```
INDIAN INSTITUTE OF INFORMATION TECHNOLOGY DESIGN & MANUFACTURING
                             KURNOOL
          DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING

------------------------------------------------------------------------------------------------

          DROWZY: DRIVER DROWSINESS DETECTION WITH
               NLP-BASED FATIGUE TEXT ANALYSIS

   A Hybrid Artificial Intelligence System Integrating Computer Vision and
             Natural Language Processing for Driver Safety


         Submitted in partial fulfillment of the requirements for the degree of
               Bachelor of Technology in Computer Science & Engineering



                  Candidate Name:  B. Sylesh
       Roll Number / Registration:  124cs0072
       Department & Institution:  Dept. of Computer Science & Engineering, IIITDM Kurnool



                            ACADEMIC YEAR 2026–2027
   IIITDM Kurnool, Jagannathagattu, Dinnedevarapadu, Kurnool – 518008, AP, India
```

---

## 2. ABSTRACT

Driver fatigue, drowsiness, and cognitive exhaustion are among the primary causes of highway vehicular accidents globally. Lapses of driver attention lasting merely 1 to 2 seconds (microsleeps) can lead to catastrophic loss of vehicular control. To address this challenge comprehensively, this project develops **DROWZY**, a hybrid artificial intelligence driver safety system integrating real-time **Computer Vision (CV)** and **Natural Language Processing (NLP)**.

The Computer Vision component provides non-invasive, live physical driver monitoring. Utilizing a standard webcam stream at 30 FPS, each video frame is downscaled and converted to grayscale. dlib's Histogram of Oriented Gradients (HOG) and Linear SVM frontal face detector identifies the facial region, after which a 68-point shape predictor localizes the precise coordinate contours of both eyes. The Eye Aspect Ratio (EAR) is computed dynamically. To eliminate false alarms caused by natural 150 ms blinks, the system enforces an EAR threshold of `0.30` over a persistence window of `48 consecutive frames` (~1.6 seconds). When prolonged eye closure is confirmed, a high-visibility on-screen warning banner (`DROWSINESS DETECTED`) and an immediate 2500 Hz acoustic alert are triggered.

To complement visual monitoring with cognitive state comprehension, an NLP-based Driver Fatigue Text Analyzer is integrated. The NLP module allows drivers to input natural language statements or speech-to-text transcripts regarding their physiological and emotional condition. A robust NLP pipeline executes text normalization, punctuation cleaning, tokenization, stopword removal (179-word corpus), and morphological lemmatization. Text features are vectorized using Term Frequency–Inverse Document Frequency (TF-IDF) with unigram and bigram extraction and sublinear scaling. A Multinomial Logistic Regression classifier trained on a curated 300-sample balanced dataset categorizes driver utterances into five operational classes: **FATIGUE**, **ALERT**, **REST_REQUEST**, **STRESS**, and **NEUTRAL**.

On an independent stratified test set (60 samples), the NLP classifier achieves a verified accuracy of **90.00%**, macro precision of **90.49%**, macro recall of **90.00%**, and macro F1-score of **90.12%**. Each classification is coupled with actionable, context-aware safety guidance. DROWZY demonstrates how computer vision and natural language processing complement each other to form a reliable, multi-modal Advanced Driver Assistance System (ADAS) running on standard commodity hardware without specialized sensors.

---

## 3. TABLE OF CONTENTS

```
Chapter No.  Title                                                          Page No.
------------------------------------------------------------------------------------
             Cover Page .....................................................   i
             Abstract .......................................................  ii
             Table of Contents .............................................. iii

1.           Introduction ...................................................   1
2.           Problem Statement ..............................................   3
3.           Objectives .....................................................   4
4.           Existing System ................................................   5
5.           Proposed System: Hybrid Dual-Modality Framework ................   7
6.           System Architecture ............................................   9
7.           Hardware and Software Requirements .............................  11
8.           Computer Vision Component: Face & Eye Landmark Detection .......  13
9.           Eye Aspect Ratio (EAR) & Drowsiness Logic ......................  15
10.          Natural Language Processing (NLP) Component Introduction .......  18
11.          Text Preprocessing Methodology .................................  20
12.          Feature Extraction via TF-IDF Vectorization ....................  22
13.          Machine Learning Classification: Logistic Regression ...........  24
14.          Dataset Description & Analysis .................................  26
15.          Model Training, Evaluation & Experimental Results ..............  28
16.          Safety Recommendations & Guidance Engine .......................  31
17.          Implementation & Code Explanation ..............................  33
18.          System Interface & Output Demonstration ........................  38
19.          Testing & Observations .........................................  41
20.          Advantages .....................................................  44
21.          Limitations ....................................................  45
22.          Future Scope ...................................................  46
23.          Conclusion .....................................................  47
24.          References .....................................................  48
------------------------------------------------------------------------------------
```

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background & Context
Transportation safety is a critical pillar of modern civil society and industrial logistics. Despite notable technological advances in vehicular braking and passenger cabin restraint systems, driver inattention caused by physical exhaustion, chronic fatigue, and prolonged highway monotony remains a leading cause of traffic fatalities worldwide. Statistical surveys published by the World Health Organization (WHO) and transportation safety administrations indicate that driver fatigue is implicated in 20% to 30% of commercial vehicle crashes.

Drowsiness significantly impairs neuromuscular reflexes, reduces situational awareness, and slows emergency reaction time. A vehicle traveling at a typical highway cruising speed of 100 km/h travels approximately 28 meters per second. Consequently, a brief microsleep lasting merely 2 seconds results in over 55 meters of uncontrolled travel—an interval more than sufficient to trigger a catastrophic roadway departure, head-on collision, or multi-vehicle pileup.

### 1.2 The Paradigm of Multi-Modal Artificial Intelligence
Conventional driver monitoring solutions typically focus on a single physical indicator: either observing the vehicle's trajectory on the road or observing the driver's face. While physical eye closure is a definitive late-stage manifestation of microsleep, cognitive fatigue, driving stress, and mental exhaustion begin accumulating well before the driver's eyelids physically droop.

By combining Computer Vision (CV) for live physical monitoring with Natural Language Processing (NLP) for cognitive state analysis, an intelligent vehicle system gains dual-modality perception. The driver can communicate their feelings through voice or text transcripts, allowing proactive fatigue intervention before an emergency occurs.

---

## CHAPTER 2: PROBLEM STATEMENT

Driver fatigue produces involuntary physiological episodes known as microsleeps. Developing an effective, accessible safety monitoring system involves several real-world technical challenges:
1. **Differentiating Natural Blinking from Hazardous Microsleeps:** The human eye blinks involuntarily every few seconds to lubricate the cornea (lasting 100–300 ms). An automated vision system must not produce distracting false alarms during normal blinks, yet must reliably alert the driver when eyes remain closed continuously beyond a safe temporal threshold.
2. **Intrusiveness and Driver Compliance:** Traditional biomedical fatigue monitors rely on electroencephalogram (EEG) caps, electrooculogram (EOG) electrodes, or wrist-worn physiological monitors. Drivers resist wearing body attachments while operating vehicles due to physical discomfort and restricted head motion.
3. **Latency and Hardware Constraints:** Heavy deep learning models (such as 3D CNNs or large transformers) require dedicated graphical processing units (GPUs) with substantial power draw and thermal dissipation. A mass-market system must operate at real-time frame rates (25–30 FPS) on standard CPUs.
4. **Invisibility of Cognitive Exhaustion:** Optical camera systems detect eye closure only after physical sleep has begun. They cannot capture early mental strain, highway anxiety, or explicit driver intentions to stop unless complemented by natural language understanding.

---

## CHAPTER 3: OBJECTIVES

The core technical objectives of the DROWZY hybrid project are:
- **Real-Time Video Acquisition:** Stream frames from webcam index 0 at 30 FPS with minimal frame latency.
- **Facial Landmark Localization:** Accurately localize the driver's face and predict 68 structural landmark coordinates using dlib.
- **Eye Aspect Ratio (EAR) Engine:** Calculate vertical and horizontal Euclidean distances across eye landmarks to measure eye openness continuously.
- **Temporal Filtering:** Enforce an EAR threshold of 0.30 and a consecutive frame counter of 48 frames (~1.6 seconds) to prevent false alarms from natural blinking.
- **Immediate Audio-Visual Intervention:** Overlay 'DROWSINESS DETECTED' on the video canvas and trigger a 2500 Hz acoustic alarm via winsound.
- **NLP Fatigue Text Classifier:** Build an NLP model that categorizes driver statements into five classes: FATIGUE, ALERT, REST_REQUEST, STRESS, and NEUTRAL.
- **Standardized NLP Preprocessing:** Implement tokenization, stopword filtering, and morphological lemmatization in Python.
- **Machine Learning Pipeline:** Vectorize text using TF-IDF and train a Multinomial Logistic Regression model using Scikit-Learn.
- **Actionable Safety Guidance:** Provide contextual recommendations paired with predicted categories.
- **Integrated User Interface:** Deliver both a direct OpenCV executable (`main.py`) and an interactive Streamlit application (`app.py`).

---

## CHAPTER 4: EXISTING SYSTEM

Existing driver monitoring methods fall into three main categories:
1. **Physiological Sensor Monitoring (EEG/ECG):** Highly intrusive and rejected by drivers.
2. **Vehicle Telemetry Analysis (Steering/Lane departure):** Delayed reaction; alerts only after the car drifts.
3. **Single-Modality Optical Monitoring:** Blind to early cognitive stress and driver verbal communication.

---

## CHAPTER 5: PROPOSED SYSTEM: HYBRID DUAL-MODALITY FRAMEWORK

| Operational Dimension | Pillar 1: Computer Vision (CV) | Pillar 2: Natural Language Processing (NLP) |
| :--- | :--- | :--- |
| **Input Data** | Live video frames from webcam (index 0) | Driver text inputs or voice transcripts |
| **Monitored Target** | Physical eyelid closure, blinks, microsleeps | Cognitive exhaustion, mental stress, rest intent |
| **Detection Timing** | Immediate emergency (seconds before crash) | Proactive / Preventive (minutes or hours ahead) |
| **Core Algorithm** | 68-point ERT + Euclidean Eye Aspect Ratio | NLTK Preprocessing + TF-IDF + Logistic Regression |
| **Threshold / Metric** | EAR < 0.30 for >= 48 consecutive frames | Multiclass probability across 5 calibrated categories |
| **Intervention** | 2500 Hz audio alarm + on-screen visual alert | Context-aware safety advice & rest stop guidance |

---

## CHAPTER 6: SYSTEM ARCHITECTURE

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
```

---

## CHAPTER 7: HARDWARE AND SOFTWARE REQUIREMENTS

| Component / Library | Version | Role in DROWZY |
| :--- | :--- | :--- |
| **Python** | 3.10–3.12 | Core execution environment |
| **OpenCV (cv2)** | 4.8+ | Frame capture, downscaling, color conversion, contour rendering |
| **dlib** | 19.24+ | HOG frontal face detector and 68-point shape predictor |
| **imutils** | 0.5.4 | Aspect-ratio resizing and landmark index slicing |
| **SciPy** | 1.10+ | Euclidean distance calculation (`dist.euclidean`) |
| **Scikit-Learn** | 1.3+ | TF-IDF Vectorization, Logistic Regression, and Evaluation Metrics |
| **NLTK** | 3.8+ | Tokenization, English stopwords filtering, and lemmatization |
| **Streamlit** | 1.28+ | Interactive web dashboard combining CV launcher and NLP analyzer |
| **winsound** | Built-in | Generating 2500 Hz motherboard acoustic alarm tones on Windows |

---

## CHAPTER 8: COMPUTER VISION COMPONENT: FACE & EYE LANDMARK DETECTION

- **Face Detector:** dlib's `get_frontal_face_detector()` implements HOG + Linear SVM.
- **Landmark Model:** `shape_predictor_68_face_landmarks.dat` implements the Ensemble of Regression Trees (ERT) algorithm.
- **Eye Coordinates:**
  - Left Eye: Points `42` to `47`
  - Right Eye: Points `36` to `41`

---

## CHAPTER 9: EYE ASPECT RATIO (EAR) & DROWSINESS LOGIC

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- Open Eye EAR: `0.32` to `0.40`
- Closed Eye EAR: Drops below `0.20`
- Detection Parameters:
  - `earThresh = 0.30`
  - `earFrames = 48` ($\approx 1.6\text{ seconds}$ at 30 FPS)
  - Visual Alert: `"DROWSINESS DETECTED"` in red text at `(10, 30)`
  - Acoustic Alarm: `winsound.Beep(2500, 1000)`

---

## CHAPTER 10: NLP COMPONENT INTRODUCTION

The NLP subsystem categorizes driver self-reports into five operational classes:
1. **FATIGUE:** Physical/mental exhaustion, heavy eyes, yawning.
2. **ALERT:** Attentive, focused, energized driving state.
3. **REST_REQUEST:** Explicit requests to pull over, drink coffee, take a break.
4. **STRESS:** Traffic anxiety, bad weather strain, road frustration.
5. **NEUTRAL:** Route settings, vehicle speed, weather statements.

---

## CHAPTER 11: TEXT PREPROCESSING METHODOLOGY

The preprocessing pipeline (`nlp/preprocessing.py`) executes:
1. **Lowercasing & Normalization**
2. **Punctuation Cleaning**
3. **Word Tokenization** (`\b[a-zA-Z]{2,}\b`)
4. **Stopword Removal** (NLTK 179-word list)
5. **Morphological Lemmatization** (WordNetLemmatizer + driving domain lemma map)

---

## CHAPTER 12: FEATURE EXTRACTION VIA TF-IDF VECTORIZATION

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

- Unigrams and Bigrams (`ngram_range=(1, 2)`)
- Sublinear TF scaling (`sublinear_tf=True`)

---

## CHAPTER 13: MACHINE LEARNING CLASSIFICATION: LOGISTIC REGRESSION

Multinomial Logistic Regression with L2 regularization inside a Scikit-Learn `Pipeline`:
```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))
])
```

---

## CHAPTER 14: DATASET DESCRIPTION & ANALYSIS

`nlp/dataset.csv` contains **300 balanced samples** across 5 categories ($60$ samples each):
- FATIGUE (60 samples, 20%)
- ALERT (60 samples, 20%)
- REST_REQUEST (60 samples, 20%)
- STRESS (60 samples, 20%)
- NEUTRAL (60 samples, 20%)

---

## CHAPTER 15: MODEL TRAINING, EVALUATION & EXPERIMENTAL RESULTS

Evaluated on an independent stratified 20% test split (60 samples):
- **Accuracy:** `90.00%` (54/60 correct)
- **Macro Precision:** `90.49%`
- **Macro Recall:** `90.00%`
- **Macro F1-Score:** `90.12%`

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **ALERT** | `100.00%` | `100.00%` | `100.00%` | 12 |
| **FATIGUE** | `84.62%` | `91.67%` | `88.00%` | 12 |
| **NEUTRAL** | `90.91%` | `83.33%` | `86.96%` | 12 |
| **REST_REQUEST** | `100.00%` | `91.67%` | `95.65%` | 12 |
| **STRESS** | `76.92%` | `83.33%` | `80.00%` | 12 |

---

## CHAPTER 16: SAFETY RECOMMENDATIONS & GUIDANCE ENGINE

Contextual guidance generated for each predicted state with non-medical safety disclaimers.

---

## CHAPTER 17: IMPLEMENTATION & CODE EXPLANATION

Walkthrough of:
- `main.py`: Webcam vision detection loop
- `nlp/preprocessing.py`: Text pipeline
- `nlp/train_model.py`: Model training
- `nlp/fatigue_classifier.py`: Inference engine
- `app.py`: Integrated Streamlit dashboard

---

## CHAPTER 18: SYSTEM INTERFACE & OUTPUT DEMONSTRATION

- **Module 1:** Live OpenCV camera feed with convex eye contours and red alert overlay.
- **Module 2:** Streamlit interactive NLP text analyzer with preprocessing token display and confidence bars.

---

## CHAPTER 19: TESTING & OBSERVATIONS

Systematic test suite:
- Computer Vision: 7 test cases (TC-01 to TC-07) passing.
- NLP Generalization: 6 out-of-sample test cases (TC-08 to TC-13) passing.

---

## CHAPTER 20: ADVANTAGES

1. Contactless and non-intrusive.
2. Dual-modality protection (physical symptoms + cognitive self-reports).
3. False-alarm resilience against natural blinking.
4. Smooth real-time execution at 25–30 FPS on standard CPUs.
5. Fully offline and self-contained.

---

## CHAPTER 21: LIMITATIONS

1. Optical degradation in total nighttime darkness (requires IR camera).
2. Occlusion from dark polarized sunglasses.
3. Manual text input in current prototype (future ASR needed).
4. Synchronous audio beep call.

---

## CHAPTER 22: FUTURE SCOPE

- Automatic Speech Recognition (ASR) via offline Whisper.
- Yawning detection (Mouth Aspect Ratio - MAR).
- Head pose tracking for distraction monitoring.
- Infrared (NIR) night-vision hardware.
- Vehicle CAN bus and embedded edge deployment (Raspberry Pi / Jetson).

---

## CHAPTER 23: CONCLUSION

DROWZY demonstrates that uniting Computer Vision and Natural Language Processing produces a more effective, non-invasive driver safety system that protects drivers against both physical microsleeps and cognitive exhaustion.

---

## CHAPTER 24: REFERENCES

1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* CVWW.
2. Kazemi, V., & Sullivan, J. (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees.* IEEE CVPR.
3. Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection.* IEEE CVPR.
4. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing (3rd ed.).* Pearson.
5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
7. Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal.
8. King, D. E. (2009). *Dlib-ml: A Machine Learning Toolkit.* JMLR.
