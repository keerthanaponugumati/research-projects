# NeuroQ-Intent: EEG Motor Imagery Classification

**Classical (CSP + LDA) and Quantum-Kernel Machine Learning for Brain-Computer Interfaces**

> **Core Task:** Classify EEG trials as imagined hands movement or imagined feet movement.
> **Dataset:** MNE EEGBCI / PhysioNet EEG Motor Movement/Imagery data (Subject 1, Runs 6, 10, 14).
> **Current Status:** Classical baseline (CSP + LDA) established. Quantum-kernel SVM integration in progress.

---

## 1. Project Goal & Scientific Context
The goal of **NeuroQ-Intent** is to build a realistic prototype at the intersection of neurotechnology and quantum computing. The model uses real EEG data from a motor imagery task to predict whether a trial corresponds to imagined hands movement or imagined feet movement. 

Motor imagery decoding is a foundational element of brain-computer-interface (BCI) research. Reliable detection of imagined movement from EEG paves the way for assistive technologies, communication interfaces, and prosthetic control. 

**The Quantum Direction:** The quantum-computing component of this project is highly exploratory. It tests how compact EEG features can be mapped into a quantum feature space and classified using a quantum kernel. The scientific priority here is an honest, rigorous comparison between classical and quantum methods, avoiding quantum hype.

## 2. Dataset and Task
* **Source:** EEGBCI dataset loaded via MNE-Python.
* **Subject:** 1
* **Runs:** 6, 10, and 14
* **Classes:** Hands imagery vs. Feet imagery. *(Note: Raw dataset labels were remapped from generic 'T1/T2' to readable 'hands/feet' for clarity).*

## 3. The End-to-End Pipeline

| Step | Action | Reason |
| :--- | :--- | :--- |
| **1** | Install and import libraries | `mne` for EEG; `scikit-learn` for classical ML; `qiskit-machine-learning` for quantum-kernel classification. |
| **2** | Load EEG recordings | Downloaded continuous EDF files into an MNE raw EEG object. |
| **3** | Standardize channels/montage | Attached standard EEG electrode positions to represent spatial structure properly. |
| **4** | Rename labels | Converted abstract T1/T2 labels into hands/feet classes. |
| **5** | Reference and filter EEG | Applied an EEG reference and bandpass filtered (7–30 Hz) to isolate motor-imagery rhythms and reduce noise. |
| **6** | Create epochs | Segmented continuous EEG into short, labeled trials around each task event. |
| **7** | Create X and y arrays | Converted epochs into `X` (EEG data) and `y` (labels: 0=hands, 1=feet). |
| **8** | Train classical baseline | Built a CSP + LDA model. (CSP extracts spatial patterns; LDA classifies them). |
| **9** | Evaluate (Confusion Matrix) | Tested the model on held-out data to visualize correct/incorrect predictions. |
| **10** | Prepare quantum comparison | Next step: Extract compact CSP features to compare a classical SVM with a quantum-kernel SVM. |

## 4. Current Results & Quantum Integration
The current completed milestone is the classical **CSP + LDA baseline**. The held-out test set contained 12 trials (6 hands, 6 feet). The classical model classified all 12 correctly in this split.

*(Insert Confusion Matrix Image Here)*
> **Figure 1.** Confusion matrix for the classical CSP + LDA model. The model correctly predicted 6 hands trials and 6 feet trials, with zero errors in this held-out split.

| Model | Feature Method | Held-out Test Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **CSP + LDA** | CSP inside pipeline | 12 / 12 correct (current split) | Strong baseline result, but requires further cross-validation to prove universal generalization. |
| **Classical RBF SVM** | 2 CSP features | *In progress* | Serves as a fair classical comparison benchmark for the quantum model. |
| **Quantum Kernel SVM** | 2 CSP features scaled (0 to π) | *In progress* | Exploratory quantum-kernel comparison (not a claim of quantum advantage). |

## 5. Limitations & Next Steps
### Current Limitations
* The pipeline has currently been validated on a single subject.
* The 100% accuracy result is derived from a single train/test split.
* A strong result on hands vs. feet imagery may not transfer directly to harder classification tasks (e.g., left hand vs. right hand).
* **Disclaimer:** This model is a research prototype, not a clinical system or medical device.

### Immediate Research Roadmap
1. **Feature Extraction:** Run the CSP feature extraction step isolating 2 components.
2. **Classical Benchmark:** Train a classical RBF SVM on those 2 CSP features.
3. **Quantum Integration:** Train a quantum-kernel SVM on the same CSP features (scaled from $0$ to $\pi$).
4. **Analysis:** Create a final performance comparison table (CSP+LDA vs. Classical SVM vs. Q-Kernel SVM).
5. **Generalization:** Repeat the experiment on additional subjects to test cross-subject robustness.

## 6. Project Summary
> "NeuroQ-Intent explores the intersection of neurotechnology and quantum machine learning. I began by creating a reliable EEG motor-imagery decoding baseline using real EEGBCI data from MNE, focusing on hands versus feet imagery. The continuous signal was filtered into a motor-imagery-relevant frequency range and converted into labeled epochs. I trained a CSP + LDA classifier, where CSP extracted spatial EEG patterns and LDA performed the classification. In the current held-out split, this classical model achieved 100% accuracy. Moving forward, I am treating this result carefully as it relies on one subject and split. The next stage of the architecture is to use these same compact CSP features to benchmark a classical SVM directly against a Quantum-Kernel SVM."

---

## Appendix: Key Concepts Glossary

| Term | Meaning |
| :--- | :--- |
| **EEG** | A non-invasive recording of tiny electrical signals from the scalp. |
| **Motor Imagery** | The cognitive process of imagining movement without physically moving. |
| **Epoch** | A short, time-locked EEG segment around a specific task cue. |
| **Filtering** | Isolating useful frequency ranges while suppressing irrelevant signal noise. |
| **CSP (Common Spatial Pattern)** | A feature extraction method that finds EEG channel patterns maximizing the variance between two classes. |
| **LDA (Linear Discriminant Analysis)** | A classifier that separates feature patterns using a linear decision boundary. |
| **Quantum Kernel** | A similarity function created by mapping classical data features into a high-dimensional quantum feature space. |

## References and Tools
* [MNE-Python](https://mne.tools/stable/index.html) - EEGBCI dataset loader and motor imagery tools.
* [PhysioNet](https://physionet.org/) - EEG Motor Movement/Imagery dataset.
* [scikit-learn](https://scikit-learn.org/) - LDA, SVM, and evaluation tools.
* [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/) - Quantum kernel and QSVC implementation.
