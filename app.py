"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21386914
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: ChatGPT, GitHub Copilot, Codex
#   - How you used them: I used ChatGPT and GitHub Copilot to help generate code snippets, provide suggestions for function implementations, and assist with debugging. I also used them to clarify concepts related to the SMS spam detection model and the Streamlit framework.
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   YOUR_ANSWER: The primary business user of this app is a security analyst who needs to quickly identify and filter out spam or scam-like messages from inbound SMS communications.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   YOUR_ANSWER: My app achieved 99.00% accuracy on the 100-sample test.
#   Accuracy alone is insufficient because errors have different business costs.
#   False positives could block bank messages or one-time codes, while false
#   negatives could expose users to spam or scams.
#   The app should therefore support initial screening rather than automatically
#   block messages on its own.
# =============================================================================

import streamlit as st
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21386914"  # STUDENT: replace with your real student ID
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    # STUDENT: call the pipeline on this one text, map the raw label with
    # map_label(...), and return the display label and the confidence score.
    result = classifier(text)[0]  # Call the classifier pipeline on the text
    display_label = map_label(result['label'])  # Map the raw label to display label
    score = result['score']  # Extract the confidence score
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    # STUDENT: accuracy = (number of matches) / (number of samples).
    correct = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == true_labels[i]:
            correct += 1
    accuracy = correct / len(pred_labels)  # Calculate accuracy
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    # STUDENT: also show Student ID and COURSE_SECRET on the page.
    st.write("Student ID:", STUDENT_ID)
    st.write("Course secret:", COURSE_SECRET)
    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except st.errors.StreamlitSecretNotFoundError:
        hf_token = None

    # STUDENT: load the model with load_model(hf_token)
    classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        # STUDENT: if the text box is not empty, call classify_text,
        # then show the display label and the confidence.
        if alert_text.strip():
            display_label, confidence = classify_text(classifier, alert_text)
            st.write("Display Label:", display_label)
            st.write("Confidence:", confidence)
        else:
            st.warning("Please enter an SMS message.")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        # STUDENT: call load_test_samples(), classify each message with
        # classify_text, then compute_accuracy. Show the accuracy and a
        # results table with at least these columns:
        # # | Text (short) | Ground truth | Prediction | Confidence | Match
        texts, true_labels = load_test_samples()
        pred_labels = []
        confidence_scores = []

        for text in texts:
            display_label, confidence = classify_text(classifier, text)
            pred_labels.append(display_label)
            confidence_scores.append(confidence)

        accuracy = compute_accuracy(pred_labels, true_labels)
        st.write(f"Accuracy: {accuracy:.2%}")

        results = []

        for i in range(len(texts)):
            row = {
                "#": i + 1,
                "Text (short)": texts[i][:80],
                "Ground truth": true_labels[i],
                "Prediction": pred_labels[i],
                "Confidence": confidence_scores[i],
                "Match": pred_labels[i] == true_labels[i],
            }
            results.append(row)

        st.dataframe(results)


if __name__ == "__main__":
    main()
