import streamlit as st
from transformers import pipeline

# --- Agent Definitions ---

class ConversationAgent:
    def chat(self, user_input):
        return "I'm here for you. Please share more if you're comfortable."

class EmotionDetectionAgent:
    def __init__(self):
        # Load from trained model directory
        self.classifier = pipeline("text-classification", model="trained_emotion_model", tokenizer="trained_emotion_model")

    def detect_emotion(self, text):
        result = self.classifier(text)[0]
        return result["label"], result["score"]

class AlertAgent:
    def send_alert(self, emotion):
        if emotion.lower() in ["sadness", "anger", "fear"]:
            return True
        return False

class RecommendationAgent:
    def recommend(self, emotion):
        return {
            "joy": "Keep doing what makes you happy!",
            "sadness": "Try a deep breathing exercise: https://youtu.be/SEfs5TJZ6Nk",
            "anger": "Take a walk and focus on your breath.",
            "fear": "Try grounding techniques or talk to someone you trust.",
            "love": "Share kind words with someone today.",
            "surprise": "Reflect and write about it.",
            "neutral": "Practice mindfulness or journaling."
        }.get(emotion.lower(), "Take care of yourself and stay present.")

# --- Streamlit UI ---

st.set_page_config(page_title="Mental Health AI (Custom Model)", layout="centered")
st.title("🧠 Mental Health Support Assistant (Trained Model)")

st.markdown("Enter how you're feeling and let our AI assistant respond with support, emotion insight, and helpful suggestions.")

user_input = st.text_area("🗣️ How are you feeling today?", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        convo = ConversationAgent()
        emotion_detector = EmotionDetectionAgent()
        alert_agent = AlertAgent()
        recommender = RecommendationAgent()

        st.subheader("🤖 Chatbot Response")
        st.write(convo.chat(user_input))

        emotion, confidence = emotion_detector.detect_emotion(user_input)
        st.subheader("🎭 Detected Emotion")
        st.write(f"**{emotion.capitalize()}** (Confidence: {confidence:.2f})")

        if alert_agent.send_alert(emotion):
            st.error("🚨 Crisis emotion detected! An alert would be sent to a caregiver.")
        else:
            st.success("No crisis detected.")

        st.subheader("💡 Recommendation")
        st.info(recommender.recommend(emotion))
