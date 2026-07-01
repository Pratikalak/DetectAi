import streamlit as st

st.set_page_config(page_title="About DetectAI", layout="centered")

if st.button("← Back to DetectAI"):
    st.switch_page("app.py")

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("About DetectAI")

st.header("The Problem")
st.markdown("""
AI-generated deepfake images are increasingly used as instruments of impersonation, harassment,
and financial scams across social media platforms. The technology required to produce convincing
synthetic facial imagery has become widely accessible, lowering the barrier for misuse.
Human perception alone can no longer reliably distinguish high-quality manipulated images from
authentic photographs. This creates real and measurable harm for individuals who encounter
fabricated content depicting themselves or others without consent.
""")

st.header("Why We Built This")
st.markdown("""
DetectAI was developed with particular concern for women aged 18 to 35, who are
disproportionately targeted by deepfake-based impersonation and non-consensual synthetic imagery
on social media. The tool aims to provide an accessible, non-technical means of flagging
potentially manipulated facial images without requiring forensic expertise or specialised software.
""")

st.header("How It Works")
st.markdown("""
DetectAI uses a deep learning model trained to identify visual patterns associated with
AI-generated facial manipulation. A user uploads a facial image and receives a classification
verdict alongside a confidence score indicating the model's certainty. A confidence score
below 70% triggers an explicit warning, and results in that range should be treated as
inconclusive rather than reliable.
""")

st.header("Limitations and Bias")
st.markdown("""
- The model was trained on a specific dataset and may not perform equally across all
  demographics, skin tones, lighting conditions, or deepfake generation techniques.
- It is not a forensic tool and its output should not be used as sole evidence in any
  legal, journalistic, or investigative context.
- Results are probabilistic, not definitive. A classification of real or fake reflects a
  statistical likelihood, not a verified finding.
- Newer deepfake generation methods not represented in the training data may evade detection
  entirely or produce unreliable results.
""")

st.header("A Note on Use")
st.markdown("""
DetectAI is a research prototype developed for educational purposes. It is intended to
support awareness and prompt further investigation, not to replace it. Any result produced
by this tool should be considered as one signal among others, and users are encouraged to
seek additional verification before drawing conclusions about an image's authenticity.
""")

st.markdown("""
---
**Contact:** pratikalakarki@gmail.com
""")
