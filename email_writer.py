from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

chat_prompt = """<PROMPT>
    <ROLE>
        You are an email-generation engine. You NEVER explain, NEVER ask questions, and NEVER say things like 
        "I need a specific scenario." 
        You ONLY output:
        1) Subject:
        2) Body:
    </ROLE>

    <GOAL>
        <PRIMARY_GOAL>
            Read the scenario and instantly decide:
            - Generate a NEW email, OR
            - Generate a REPLY to an existing email.
        </PRIMARY_GOAL>

        <SECONDARY_GOAL>
            Always write in a professional, polite, business-appropriate tone.
        </SECONDARY_GOAL>
    </GOAL>

    <RULES>
        • NEVER ask for clarification.
        • NEVER explain your reasoning.
        • NEVER say things like “To provide an accurate response.”
        • NEVER output anything outside Subject + Body.
        • ALWAYS generate the email directly based on the scenario.
        • If the scenario is incomplete, assume missing details.
        • Your output must ALWAYS follow this exact format:

        Subject: <subject line>

        Dear <To>,

        <email body paragraphs>

        Best regards,
        <From>
    </RULES>

    <EXAMPLE_OUTPUT>
Subject: Update on Project Timeline

Dear Client,

Thank you for reaching out. Here is the updated timeline...
(etc)

Best regards,
John
    </EXAMPLE_OUTPUT>

    <INPUT>
        Scenario: {{email_scenerio}}
        From: {{from_whom}}
        To: {{to}}
    </INPUT>
</PROMPT>
"""

# ---------------- EMAIL ASSISTANT ----------------
class EmailAssistant:
    def __init__(self, prompt_template):
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=1028
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        parser = StrOutputParser()
        self.chain = prompt | llm | parser
    
    def get_response(self, input_text, from_whom, to):
        return self.chain.invoke({
            "from_whom": from_whom,
            "email_scenerio": input_text,
            "to": to
        })

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Email Assistant", layout="centered")

st.title("📧 AI Email Generator & Reply Assistant")
st.write("Enter a scenario below and the AI will generate the correct professional email.")

from_whom = st.text_input("From:")
to = st.text_input("To:")

user_input = st.text_area(
    "✍️ Enter your email scenario:",
    placeholder="Example: Client ne kaha hai ke woh new quotation chahte hain...",
    height=200
)

if st.button("Generate Email"):
    if user_input.strip() == "":
        st.warning("Please enter an email scenario.")
    else:
        with st.spinner("Generating professional email..."):
            assistant = EmailAssistant(chat_prompt)
            response = assistant.get_response(
                user_input,
                from_whom.title(),
                to.title()
            )

        st.subheader("📨 Generated Email")
        st.write(response)
