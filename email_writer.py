from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

chat_prompt = """<PROMPT>
    <ROLE>
        You are an expert email assistant specialized in analyzing scenarios and generating professional emails or replies based on the given input.
    </ROLE>

    <GOAL>
        <PRIMARY_GOAL>
            Carefully analyze the provided text (scenario). Decide whether the user needs:
            1) A completely new email to be generated, OR
            2) A professional reply to an existing email.
        </PRIMARY_GOAL>

        <SECONDARY_GOAL>
            Always write emails in a clear, polite, and professional tone.
            Return both:
            - A suitable Subject line
            - A complete Body message
        </SECONDARY_GOAL>
    </GOAL>

    <INSTRUCTIONS>
        • The input text will be a scenario written by the user.  
        • Understand the context deeply and determine the correct action (new email or reply).  
        • If it's a reply: respond directly to the sender’s message.  
        • If it's a new email: generate a fresh, properly structured email.  
        • Keep the tone professional, respectful, and concise.  
        • Output must contain two sections:
            1) Subject
            2) Body 
    </INSTRUCTIONS>

    <INPUT_EXAMPLE>
        "Client ne kaha hai ke woh project timeline ka update chahte hain.  
        Unko professional reply likh kar do."
    </INPUT_EXAMPLE>

    <OUTPUT_EXAMPLE>
        Subject: Update on Project Timeline
        
        Dear {{to}},  

        I hope you are doing well. As requested, I am sharing the latest update on the project timeline.  
        Our team has completed the current milestones and we are moving toward the next phase as planned.  
        I will continue to keep you updated with progress.  

        Best regards,

        {{from_whom}}
    </OUTPUT_EXAMPLE>

    <INPUT>
        Scenario: {email_scenerio}
        From: {from_whom}
        To: {to}
    </INPUT>
</PROMPT>

"""

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
