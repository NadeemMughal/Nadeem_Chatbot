import os
import time
import uuid
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
import gradio as gr
from datetime import datetime
import pytz
from dateutil import relativedelta

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "nadeem-knowledge-base-google-embeddings"
index = pc.Index(index_name)

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Pakistan Time Tool
def get_pakistan_time():
    """Returns the current time in Pakistan (PKT, UTC+5)."""
    pkt_tz = pytz.timezone("Asia/Karachi")
    current_time = datetime.now(pkt_tz).strftime("%I:%M %p, %B %d, %Y")
    return f"The current time in Pakistan is {current_time}."

# Time Elapsed Since Role Start
def calculate_time_in_role():
    """Calculates time elapsed since starting as Associate AI/ML Engineer (Nov 1, 2024)."""
    start_date = datetime(2024, 11, 1, tzinfo=pytz.UTC)
    current_date = datetime.now(pytz.UTC)
    delta = relativedelta.relativedelta(current_date, start_date)
    months = delta.months + (12 * delta.years)
    days = delta.days
    if months == 0:
        return f"Muhammad Nadeem has been an Associate AI/ML Engineer for {days} days since November 1, 2024."
    return f"Muhammad Nadeem has been an Associate AI/ML Engineer for {months} months and {days} days since November 1, 2024."

# History Manager Class
class HistoryManager:
    def __init__(self, max_pairs=10):
        self.max_pairs = max_pairs  # Maximum number of Human-AI message pairs to store
        self.history = []  # List to store recent message pairs
        self.summaries = []  # List to store summaries of older messages

    def add_message(self, human_msg, ai_msg):
        """Add a new Human-AI message pair to the history."""
        self.history.append({"role": "user", "content": human_msg})
        self.history.append({"role": "assistant", "content": ai_msg})

        # If history exceeds max_pairs, summarize the oldest pair
        while len(self.history) // 2 > self.max_pairs:
            # Take the oldest Human-AI pair (first two messages)
            pair_to_summarize = self.history[:2]
            self.history = self.history[2:]  # Remove the oldest pair

            # Summarize the pair
            summary_prompt = f"Summarize the following conversation in as you needed for further follow ups:\nUser: {pair_to_summarize[0]['content']}\nAssistant: {pair_to_summarize[1]['content']}"
            summary = ""
            for chunk in llm.stream(summary_prompt):
                summary += chunk.content
            self.summaries.append(summary)

    def get_history(self):
        """Return the current history as a list of (human, ai) pairs."""
        return [(self.history[i]["content"], self.history[i + 1]["content"])
                for i in range(0, len(self.history), 2)]

    def get_full_context(self):
        """Return the summarized history and recent history as a string for context."""
        # Combine summaries
        summary_text = "\n\n".join(self.summaries) if self.summaries else "No previous summaries."
        # Format recent history
        history_str = ""
        for i in range(0, len(self.history), 2):
            history_str += f"User: {self.history[i]['content']}\nAssistant: {self.history[i + 1]['content']}\n\n"
        return f"Previous Summaries:\n{summary_text}\n\nRecent Conversation:\n{history_str}"

# Global dictionary to store histories by session ID
session_histories = {}

# Updated System Prompt
system_prompt = """
    Greetings! I am the Virtual Assistant of Muhammad Nadeem, your friendly guide to all things related to Nadeem’s expertise as a Generative AI Engineer and Machine Learning Specialist. Nadeem started his professional journey as an Associate AI/ML Engineer on November 1, 2024, and I’m here to share insights about his skills, projects, and achievements in a warm, professional, and engaging way.

    **My Capabilities**:
    - Answer questions about Nadeem’s experience, technical skills, or projects using provided context and conversation history.
    - When user want to know about experience in metaviz then use tool get_pakistan_time and calculate time from it.
    - Calculate how long Nadeem has been in his current role when queried about his experience.
    - Nadeem is expert in Voice Agents and Chatbots also deploy these on cloud, including code and no code platforms
    - for chatbot Nadeem Expert in code platforms (langchain, langgraph, crew ai), and no code platforms (closebot, n8n etc).
    - for voice agents nadeem has experienced in Elevenlabs, retell ai, Synthflow also developed voice agents from scratch using langchain, and API's like Deepgram, Assembly ai, Elevenlabs etc.
    - Nadeem has also works with CRM's like GoHighLevel, ServiceTitan (job management software), Cliniko, etc 
    - Nadeem has also experience in some other technologies mcp, manus ai, FastAPI, Agent_to_Agent protocol, Google agents development kit and OpenAI Agents SDK and integration platforms make.com, zapier and cassidy. 

    **Instructions**:
    - Respond in a concise, conversational tone, avoiding overly technical jargon unless requested.
    - If asked about Nadeem’s role duration or experience, use the time elapsed tool and integrate its output into a polished, conversational response (e.g., instead of just stating the duration, explain its significance or context).
    - Use context from Pinecone and conversation history for accurate, personalized answers.
    - If information is missing, say so politely and offer a general response or suggest related topics. which I mentioned.
    - When providing Nadeem’s information, first highlight his experience with chatbots and voice agents, then discuss integrations between platforms using tools like Make.com and Zapier, Deployment platforms AWS, GCP, Azure, Vertex and finally emphasize his deep knowledge of ML and DL algorithms.
    - Nadeem holds a degree in Data Science from GIFT University, completed in 2024.
    - Where appropriate, use bullet points for clarity to make information easily understandable for leads.
    - Include a follow-up question at the end of each response to engage the user further.
    - At the end of the conversation, include a polite farewell greeting to the lead.
    - Don't provide current time as your own until user asked.
    - Don't expose that which LLM you are using and which embedding model you using and where you stored embeddings.
    - When you want to give information about nadeem make sure highlight Chabots, Voice Agents, Deployment like (cloud platform AWS, Azure, GCP, Vertex AI, Digital Ocean, Vercel), CRM's, integration platforms. 
    - If user want to end conversation then end politely end with good message.
    - when user want to know about Nadeem Achievements then share this link of research article link with beautification "https://arxiv.org/abs/2412.18199". in this research article our focus is too extract handwritten medicine name from doctor handwritten prescription using transformer based models.
    - linkedin url is https://www.linkedin.com/in/muhammad-nadeem-ml-engineer-researcher/ and github url https://github.com/NadeemMughal.

    **Experience**
    - Nadeem Started his journey as a Freelancer during his graduation, furthermore he is done 2 month Internship with Comsats University Teacher Dr. Muhammad Adeel Rao, here worked on Generative AI Field and cover many basic concepts.
    - After that Joined Metavizpro on 01 November 2024 as a AI/ML Engineer currently worked here on Generative AI, where he explored many more things that are related to chatbots, voice agents, integrations between different platforms using integration tools.

    **Context**:
    {context}

    **Conversation History**:
    {history}
"""

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "history", "query", "tool_output"],
    template=system_prompt + "\nUser Query: {query}\nTool Output (if applicable): {tool_output}\nAnswer:"
)

# Retrieve Context from Pinecone
def retrieve_context(query, top_k=2):
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    context = "\n\n".join(match["metadata"]["text"] for match in results["matches"])
    return context

# Generate Streaming LLM Response
def get_response(query, previous_history, session_id):
    try:
        tool_output = ""
        if "time" in query.lower() or "current time" in query.lower():
            tool_output = get_pakistan_time()
        elif "how long" in query.lower() or "experience" in query.lower() or "role" in query.lower():
            tool_output = calculate_time_in_role()

        context = retrieve_context(query)
        history_mgr = session_histories.get(session_id, HistoryManager())
        history_str = history_mgr.get_full_context()

        formatted_prompt = prompt_template.format(
            context=context,
            history=history_str,
            query=query,
            tool_output=tool_output if tool_output else "None"
        )

        response = ""
        for chunk in llm.stream(formatted_prompt):
            response += chunk.content
            yield response

        # After generating the response, add the message pair to history
        history_mgr.add_message(query, response)
        session_histories[session_id] = history_mgr
    except Exception as e:
        yield f"Error: {str(e)}"

# Custom CSS
custom_css = """
    * {
        box-sizing: border-box;
    }
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f4f8;
        margin: 0;
    }
    #navbar {
        background-color: #ffffff;
        padding: 4vw;
        border-right: 1px solid #e0e0e0;
        min-width: 200px;
        flex: 1;
        text-align: center;
        overflow: auto;
    }
    #navbar h1 {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    #navbar h2 {
        font-size: 1rem;
        color: #7f8c8d;
        margin: 0.5rem 0;
    }
    #navbar p {
        font-size: 0.9rem;
        color: #34495e;
        margin: 1rem 0;
    }
    #navbar a {
        color: #3498db;
        text-decoration: none;
    }
    #navbar a:hover {
        text-decoration: underline;
    }
    #navbar .button {
        display: inline-block;
        background-color: #3498db;
        color: white;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        text-decoration: none;
        font-size: 0.9rem;
        border: none;
        cursor: pointer;
    }
    #navbar .button:hover {
        background-color: #2980b9;
    }
    .profile-description {
        text-align: justify;
        margin: 1rem 0;
    }
    .email-display {
        text-align: justify;
        overflow-wrap: break-word;
        max-width: 100%;
        padding: 0.5rem;
        background-color: #f1f5f9;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #34495e;
    }
    #header {
        background-color: #3498db;
        color: white;
        padding: 2vw;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        border-bottom: 1px solid #2980b9;
    }
    #footer {
        background-color: #2c3e50;
        color: white;
        padding: 1.5vw;
        text-align: center;
        width: 100%;
    }
    #footer a {
        color: #3498db;
        margin: 0 1vw;
        text-decoration: none;
    }
    #footer a:hover {
        text-decoration: underline;
    }
    .gradio-container {
        max-width: 90%;
        margin: 0 auto;
        padding: 1rem;
    }
    .chatbot-container {
        flex: 3;
        min-width: 300px;
        overflow: auto;
    }
    @media (max-width: 768px) {
        #navbar {
            display: none;
        }
        #header {
            font-size: 1.4rem;
            padding: 1rem;
        }
        #footer {
            position: static;
            padding: 1rem;
        }
        .gradio-container {
            max-width: 100%;
            padding: 0.5rem;
        }
        #navbar h1 {
            font-size: 1.2rem;
        }
        #navbar h2 {
            font-size: 0.9rem;
        }
        #navbar p {
            font-size: 0.8rem;
        }
        #navbar .button {
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }
        .chatbot-container {
            width: 100%;
        }
    }
    @media (max-width: 480px) {
        #header {
            font-size: 1.2rem;
        }
        #navbar {
            padding: 1rem;
        }
        #navbar h1 {
            font-size: 1rem;
        }
        #navbar h2 {
            font-size: 0.8rem;
        }
        #navbar p {
            font-size: 0.7rem;
        }
        #navbar .button {
            font-size: 0.7rem;
            padding: 0.3rem 0.6rem;
        }
    }
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css,
               theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="slate")) as demo:
    # Header
    gr.HTML("""
            <div id="header">
                Muhammad Nadeem Virtual Assistant
            </div>
        """)

    # State to store session ID
    session_id_state = gr.State(value=None)

    with gr.Row():
        # Sidebar with profile
        with gr.Column(scale=1, min_width=200):
            gr.HTML("""
                    <div id="navbar">
                        <h1>Muhammad Nadeem</h1>
                        <h2>AI/ML Engineer</h2>
                        <p class="profile-description">Passionate to building innovative AI solutions with expertise in deep learning, LLM's and full-stack AI systems. Started as AI/ML Engineer on November 1, 2024.</p>
                        <a href="https://www.linkedin.com/in/muhammad-nadeem-ml-engineer-researcher/" target="_blank" class="button">LinkedIn</a>
                        <a href="https://github.com/NadeemMughal" target="_blank" class="button">GitHub</a>
                        <p>Published an article on advanced AI techniques.</p>
                        <a href="https://arxiv.org/abs/2412.18199" target="_blank" class="button">Read Article</a>
                        <p class="email-display"><strong>nadeem.dev51@gmail.com</strong></p>
                    </div>
                """)

        # Chatbot interface
        with gr.Column(scale=3, min_width=300, elem_classes=["chatbot-container"]):
            chatbot = gr.Chatbot(type="messages", label="Chat with Nadeem's Virtual Assistant", value=[])
            user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            submit_button = gr.Button("Send")

            def initialize_session(session_id):
                """Initialize a new session ID if none exists."""
                if session_id is None:
                    session_id = str(uuid.uuid4())
                return session_id

            def submit_message(message, chat_history, session_id):
                if not message:  # Skip if message is empty
                    yield chat_history, "", session_id
                    return

                # Initialize session ID if not set
                session_id = initialize_session(session_id)

                # Add user's message to chat history
                chat_history = chat_history + [{"role": "user", "content": message}]

                # Initialize assistant's response
                assistant_response = ""
                chat_history = chat_history + [{"role": "assistant", "content": assistant_response}]

                # Yield initial state with empty assistant response and clear user_input
                yield chat_history, "", session_id

                # Stream the response with delay
                for response_chunk in get_response(message, None, session_id):
                    assistant_response = response_chunk
                    chat_history[-1]["content"] = assistant_response
                    time.sleep(0.5)  # Slow down streaming
                    yield chat_history, "", session_id

            submit_button.click(
                submit_message,
                inputs=[user_input, chatbot, session_id_state],
                outputs=[chatbot, user_input, session_id_state]
            )
            user_input.submit(
                submit_message,
                inputs=[user_input, chatbot, session_id_state],
                outputs=[chatbot, user_input, session_id_state]
            )

    # Footer
    gr.HTML("""
            <div id="footer">
                Connect with me:
                <a href="https://www.linkedin.com/in/muhammad-nadeem-ml-engineer-researcher/" target="_blank">LinkedIn</a> |
                <a href="https://github.com/NadeemMughal" target="_blank">GitHub</a>
            </div>
        """)

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=True)