import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

# Initialize Wikipedia tool
wikipedia = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)

# Available OpenAI models
AVAILABLE_MODELS = {
    "GPT-4o": "gpt-4o",
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}


def chat_with_agent(message, history, model_name):
    """
    Process user message with the LangChain agent.

    Args:
        message: User's input message
        history: Chat history (managed by Gradio)
        model_name: Selected OpenAI model name

    Returns:
        Response string
    """
    try:
        # Check if API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            return (
                "⚠️ Please set your OPENAI_API_KEY in the .env file. "
                "Get your API key from "
                "https://platform.openai.com/api-keys"
            )

        # Initialize LLM with selected model
        llm = ChatOpenAI(
            model=AVAILABLE_MODELS[model_name],
            temperature=0.7
        )

        # Try to get Wikipedia context
        try:
            wiki_result = wikipedia_tool.run(message)
            context = f"Wikipedia information: {wiki_result}\n\n"
        except Exception:
            context = ""

        # Create prompt with context
        prompt = f"{context}Question: {message}\n\nAnswer:"

        # Get response from LLM
        response = llm.invoke(prompt)

        # Extract the content
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)

        return answer

    except Exception as e:
        error_msg = (
            f"❌ Error: {str(e)}\n\n"
            "Please check your API key and internet connection."
        )
        return error_msg


# Create Gradio interface with Blocks for better layout control
with gr.Blocks(title="LangChain Wikipedia Chat") as demo:
    gr.Markdown(
        """
        # 🤖 LangChain Wikipedia Chat

        Ask me anything! I can search Wikipedia to provide accurate,
        factual answers.

        **Powered by:** OpenAI + LangChain + Wikipedia
        """
    )

    # Model selector dropdown outside chat interface
    model_dropdown = gr.Dropdown(
        choices=list(AVAILABLE_MODELS.keys()),
        value="GPT-4o",
        label="Select OpenAI Model",
        info="Choose which model to use for responses"
    )

    # Chat interface
    chat_interface = gr.ChatInterface(
        fn=chat_with_agent,
        additional_inputs=[model_dropdown],
        examples=[
            ["What is quantum computing?"],
            ["Tell me about Marie Curie"],
            ["What is the history of the Internet?"],
            ["Explain artificial intelligence"]
        ]
    )

if __name__ == "__main__":
    print("🚀 Starting Gradio app...")
    print("📝 Make sure your OPENAI_API_KEY is set in the .env file")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
