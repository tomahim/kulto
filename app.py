import os
from dotenv import load_dotenv
import gradio as gr
import wikipedia as wiki_api
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

# Initialize Wikipedia tool
wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
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
        Response string with tool call details
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
        wiki_used = False
        wiki_result = None
        wiki_pages = []
        try:
            # Get Wikipedia results
            wiki_result = wikipedia_tool.run(message)

            # Extract page titles by searching Wikipedia
            search_results = wiki_api.search(message, results=2)

            # Build page URLs
            for i, page_title in enumerate(search_results[:2], 1):
                try:
                    page_url = (
                        f"https://en.wikipedia.org/wiki/"
                        f"{page_title.replace(' ', '_')}"
                    )
                    wiki_pages.append({
                        'title': page_title,
                        'url': page_url,
                        'index': i
                    })
                except Exception:
                    continue

            context = f"Wikipedia information: {wiki_result}\n\n"
            wiki_used = True
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

        # Format response with tool call details and citations
        if wiki_used and wiki_pages:
            summary = (wiki_result[:200] + "...") if len(
                wiki_result) > 200 else wiki_result

            # Build citations footer
            citations = "\n\n---\n\n**Sources:**\n\n"
            for page in wiki_pages:
                citations += (
                    f"[{page['index']}] "
                    f"[{page['title']}]({page['url']})\n"
                )
            
            tool_details = f"""
<details>
<summary>🔧 Tool Calls</summary>

**Tool Used:** Wikipedia Search
**Query:** {message}
**Result Summary:** {summary}

</details>

---

"""
            final_answer = tool_details + answer + citations
        elif wiki_used:
            # Wikipedia was used but no pages extracted
            summary = (wiki_result[:200] + "...") if len(
                wiki_result) > 200 else wiki_result
            tool_details = f"""
<details>
<summary>🔧 Tool Calls</summary>

**Tool Used:** Wikipedia Search
**Query:** {message}
**Result Summary:** {summary}

</details>

---

"""
            final_answer = tool_details + answer
        else:
            final_answer = (
                "ℹ️ *No Wikipedia search performed*\n\n---\n\n" + answer
            )

        return final_answer

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
