# plugins/aeon-gemma3-270m/main.py
from pathlib import Path

from llama_cpp import Llama
from langchain.docstore.document import Document
from src.utils.conversation import saveConversation
from src.libs.messages import (print_plugin_message,
                               print_error_message)


def _ingest_conversation_turn(user_input, aeon_output, vectorstore, text_splitter, llama_embeddings):
    try:
        conversation_text = f"QUESTION: {user_input}\nANSWER: {aeon_output}"
        
        # Create a LangChain Document object
        conversation_document = Document(
            page_content=conversation_text,
            metadata={"source": "gemma-3-270m-it"}
        )
        
        # Split the document into chunks
        docs = text_splitter.split_documents([conversation_document])
        success, failed = 0, 0
        for i, chunk in enumerate(docs, start=1):
            try:
                vectorstore.add_documents([chunk])
                success += 1
               
            except Exception as e:
                failed += 1
                print_error_message(f" Failed on chunk {i}: {e}")

        
    except Exception as e:
        print_error_message(f"Failed to ingest conversation turn: {e}")

def run_plugin(args: str, **kwargs) -> dict:
    plugin_config = kwargs.get('plugin_config')
    plugin_name = plugin_config.get("plugin_name")
    
    vectorstore = kwargs.get('vectorstore')
    text_splitter = kwargs.get('text_splitter')
    llama_embeddings = kwargs.get('llama_embeddings')
    conversation_filename = kwargs.get('conversation_filename')
    current_memory_path = kwargs.get('current_memory_path')
    current_chat_history=kwargs.get("current_chat_history")
    
    # The first argument is the plugin command, so we slice the args list.
    if not args:
        print_error_message(f"Usage: /{plugin_name} <PROMPT>")


    prompt = args
    
    model_path = plugin_config.get("model_path")
    
    plugin_dir = Path(__file__).parent
    model_file_path = plugin_dir / model_path

    if not model_file_path.exists():
        print_error_message(f"Model file not found at: {model_file_path}")

    try:
        print_plugin_message(f"Loading model: {model_file_path.name}...")
        llm = Llama(
            model_path=str(model_file_path),
            n_ctx=4096,
            verbose=False,
        )

        print_plugin_message("Generating response...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            stream=False
        )

        message_content = response['choices'][0]['message']['content']

        _ingest_conversation_turn(
            prompt,
            message_content,
            vectorstore,
            text_splitter,
            llama_embeddings
        )

        saveConversation(
            prompt,
            message_content,
            plugin_name,
            current_memory_path,
            conversation_filename
        )

        current_chat_history.append(
            {"user": prompt, plugin_name: prompt, "source": plugin_name}
        )

        print_plugin_message(f"\033[1;33m[GEMMA3]\033[0m: {message_content}")

    except Exception as e:
        print_error_message(f"An error occurred during model inference: {e}")