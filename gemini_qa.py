import os
import google.generativeai as genai

# Set your Gemini API key here
genai.configure(api_key="your-api-key")

print("Using API key:", os.getenv("api_key"))

# Initialize the chat model
model = genai.GenerativeModel("models/gemini-2.5-pro")
system_prompt={
    "author":"system",
    "content": (
        "You are a helpful, accurate, and safe medical assistant."
        "Always provide factual information, include disclaimers."
    )
}

chat_session = model.start_chat()

def gemini_chat(message):
    try:
        response = chat_session.send_message(message)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def main():
    print("Gemini Chatbot: Type 'exit' to end.")
    while True:
        user_input=input("You: ")
        if user_input.lower()=='exit':
            print("Goodbye!")
            break
    response=chat_session.send_message([system_prompt])
if __name__ == "__main__":
    main()
