from langchain_google_genai import ChatGoogleGenerativeAI

def load_light_llm():
    system_instruction = (
        "You are a helpful and knowledgeable medical assistant. "
        "Use the provided documents as context, but if they lack specific treatment details or drug names, "
        "you may provide accurate and up-to-date medical information from your own knowledge. "
        "Focus on answering the user's question clearly and directly, including treatment names or drug recommendations when appropriate."
        "If asked by the user, you can also help with general stuff i.e. non-medical"
    )

    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.7,
        max_output_tokens=256,
        model_kwargs={"system_instruction": system_instruction}
    )

