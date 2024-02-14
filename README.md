# LangChain Chatbot

Welcome to the LangChain Chatbot project! This repository contains a Flask-based application designed to showcase the integration of LangChain with OpenAI's powerful language models to create an interactive chatbot. Utilizing advanced AI techniques, this chatbot can engage in meaningful conversations, provide answers to queries, and much more.

## Features

- **Chat Interface**: Leveraging `ChatOpenAI` for dynamic conversation capabilities.
- **Question Answering**: Implements `RetrievalQA` for efficient information retrieval and question answering.
- **Document Handling**: Utilizes `CSVLoader` for document management, making it easy to load and retrieve information from CSV files.
- **Embeddings and Vector Stores**: Incorporates `OpenAIEmbeddings` and `FAISS` for managing embeddings and vector storage, enhancing the chatbot's ability to understand and generate relevant responses.
- **Conversational Memory**: Features `ConversationBufferMemory` to maintain context and continuity across conversations.

## Installation

To set up the LangChain Chatbot on your local machine, follow these steps:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/langchain-chatbot.git
   cd langchain-chatbot
   ```

2. **Install Dependencies**

   Ensure you have Python 3.6+ installed. Then, install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. **Environment Variables**

   Set up the necessary environment variables (e.g., API keys for OpenAI):

   ```sh
   export OPENAI_API_KEY='your_openai_api_key_here'
   ```

4. **Run the Application**

   Start the Flask server:

   ```sh
   flask run
   ```

   Your chatbot is now accessible at `http://localhost:5000`.

## Usage

Send a POST request to `http://localhost:5000/chat` with a JSON payload containing your message. For example:

```json
{
  "message": "Hello, chatbot!"
}
```

The chatbot will respond with an appropriate reply based on its training and capabilities.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

