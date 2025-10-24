# Whats Not App
This project is a WhatsApp bot built using the @whiskeysockets/baileys library for WhatsApp communication and a FastAPI backend for message classification. The bot receives messages, sends them to a FastAPI endpoint for classification (categorizing messages into "friends," "work," or "Other" with urgency levels "Low," "Medium," or "High"), and logs the results securely. The classification system uses a local document store, FAISS for vector search, and the Nous-Hermes-2-Mistral-7B-DPO model for natural language processing.

# Features

- WhatsApp Integration: Connects to WhatsApp via QR code scanning for authentication.
- Message Classification: Classifies incoming messages using a FastAPI endpoint powered by a local LLM (Nous-Hermes-2-Mistral-7B-DPO) and a FAISS vector store for context retrieval.
- Safe Logging: Stores classified messages in a secure JSONL log file.
- Document Processing: Processes local text and PDF documents to provide context for classification.
- Error Handling: Robust error handling for connection issues, message processing, and API failures.
- Modular Design: Separates WhatsApp bot logic (index.js) and classification logic (watch.py).

# Prerequisites

- Node.js: Version 16 or higher
- Python: Version 3.8 or higher
- WhatsApp Account: For QR code scanning and bot operation
- GPU (Optional): For faster model inference with GPT4All
- Documents: Place .txt or .pdf files in the project root for context-based classification
-
# Installation
1. Clone the repo
 ```bash
git clone https://github.com/c137v8/Whats-Not-App-AI
cd Whats-Not-App-AI
```
2. Install Node.js Dependencies
```bash
npm install
```
3. Install Python Dependencies
```bash
pip install -r requirements.txt
```
4. Set Up Environment
```bash
PHONENUMBER=<your-phone-number>  # Your WhatsApp number in international format (e.g., 1234567890)
```
5. Prepare Documents
Place .txt or .pdf files in the project root (or a specified folder) to provide context for message classification.

# Usage
1. Start the FastAPI Server
```bash
python watch.py
```
This starts the FastAPI server on http://localhost:8000. The server loads documents, initializes the FAISS vector store, and prepares the GPT4All model.

2. Start the WhatsApp Bot
```bash
node index.js
```
- The bot will display a QR code in the terminal.
- Open WhatsApp on your phone, go to Settings → Linked Devices → Link a Device, and scan the QR code.
- Once connected, the bot will log incoming messages, send them to the FastAPI endpoint for classification, and reply to yourself with high-urgency messages.

3. Message Processing
- The bot listens for incoming WhatsApp messages (excluding your own).
- Messages are sent to the FastAPI /classify endpoint.
- The FastAPI server classifies messages based on content and document context, returning a JSON object with category, urgency, and reason.
- High-urgency messages trigger a reply to the phone number specified in the .env file.
- Classified messages are logged to ./.data/message_logs.jsonl and ./classified_messages.log.

# Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.
