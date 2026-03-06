# ✈️ Multimodal Multi-Agent Travel Support Bot

A multimodal AI assistant that helps users **search and book flights, hotels, tours**, and **reserve airport shuttle services**, powered by a multi-agent architecture built with [LangGraph](https://github.com/langchain-ai/langgraph).

## 💡 Key Features

- 🤖 **Multi-Agent System**: Uses a Supervisor Agent to coordinate:
  - **Flight Agent** – Searches and books flight tickets
  - **Hotel Agent** – Searches and books hotels
  - **Tour Agent** – Recommends and searches for tours
  - **Shuttle Agent** – Finds and reserves airport shuttle services
- 🧠 **Multimodal Input**: Supports user input via **text**, **image**, or **audio**
- 📦 **Real-Time Database Integration**:
  - Fetches available flights, hotels, tours, and shuttles based on user query
  - Manages user balance and stores booking data in **MongoDB**
- 🔁 **Transactional Booking Flow**: Ensures atomicity between booking and payment
- � **Memory System**: Stores and recalls user preferences across conversationcs

## 📌 Architecture
![image](https://github.com/user-attachments/assets/c9a57cb8-b155-4c9a-a94f-0b1cb00f6170)

## 🛠️ Technologies Used

- **LangGraph** – Multi-agent coordination
- **LangChain** – Agent logic & tool calling
- **MongoDB** – Booking & user data storage
- **Chainlit** – Web interface
- **Python** – Core logic
- **Gemini** – LLM for agent reasoning
- **Groq** – Speech-to-text & vision models

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/NgNMinh/Multimodal-Multi-Agent-Travel-Support-Bot-.git
cd Multimodal-Multi-Agent-Travel-Support-Bot-
```

### 2. Create and activate virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup MongoDB
Make sure MongoDB is running on your local machine:
```bash
# Default connection: mongodb://localhost:27017
# Database name: flight_booking
```

### 5. Configure environment variables
Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
TOGETHER_API_KEY=your_together_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key

# Optional: Override default MongoDB settings
MONGODB_URI=mongodb://localhost:27017
DB_NAME=flight_booking
```

### 6. Run the app
```bash
chainlit run app.py -w
```

## 📁 Project Structure

```
.
├── app.py                      # Main Chainlit application
├── src/
│   ├── agents/
│   │   └── agents.py          # Agent definitions and prompts
│   ├── core/
│   │   ├── nodes.py           # Graph nodes and routing logic
│   │   └── state.py           # State management
│   ├── database/
│   │   └── db.py              # Vector store setup
│   ├── tools/
│   │   └── tools.py           # Tool functions for agents
│   └── utils/
│       └── prompt.py          # Memory extraction prompts
├── assets/
│   └── tourist_destination.pdf # Tourist information data
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── requirements.txt           # Python dependencies
```

## 🔑 Required API Keys

- **GEMINI_API_KEY**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **GROQ_API_KEY**: Get from [Groq Console](https://console.groq.com/)
- **OPENWEATHERMAP_API_KEY**: Get from [OpenWeatherMap](https://openweathermap.org/api)
- **TOGETHER_API_KEY**: (Optional) Get from [Together AI](https://together.ai/)

## 🎯 Usage

1. Start a conversation with the Agent
2. Ask about flights, hotels, tours, or shuttles
3. The Agent will search available options, process payment, and store your booking

