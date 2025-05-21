# ✈️ Multimodal Multi-Agent Travel Support Bot

A multimodal AI assistant that helps users **search and book flights**, and **reserve airport shuttle services**, powered by a multi-agent architecture built with [LangGraph](https://github.com/langchain-ai/langgraph).

## 💡 Key Features

- 🤖 **Multi-Agent System**: Uses a Supervisor Agent to coordinate:
  - **Flight Agent** – Searches and books flight tickets.
  - **Shuttle Agent** – Finds and reserves airport shuttle services.
- 🧠 **Multimodal Input**: Supports user input via **text**, **image**, or **audio**.
- 📦 **Real-Time Database Integration**:
  - Fetches available flights and shuttles based on user query.
  - Deducts user balance and stores booking data in **MongoDB** and **SQL**.
- 🔁 **Transactional Booking Flow**: Ensures atomicity between booking and payment.

## 📌 Architecture
![image](https://github.com/user-attachments/assets/c9a57cb8-b155-4c9a-a94f-0b1cb00f6170)



## 🛠️ Technologies Used

- **LangGraph** – Multi-agent coordination
- **LangChain** – Agent logic & tool calling
- **MongoDB & SQL** – Booking & user data
- **Python** – Core logic
- **Gemini / STT / Image Models** – For multimodal input (if used)

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/NgNMinh/Multimodal-Multi-Agent-Travel-Support-Bot-.git
   cd Multimodal-Multi-Agent-Travel-Support-Bot-

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Configure environment variables**
   ```bash
   GEMINI_API_KEY=your_key
   GROQ_API_KEY=your_key
   TOGETHER_API_KEY=your_key

4. **Run the app**
   ```bash
   chainlit run app.py -w
