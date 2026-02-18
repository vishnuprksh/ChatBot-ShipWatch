# ChatBot-ShipWatch

A sophisticated chatbot application for maritime noon data entry with intelligent contradiction detection and resolution using Google's Gemini AI.

## Features

- **Intelligent Data Entry**: Enter noon reports for vessels with Laden/Ballast status and report types.
- **Contradiction Detection**: Automatically detects potential inconsistencies in vessel status based on historical data.
- **AI-Powered Chat Resolution**: Uses Gemini AI to engage in natural conversation to resolve detected contradictions.
- **Dual Interface**: Available as both a Streamlit web app and a FastAPI backend with HTML frontend.
- **Data Persistence**: Stores noon data in memory (can be extended to database storage).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChatBot-ShipWatch.git
   cd ChatBot-ShipWatch
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Google Gemini API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

### Streamlit App (Main Interface)

Run the Streamlit application:
```bash
streamlit run app.py
```

This will start the web interface for entering noon data and interacting with the chatbot.

### FastAPI Backend

Run the FastAPI server:
```bash
uvicorn WebApp.main:app --reload
```

Then open `http://localhost:8000` in your browser for the HTML interface.

## Project Structure

```
ChatBot-ShipWatch/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── WebApp/               # FastAPI backend
    ├── main.py           # FastAPI application
    ├── models.py         # Pydantic models
    ├── storage.py        # Data storage logic
    ├── logic.py          # Business logic for contradiction checks
    ├── gemini_api.py     # Gemini AI integration
    ├── static/           # Static files (CSS, JS)
    └── templates/        # HTML templates
```

## API Endpoints

- `GET /`: Serve the main HTML page
- `POST /add_entry`: Add a new noon entry
- `POST /check_contradiction`: Check for contradictions in new data
- `POST /chat`: Interact with the AI chatbot for contradiction resolution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details