# Llama-Trade - AI-driven cryptocurrency trading strategy 

[中文简体](readme-zh.md)

Llama-Trade is an intelligent trading strategy system based on GPT. It integrates advanced AI decision-making capabilities with the Freqtrade trading framework, providing intelligent decision-making support for cryptocurrency trading. 
We sincerely hope that this project can provide you with some inspiration and assistance. We also sincerely invite you to participate in the development and improvement of this project. 

## Project Features 

- **AI-driven decision-making**: Utilize the GPT model for analyzing trading decisions

- **Memory system**: Records historical trading decisions and results, supporting persistent storage

- **Technical indicators**: Integrates multiple technical indicators to analyze market trends

- **Long/short strategy**: Supports both long and short trading simultaneously

- **Docker deployment**: Provides a simple Docker deployment solution 

## Quick Start 

### Prerequisites 

- Docker and Docker Compose
- Freqtrade environment
- Exchange API keys (such as for OKX, Binance, etc.)
- GPT model API (such as the API of OpenAI) 

### Installation and Configuration 

1. **Clone Repository** 

```bash
git clone https://github.com/thebe4st/llama-trade
cd llama-trade
```

2. **Configure API Key** 
Edit the `user_data/config-dev.json` file and fill in your API key and other configuration information: 
```json
{
"exchange": {
"name": "okx",
"key": "YOUR_EXCHANGE_API_KEY",
"secret": "YOUR_EXCHANGE_API_SECRET",
"password": "YOUR_EXCHANGE_API_PASSWORD"
},
"gpt": {
"model": "YOUR_GPT_MODEL",
"api_key": "YOUR_GPT_API_KEY",
"base_url": "YOUR_GPT_API_BASE_URL",
"system_prompt_file": "user_data/strategies/prompt/v1.txt"
}
// Other configurations... }
```

3. **Configuration Prompt Words** 

Edit the file `user_data/strategies/prompt/v1.txt` and customize the prompt text according to your requirements. 
Use Docker to start 
```bash
./start.sh dev
```


## Memory System 

The strategy has a memory function, which can record historical trading decisions and results, and retain this information even after the program restarts. The memory data will be automatically saved to the file `user_data/strategies/memory.json`. 

## Risk Warning 

Cryptocurrency trading involves high risks. Please use this strategy with caution. It is recommended to conduct thorough backtesting and simulated trading before actual trading. 

## Open Source License 

This project is released under the [Apache License](LICENSE). 

## Contribution Guidelines 

Welcome to submit Issues and Pull Requests to improve this project. Before submitting the code, please make sure it has passed the code style check. 

## Disclaimer 

This project is solely for learning and research purposes and does not constitute any investment advice. The author assumes no responsibility for any losses incurred from using this project.