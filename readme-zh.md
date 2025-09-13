# Llama-Trade - AI驱动的加密货币交易策略

Llama-Trade是一个基于GPT的智能交易策略系统，它结合了先进的AI决策能力和Freqtrade交易框架，为加密货币交易提供智能化的决策支持。

希望本项目能够给您带来一些启发和帮助，同时也欢迎您参与到本项目的开发和改进中来。

## 项目特点

- **AI驱动决策**：利用GPT模型进行交易决策分析
- **记忆系统**：记录历史交易决策和结果，支持持久化存储
- **技术指标**：集成多种技术指标分析市场趋势
- **多空策略**：同时支持多头和空头交易
- **Docker部署**：提供简单的Docker部署方案

## 快速开始

### 前提条件

- Docker和Docker Compose
- Freqtrade环境
- 交易所API密钥（如OKX、Binance等）
- GPT模型API（如OpenAI的API）

### 安装与配置

1. **克隆仓库**

```bash
git clone https://github.com/thebe4st/llama-trade
cd llama-trade
```

2. **配置API密钥**

编辑`user_data/config-dev.json`文件，填写您的API密钥和其他配置信息：

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
  // 其他配置...
}
```

3. **配置提示词**

编辑`user_data/strategies/prompt/v1.txt`文件，根据您的需求自定义提示词。

### 使用Docker启动

```bash
./start.sh dev
```


## 记忆系统

策略具有记忆功能，可以记录历史交易决策和结果，并在程序重启后保留这些信息。记忆数据会自动保存到`user_data/strategies/memory.json`文件中。

## 风险提示

加密货币交易具有高风险，请谨慎使用本策略。建议在实盘交易前进行充分的回测和模拟交易。

## 开源许可证

本项目采用 [Apache License](LICENSE) 开源。

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目。在提交代码前，请确保通过了代码风格检查。

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用本项目产生的任何损失，作者不承担任何责任。
