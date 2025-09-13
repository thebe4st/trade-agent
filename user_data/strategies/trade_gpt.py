# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import json
import requests
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Union
import pandas as pd
from pandas import DataFrame
import uuid

import os 

logger = logging.getLogger(__name__)

class TradeGPT:
    """
    封装GPT API调用、提示词管理和记忆系统的类
    用于交易策略中获取AI决策支持
    """
    
    def __init__(self, 
                 api_base: str = "",
                 api_key: str = "",
                 model: str = "",
                 prompt_file_path: str = '',
                 max_memory_length: int = 10,
                 memory_file_path: str = 'user_data/strategies/memory.json'):
        """
        初始化TradeGPT实例
        
        :param api_base: GPT API基础URL
        :param api_key: GPT API密钥
        :param model: 使用的模型名称
        :param prompt_file_path: 系统提示词文件路径
        :param max_memory_length: 最大记忆长度
        :param memory_file_path: 记忆文件保存路径
        """
        # GPT API配置
        self.gpt_api_base = api_base
        self.gpt_api_key = api_key
        self.gpt_model = model
        
        # 记忆系统
        self.memory = []
        self.max_memory_length = max_memory_length
        self.memory_file_path = memory_file_path
        
        # 加载系统提示词
        self.system_prompt = ''
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        except Exception as e:
            logger.error(f"读取系统提示词文件失败: {str(e)}")
        
        # 加载之前保存的记忆
        self.load_memory_from_file()
    
    def load_memory_from_file(self) -> None:
        """
        从文件中加载记忆
        """
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    loaded_memory = json.load(f)
                    # 确保加载的是列表类型
                    if isinstance(loaded_memory, list):
                        self.memory = loaded_memory
                        # 限制记忆长度
                        if len(self.memory) > self.max_memory_length:
                            self.memory = self.memory[-self.max_memory_length:]
                        logger.info(f"成功从文件加载记忆，共 {len(self.memory)} 条记录")
                    else:
                        logger.error("加载的记忆数据不是有效的列表格式")
            else:
                logger.info(f"记忆文件不存在，将创建新文件: {self.memory_file_path}")
        except Exception as e:
            logger.error(f"加载记忆文件失败: {str(e)}")
    
    def save_memory_to_file(self) -> bool:
        """
        将记忆保存到文件
        
        :return: 是否成功保存
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.memory_file_path)), exist_ok=True)
            
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            logger.info(f"成功将记忆保存到文件: {self.memory_file_path}")
            return True
        except Exception as e:
            logger.error(f"保存记忆文件失败: {str(e)}")
            return False
    
    def _prepare_kline_data(self, dataframe: DataFrame) -> str:
        """
        准备K线数据用于发送给GPT模型
        
        :param dataframe: 包含K线数据的DataFrame
        :return: 格式化后的K线数据JSON字符串
        """
        # 获取最近的20根K线
        recent_klines = dataframe.tail(30).copy()
        
        # 转换时间格式
        if 'date' in recent_klines.columns:
            recent_klines['date'] = recent_klines['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 转换为字典列表，便于JSON序列化
        kline_dict = recent_klines.to_dict('records')
        
        # 转换为JSON字符串
        return json.dumps(kline_dict, ensure_ascii=False)
    
    def _prepare_memory(self) -> str:
        """
        准备记忆数据用于发送给GPT模型
        
        :return: 格式化后的记忆数据JSON字符串
        """
        return json.dumps(self.memory, ensure_ascii=False, default=str)
    
    def _prepare_account_info(self, account_info: Dict) -> str:
        """
        准备账户信息用于发送给GPT模型
        
        :return: 格式化后的账户信息JSON字符串
        """
        return json.dumps(account_info, ensure_ascii=False)
    
    def _call_gpt_api(self, prompt: str, mock: Dict = None) -> Dict:
        """
        调用GPT API获取交易建议
        
        :param prompt: 用户提示词
        :return: GPT返回的决策结果
        """
            # 构建API请求数据
        messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
        logger.info(f"调用GPT API的请求数据: {json.dumps(messages)}")
            
        if mock:
            return mock
            
        data = {
                "model": self.gpt_model,
                "messages": messages
            }
            
            
        try:
            # 发送请求
            response = requests.post(
                self.gpt_api_base,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.gpt_api_key}"},
                json=data,
                timeout=300
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                logger.info(f"GPT API返回的原始响应: {json.dumps(result)}")
                # 从响应中提取内容
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 尝试解析JSON响应
                try:
                    decision = json.loads(content)
                    decision["success"] = True
                    logger.info(f"GPT返回的有效JSON响应: {json.dumps(decision)}")
                    return decision
                except json.JSONDecodeError:
                    logger.error(f"GPT返回的内容不是有效的JSON: {content}")
                    return {"signal": "hold", "reason": "无法解析GPT响应", "success": False}
            else:
                logger.error(f"GPT API请求失败: {response.status_code}, {response.text}")
                return {"signal": "hold", "reason": "GPT API请求失败", "success": False}
        except Exception as e:
            logger.error(f"调用GPT API时发生错误: {str(e)}")
            return {"signal": "hold", "reason": f"API调用异常: {str(e)}", "success": False}
    
    def _update_memory(self, pair: str, decision: Dict, price: float, result: Optional[str] = None) -> None:
        """
        更新策略记忆
        
        :param pair: 交易对
        :param decision: GPT决策结果
        :param price: 当前价格
        :param result: 交易结果（可选，'profit'或'loss'）
        """
        memory_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "decision": decision,
            "price": price,
            "result": result  # 后续可以更新为"profit"或"loss"
        }
        
        # 添加新记忆
        self.memory.append(memory_entry)
        
        # 限制记忆长度
        if len(self.memory) > self.max_memory_length:
            self.memory = self.memory[-self.max_memory_length:]
        
        # 保存记忆到文件
        self.save_memory_to_file()
    
    def generate_trading_decision(self, 
                                 dataframe: DataFrame, 
                                 pair: str, 
                                 account_info: Dict,
                                 polymarket_prediction: str = "",
                                 timeframe: str = "15m",
                                 last_price: float = 0.0,
                                 mock: Dict = None) -> Dict:
        """
        生成交易决策
        
        :param dataframe: K线数据
        :param pair: 交易对
        :param polymarket_prediction: Polymarket预测信息（可选）
        :return: 交易决策结果
        """
        # 准备数据
        kline_data = self._prepare_kline_data(dataframe)
        memory_data = self._prepare_memory()
        current_price = last_price
        account_info_json = self._prepare_account_info(account_info)
        # 构建用户提示词
        user_prompt = f"""
        ## Input Data
        - Past trades: {memory_data}
        - Polymarket prediction: {polymarket_prediction}
        - K-line({timeframe}) and technical indicators :  {kline_data}
        - Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        - Holdings: {json.dumps(account_info['positions'])}
        - Wallet balance: {json.dumps(account_info['balances'])}
        - current price(usdt): **{current_price}**
        - pair: {pair}
        """
        
        # 调用GPT API获取决策
        decision = self._call_gpt_api(user_prompt, mock)
        if not decision.get("success", False):
            return decision
        decision['timestamp'] = int(datetime.now().timestamp())
        decision['uuid'] = str(uuid.uuid4())
        # 记录决策到记忆
        self._update_memory(pair, decision, current_price)
        return decision
    
    def update_trade_result(self, pair: str, result: Dict) -> bool:
        """
        更新交易结果到记忆中
        
        :param pair: 交易对
        :param result: 交易结果（'profit'或'loss'）
        :return: 是否成功更新
        """
        for i, entry in enumerate(reversed(self.memory)):
            if entry["pair"] == pair and entry["result"] is None:
                entry_index = len(self.memory) - 1 - i
                self.memory[entry_index]["result"] = result
                logger.info(f"更新交易结果: {self.memory[entry_index]['uuid']}-{json.dumps(result)}")
                # 保存更新后的记忆到文件
                self.save_memory_to_file()
                return True
        return False
