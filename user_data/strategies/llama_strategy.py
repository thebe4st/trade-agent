import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union, Dict, List
import json
import requests
import logging
from polymarket import PolymarketAPI
from freqtrade.wallets import Wallets
import talib.abstract as ta

import threading

from freqtrade.strategy import (
    IStrategy,
    informative,
    merge_informative_pair,
)
from freqtrade.persistence import Order, PairLocks, Trade

from freqtrade.exchange import timeframe_to_minutes

logger = logging.getLogger(__name__)

from trade_gpt import TradeGPT

class LlamaGPTStrategy(IStrategy):
    """
    基于GPT接口的LLAMA交易策略
    该策略会配置提示词，并把k线、提示词、以及记忆一起输入给Llama
    Llama经过判断后会返回json结构，该json结构标识了建议买入or卖出的信号、止损止盈点
    memory会记录Llama的每一次操作，并记录盈亏
    """

    # 策略名称
    strategy_name = "LlamaGPTStrategy"
    # 时间周期设置为1小时
    timeframe = "15m"

    # 只处理新的蜡烛图
    process_only_new_candles = False

    # 做空策略必须设置
    can_short = True

    # 这些值可以在配置中被覆盖
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 最小ROI配置 - 可以被模型建议的止盈点覆盖
    minimal_roi = {}

    # 禁用止损
    stoploss = -0.9
    # 使用自定义止损
    use_custom_stoploss=False
    # 追踪止损
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    ## 以下配置的含义为：当我达到百分之5的利润时，回撤3%止损
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05  # 禁用/未配置

    # 运行"populate_indicators()"的窗口大小
    startup_candle_count = 30

    # 动态的指定持仓数量
    position_adjustment_enable = True

    # 初始化
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        # 初始化TradeGPT实例
        logger.info(f"初始化TradeGPT实例，模型：{config['gpt']['model']}，提示词文件：{config['gpt']['system_prompt_file']}")
        self.trade_gpt = TradeGPT(
            api_base= config['gpt']['base_url'],
            api_key= config['gpt']['api_key'],
            model= config['gpt']['model'],
            prompt_file_path= config['gpt']['system_prompt_file'],
            max_memory_length=30,
            memory_file_path= config['gpt']['memory_file']
        )
        # 初始化polymarket API
        self.polymarket_api = PolymarketAPI()
        self.account_info = {
            "balances": {},
            "positions": {}
        }
        self.decision = {}
        self._mock_signal = config['gpt']['mock_signal']
        logger.info(f"mock信号：{self._mock_signal}")
        self.last_force_call = datetime.now() - timedelta(days=365*100) # 保存最近一次触发的K线时间
        self.gpt_interval_minutes = config['gpt']['interval_minutes']  # 从配置中获取时间间隔
        self.gpt_force_interval_minutes = config['gpt']['force_interval_minutes']  # 从配置中获取时间间隔
        logger.info(f"大模型调用时间间隔：{self.gpt_interval_minutes}分钟")
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            获取账户信息
            生成交易决策
        """
        # 当前pair
        pair = metadata['pair']

        # 计算指标
        self.technical_indicators(dataframe,metadata)
        latest = dataframe.iloc[-1]
        latest_time = latest['date']

        if latest['llm_trigger']:  # 当行情不太对时，强行唤醒大模型，否则大模型将按照固定频率调用
            ## todo 是否要降级为10分钟调用一次？
            if datetime.now() - self.last_force_call < timedelta(minutes=self.gpt_force_interval_minutes):
                return dataframe
            logger.info(f"行情波动大模型被强制唤醒，触发时间：{latest_time}")
            self.last_force_call = datetime.now() # 记录最近一次大模型强制触发的触发时间
        else:
            # 查看当前descision的时间是否超过timeframe指定的时间，如果没超时则不需要再次调用
            if  pair in self.decision and self.decision[pair]['success'] and 'timestamp' in self.decision[pair]:
                last_decision_time = datetime.fromtimestamp(self.decision[pair]['timestamp'])
                if datetime.now() - last_decision_time < timedelta(minutes=self.gpt_interval_minutes):
                    return dataframe
                
        return self._call_gpt(dataframe, pair)
    
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,) -> bool:
         # 计算利润（以报价货币计价，比如 USDT）
        profit_abs = trade.calc_profit(rate)
        # 计算利润率（百分比）
        profit_ratio = trade.calc_profit_ratio(rate)
        result = {
            "exit_rate": rate,
            "profit_abs": profit_abs,
            "profit_ratio": profit_ratio
        }
        logger.info(f"确认交易退出 - 交易对: {pair}, 订单类型: {order_type}, 退出原因: {exit_reason}, 利润: {profit_abs:.2f}, 利润率: {profit_ratio:.2%}")
        ### 更新交易盈利状态到memory
        res = self.trade_gpt.update_trade_result(pair, result=result)
        logger.info(f'更新记忆：{res}')
        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于GPT模型的分析，生成入场信号
        """
        pair = metadata["pair"]
        decision = self.decision[pair]
        if not decision:
            logger.error(f"未获取到交易决策 - 交易对: {pair}")
            return dataframe

        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0
        if decision.get('done', True) or not decision.get('success', False):
            return dataframe
        
        # 解析决策并生成信号
        signal = decision.get("signal", "hold").lower()
        if decision.get("confidence", 0) <= 0.6:
            logger.error(f"GPT信心不足0.6,跳过入场信号～")
            decision['done'] = True
            return dataframe
            
        if signal == "long":
            # 生成买入信号
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1
            dataframe.loc[dataframe.index[-1], "enter_tag"] = json.dumps(decision)
            
            logger.info(f"买入信号 - 交易对: {pair}, 价格: {dataframe['close'].iloc[-1]}, 止损: {decision.get('stop_loss')}, 止盈: {decision.get('take_profit')}, 理由: {decision.get('reason')}")
        elif signal == "short":
            # 生成卖出信号（做空）
            dataframe.loc[dataframe.index[-1], "enter_short"] = 1
            dataframe.loc[dataframe.index[-1], "enter_tag"] = json.dumps(decision)
            
            logger.info(f"卖出信号 - 交易对: {pair}, 价格: {dataframe['close'].iloc[-1]}, 止损: {decision.get('stop_loss')}, 止盈: {decision.get('take_profit')}, 理由: {decision.get('reason')}")
        elif signal == "hold":
            logger.info(f"持有信号 - 交易对: {pair}, 理由: {decision.get('reason')}")
        decision['done'] = True
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        decision = self.decision[pair]
        if not decision:
            logger.error(f"未获取到交易决策 - 交易对: {pair}")
            return dataframe

        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        if not decision.get('done', False):
            return dataframe
        if not decision.get('success', False):
            return dataframe
        if decision.get("signal", "hold").lower() == "close":
            ## close强制退出，不需要处理
            return dataframe

        take_profit = decision.get("take_profit", 0)
        stop_loss = decision.get("stop_loss", 0)
        current_price = self._get_current_price(pair)

        if not take_profit or not stop_loss:
            return dataframe

        # 如果这里用signal判断，那么当gpt发出hold指令时就不能及时止盈
        if take_profit > stop_loss:
            ### 这是一个long单
            if current_price >= take_profit:
                dataframe.loc[dataframe.index[-1], "exit_long"] = 1
                dataframe.loc[dataframe.index[-1], "exit_tag"] = 'take_profit'
            elif current_price <= stop_loss:
                dataframe.loc[dataframe.index[-1], "exit_long"] = 1
                dataframe.loc[dataframe.index[-1], "exit_tag"] = 'stop_loss'
        else:
            ### 这是一个short单
            if current_price <= take_profit:
                dataframe.loc[dataframe.index[-1], "exit_short"] = 1
                dataframe.loc[dataframe.index[-1], "exit_tag"] = 'take_profit'
            elif current_price >= stop_loss:
                dataframe.loc[dataframe.index[-1], "exit_short"] = 1
                dataframe.loc[dataframe.index[-1], "exit_tag"] = 'stop_loss'
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Dict]:
        decision = self.decision[pair]
        if not decision:
            logger.error(f"未获取到交易决策 - 交易对: {pair}")
            return None
        if decision.get("close_done", False):
            # 已经执行过退场了
            return None
        if decision.get("signal", "hold").lower() == "close":
            decision['close_done'] = True
            ## 执行退场信号后马上重新计算指标，因为很可能趋势已经逆转，需要立刻反向持仓~
            return f"GPT 希望强制退场，原因为：{decision.get('reason')}"
        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        decision = self.decision[pair]
        if not decision:
            logger.error(f"未获取到交易决策 - 交易对: {pair}")
            return None
        if decision.get("stop_loss_done", False):
            # 已经设置过止损了
            return None
        # Use parabolic sar as absolute stoploss price
        stoploss_price = decision.get("stop_loss", 0)
        logger.info(f'调整止损点至: {stoploss_price}, 当前价格{current_rate}, is_short{trade.is_short}')
        # 按文档要求，返回相对当前价格的百分比
        stoploss_pct = (current_rate - stoploss_price) / current_rate

        # 不允许负数（否则意味着止损价比当前价还高，没意义）
        stoploss_pct = min(0, stoploss_pct)
        decision['stop_loss_done'] = True        
        return self._stoploss_from_absolute(stoploss_price, current_rate, is_short=trade.is_short)

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        decision = self.decision[trade.pair]
        if not decision:
            logger.error(f"未获取到交易决策 - 交易对: {trade.pair}")
            return None
        if decision.get('adjust_position_done', False):
            # 已经调整过持仓金额了
            return None
        
        signal = decision.get("signal", 'hold').lower()
        if signal == 'close':
            # 平仓不需要调整持仓
            return None
        
        adjust_position = decision.get('adjust_position', 0)
        if adjust_position != 0:
            if adjust_position >0:
                # 最大可加仓不能超过可用余额的百分之70
                available_balance = self.wallets.get_free('USDT')
                max_adjust_position = available_balance * 0.7
                adjust_position = min(adjust_position, max_adjust_position)
            
            # 增加调整持仓金额的日志
            logger.info(f'调整仓位至: {adjust_position}')
            decision['adjust_position_done'] = True
            return adjust_position
        return None
    
    def technical_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Momentum Indicators
        # ------------------------------------

        # ADX
        # dataframe["adx"] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe["fastd"] = stoch_fast["fastd"]
        # dataframe["fastk"] = stoch_fast["fastk"]

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # MFI
        # dataframe["mfi"] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) /
        #     dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

         # 布林带
        boll = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upper'] = boll[0]
        dataframe['bb_middle'] = boll[1]
        dataframe['bb_lower'] = boll[2]
         # 成交量均值
        dataframe['volume_mean'] = dataframe['volume'].rolling(20).mean()
  # === 触发条件 ===
        # 初始化触发信号
        dataframe['llm_trigger'] = False

        # 条件 1: RSI 超买/超卖
        cond_rsi = (dataframe['rsi'] > 70) | (dataframe['rsi'] < 30)

        # 条件 2: MACD 金叉 / 死叉
        cond_macd_cross = (
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
        ) | (
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
        )

        # 条件 3: 布林带突破
        cond_bb = (dataframe['close'] > dataframe['bb_upper']) | (dataframe['close'] < dataframe['bb_lower'])

        # 条件 4: 成交量异常放大
        cond_vol = dataframe['volume'] > dataframe['volume_mean'] * 1.5

        # 合并触发条件
        dataframe.loc[cond_rsi | cond_macd_cross | cond_bb | cond_vol, 'llm_trigger'] = True

        # Parabolic SAR
        # dataframe["sar"] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        # dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        # hilbert = ta.HT_SINE(dataframe)
        # dataframe["htsine"] = hilbert["sine"]
        # dataframe["htleadsine"] = hilbert["leadsine"]

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        杠杆倍数
        """
        return 1.0
    
    def _call_gpt(self, dataframe: DataFrame, pair: str) -> DataFrame: 
             # 获取当前账户余额和持仓情况
        try:
            # 获取账户余额
            balances = self.wallets.get_all_balances()
            self.account_info["balances"] = {}
            for currency, balance in balances.items():
                self.account_info["balances"][currency] = balance._asdict()
            # 获取当前持仓情况
            active_trades = self.wallets.get_all_positions()
            self.account_info["positions"] = {}
            for currency,trade in active_trades.items():
                self.account_info["positions"][currency] = trade._asdict()
            logging.info(f"当前账户信息: {json.dumps(self.account_info)}")
            
            mock = self._mock(self._mock_signal)

             # 最新的 ticker 数据
            last_price = self._get_current_price(pair)
            logger.info(f"实时价格 {pair}: {last_price}")
            current_month = datetime.now().strftime("%B").lower()  # "september"
            # 调用polymarket API获取比特币价格预测
            polymarket_pred = ""
            polymarket_pred = self.polymarket_api.search_price_markets(query=f"bitcoin price {current_month}")

            ## todo 使用锁保证同一时刻generate_trading_decision只有一个在调用，其他调用会丢弃
            decision = self.trade_gpt.generate_trading_decision(dataframe, pair, self.account_info, polymarket_pred,last_price=last_price, mock=mock)
            decision['timestamp'] = int(datetime.now().timestamp())
            decision['done'] = False
            self.decision[pair] = decision
        except Exception as e:
            logger.exception(f"计算指标时发生错误: {e}")
            return dataframe
        return dataframe

    def _stoploss_from_absolute(self, stoploss_price: float, current_rate: float, is_short: bool = False) -> float:
        """
        将绝对止损价格转换为 freqtrade custom_stoploss 所需的百分比。

        :param current_rate: 当前价格
        :param stoploss_price: 绝对止损价格
        :param is_short: 是否为空头仓位（默认 False 表示多头）
        :return: 相对当前价格的止损百分比 (>=0)，freqtrade 会自动在当前价下方/上方设置止损
        """
        if is_short:
            # 空头：止损在当前价格之上
            stoploss_pct = (stoploss_price - current_rate) / current_rate
        else:
            # 多头：止损在当前价格之下
            stoploss_pct = (current_rate - stoploss_price) / current_rate

        # 不允许返回负数，否则 freqtrade 会认为止损价比当前价更优，没意义
        return max(0.0, stoploss_pct)
    
    def _mock(self,signal: str):
        if signal:
            return {
                        "success": True,
                        "signal": signal,
                        "confidence": 0.7,
                        "reason": "No reason provided",
                        "stop_loss": 119100,
                        "take_profit": 107000,
                        "adjust_position": 0
                    }
        else:
            return None
        
    def _get_current_price(self, pair: str) -> float:
        """
        获取当前pair的实时价格
        """
        ticker = self.dp.ticker(pair)
        return ticker['last']