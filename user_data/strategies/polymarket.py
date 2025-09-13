import requests
import json
from datetime import datetime

class PolymarketAPI:
    def __init__(self, proxy=None):
        self.base_url = "https://gamma-api.polymarket.com"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # 配置代理
        self.proxies = None
        if proxy:
            self.proxies = {
                "http": proxy,
                "https": proxy
            }
    
    def search_price_markets(self, query=None, cache=True, events_status="active", sort="volume", closed=True, limit=2):
        """
        搜索与加密货币价格相关的市场
        :param query: 搜索关键词，如"bitcoin price"或"eth price"
        :param cache: 是否使用缓存
        :param events_status: 事件状态，默认为"active"
        :param sort: 排序方式，默认为"volume"
        :return: 处理后的市场数据列表
        """
        # 设置查询参数
        params = {
            "cache": str(cache).lower(),
            "events_status": events_status,
            "sort": sort
        }
        
        # 如果提供了查询关键词，添加到参数中
        if query:
            params["q"] = query
        
        url = f"{self.base_url}/public-search"
        try:
            # 发送请求
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                proxies=self.proxies,
                timeout=30
            )
            # 检查响应状态
            if not response.ok:
                print(f"API请求失败，状态码: {response.status_code}")
                return None
            
            # 尝试解析JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("JSON解析失败")
                return None
            
            
            # 处理数据，只保留与价格、市场有关的信息
            processed_events = []
            events = data.get("events", [])
            if len(events) > limit:
                events = events[:limit]
            for event in events:
                # 提取事件级别的关键信息
                processed_event = {
                    "id": event.get("id"),
                    "title": event.get("title", "无标题"),
                    "description": event.get("description", "无描述"),
                    "startDate": event.get("startDate"),
                    "endDate": event.get("endDate"),
                    "liquidity": self._safe_float(event.get("liquidity")),
                    "volume": self._safe_float(event.get("volume")),
                    "volume24hr": self._safe_float(event.get("volume24hr")),
                    "markets": []
                }
                
                # 处理市场信息
                for market in event.get("markets", []):
                    if closed and market.get("closed", True):
                        # 只关注还没closed的
                        continue
                    
                    # 提取市场级别的关键价格信息
                    processed_market = {
                        "id": market.get("id"),
                        "question": market.get("question", "无问题描述"),
                        "liquidity": self._safe_float(market.get("liquidityNum")),
                        "volume": self._safe_float(market.get("volumeNum")),
                        "volume24hr": self._safe_float(market.get("volume24hr")),
                        "lastTradePrice": self._safe_float(market.get("lastTradePrice")),
                        "bestBid": self._safe_float(market.get("bestBid")),
                        "bestAsk": self._safe_float(market.get("bestAsk")),
                        "spread": self._safe_float(market.get("spread")),
                        "oneWeekPriceChange": self._safe_float(market.get("oneWeekPriceChange")),
                        "endDate": market.get("endDate")
                    }
                    
                    # 如果有结果价格，解析它们
                    if "outcomePrices" in market and market["outcomePrices"]:
                        try:
                            outcome_prices = json.loads(market["outcomePrices"])
                            if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                                processed_market["yesPrice"] = self._safe_float(outcome_prices[0])
                                processed_market["noPrice"] = self._safe_float(outcome_prices[1])
                        except (json.JSONDecodeError, ValueError, TypeError):
                            # 静默忽略解析错误
                            pass
                    
                    processed_event["markets"].append(processed_market)
                
                processed_events.append(processed_event)
            
            return processed_events
            
        except requests.exceptions.RequestException as e:
            print(f"获取加密货币市场失败: {e}")
            return None
    
    def _safe_float(self, value, default=0.0):
        """
        安全地将值转换为浮点数
        :param value: 要转换的值
        :param default: 转换失败时的默认值
        :return: 转换后的浮点数或默认值
        """
        if value is None:
            return default
        try:
            # 先检查是否为字符串类型且可能包含引号
            if isinstance(value, str):
                # 移除可能的引号
                value = value.strip('"\'')
            return float(value)
        except (ValueError, TypeError):
            return default

def print_market_info(markets):
    if markets:
        print(f"找到 {len(markets)} 个相关事件/市场组:\n")
        for i, event in enumerate(markets, 1):
            print(f"{i}. 事件标题: {event['title']}")
            print(f"   总流动性: {event['liquidity']:.2f} USDC")
            print(f"   总交易量: {event['volume']:.2f} USDC")
            print(f"   24小时交易量: {event['volume24hr']:.2f} USDC")
            print(f"   包含 {len(event['markets'])} 个子市场:")
            
            for market in event['markets']:
                print(f"     - 市场: {market['question']}")
                print(f"       流动性: {market.get('liquidity', 0):.2f} USDC")
                print(f"       价格信息: ")
                if 'yesPrice' in market and market['yesPrice'] is not None:
                    print(f"         YES: {market['yesPrice']:.4f} ({market['yesPrice']*100:.1f}%)")
                    if 'noPrice' in market and market['noPrice'] is not None:
                        print(f"         NO: {market['noPrice']:.4f} ({market['noPrice']*100:.1f}%)")
                if 'lastTradePrice' in market and market['lastTradePrice'] is not None:
                    print(f"         最后成交价: {market['lastTradePrice']:.4f}")
                if 'bestBid' in market and 'bestAsk' in market and market['bestBid'] is not None and market['bestAsk'] is not None:
                    print(f"         买卖价差: {market['bestBid']:.4f} - {market['bestAsk']:.4f}")
            print("   ---")
    else:
        print("未找到比特币价格相关市场")

# 使用示例
if __name__ == "__main__":
    PROXY = None  # 暂时不使用代理
    polymarket = PolymarketAPI(proxy=PROXY)
    btc_markets = polymarket.search_price_markets(query="bitcoin price")
    print(json.dumps(btc_markets, indent=2))
    