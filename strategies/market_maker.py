"""
做市策略模塊
"""
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from api.client import (
    get_balance, execute_order, get_open_orders, cancel_all_orders, 
    cancel_order, get_market_limits, get_ticker, get_order_book, get_klines
)
from ws_client.client import BackpackWebSocket
from database.db import Database
from utils.helpers import round_to_precision, round_to_tick_size, calculate_volatility
from logger import setup_logger

logger = setup_logger("market_maker")

class MarketMaker:
    def __init__(
        self, 
        api_key, 
        secret_key, 
        symbol, 
        db_instance=None,
        base_spread_percentage=0.24,  # 提高基础价差到0.24%以确保盈利
        order_quantity=None, 
        max_orders=5,
        rebalance_threshold=10.0,
        volatility_threshold=0.02,     # 波动率阈值
        min_profit_multiplier=1.5,     # 最小利润倍数
        max_position_size=50.0,        # 最大持仓规模(占总资产百分比)
        aggressive_factor=1.2,         # 市场深度良好时的进取因子
        ws_proxy=None
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_spread_percentage = base_spread_percentage
        self.order_quantity = order_quantity
        self.max_orders = max_orders
        self.rebalance_threshold = rebalance_threshold
        self.volatility_threshold = volatility_threshold
        self.min_profit_multiplier = min_profit_multiplier
        self.max_position_size = max_position_size
        self.aggressive_factor = aggressive_factor

        # 初始化數據庫
        self.db = db_instance if db_instance else Database()
        
        # 統計屬性
        self.session_start_time = datetime.now()
        self.session_buy_trades = []
        self.session_sell_trades = []
        self.session_fees = 0.0
        self.session_maker_buy_volume = 0.0
        self.session_maker_sell_volume = 0.0
        self.session_taker_buy_volume = 0.0
        self.session_taker_sell_volume = 0.0
        
        # 初始化市場限制
        self.market_limits = get_market_limits(symbol)
        if not self.market_limits:
            raise ValueError(f"無法獲取 {symbol} 的市場限制")
        
        self.base_asset = self.market_limits['base_asset']
        self.quote_asset = self.market_limits['quote_asset']
        self.base_precision = self.market_limits['base_precision']
        self.quote_precision = self.market_limits['quote_precision']
        self.min_order_size = float(self.market_limits['min_order_size'])
        self.tick_size = float(self.market_limits['tick_size'])
        
        # 交易量統計
        self.maker_buy_volume = 0
        self.maker_sell_volume = 0
        self.taker_buy_volume = 0
        self.taker_sell_volume = 0
        self.total_fees = 0

        # 市场分析相关属性
        self.market_state = {
            'volatility': 0,           # 当前波动率
            'trend': 0,                # 市场趋势（1=上升，-1=下降，0=横盘）
            'depth_score': 0,          # 市场深度评分（0-1）
            'volume_level': 0,         # 成交量水平（0-1）
            'spread_level': 0,         # 市场价差水平（0-1）
        }
        
        # 参数调整范围
        self.param_ranges = {
            'base_spread': {'min': 0.24, 'max': 0.5},
            'orders': {'min': 3, 'max': 8},
            'profit_multiplier': {'min': 1.3, 'max': 2.0},
            'position_size': {'min': 30.0, 'max': 70.0},
            'aggressive': {'min': 1.1, 'max': 1.4}
        }
        
        # 市场分析定时器
        self.last_analysis_time = time.time()
        self.analysis_interval = 300  # 5分钟分析一次

        # 添加代理参数
        self.ws_proxy = ws_proxy
        # 建立WebSocket連接
        self.ws = BackpackWebSocket(api_key, secret_key, symbol, self.on_ws_message, auto_reconnect=True, proxy=self.ws_proxy)
        self.ws.connect()
        
        # 跟蹤活躍訂單
        self.active_buy_orders = []
        self.active_sell_orders = []
        
        # 記錄買賣數量以便重新平衡
        self.total_bought = 0
        self.total_sold = 0
        
        # 交易記錄 - 用於計算利潤
        self.buy_trades = []
        self.sell_trades = []
        
        # 利潤統計
        self.total_profit = 0
        self.trades_executed = 0
        self.orders_placed = 0
        self.orders_cancelled = 0
        
        # 執行緒池用於後台任務
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 等待WebSocket連接建立並進行初始化訂閲
        self._initialize_websocket()
        
        # 載入交易統計和歷史交易
        self._load_trading_stats()
        self._load_recent_trades()
        
        logger.info(f"初始化做市商: {symbol}")
        logger.info(f"基礎資產: {self.base_asset}, 報價資產: {self.quote_asset}")
        logger.info(f"基礎精度: {self.base_precision}, 報價精度: {self.quote_precision}")
        logger.info(f"最小訂單大小: {self.min_order_size}, 價格步長: {self.tick_size}")
        logger.info(f"基礎價差百分比: {self.base_spread_percentage}%, 最大訂單數: {self.max_orders}")
    
    def _initialize_websocket(self):
        """等待WebSocket連接建立並進行初始化訂閲"""
        wait_time = 0
        max_wait_time = 10
        while not self.ws.connected and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if self.ws.connected:
            logger.info("WebSocket連接已建立，初始化數據流...")
            
            # 初始化訂單簿
            orderbook_initialized = self.ws.initialize_orderbook()
            
            # 訂閲深度流和行情數據
            if orderbook_initialized:
                depth_subscribed = self.ws.subscribe_depth()
                ticker_subscribed = self.ws.subscribe_bookTicker()
                
                if depth_subscribed and ticker_subscribed:
                    logger.info("數據流訂閲成功!")
            
            # 訂閲私有訂單更新流
            self.subscribe_order_updates()
        else:
            logger.warning("WebSocket連接建立超時，將在運行過程中繼續嘗試連接")
    
    def _load_trading_stats(self):
        """從數據庫加載交易統計數據"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 查詢今天的統計數據
            stats = self.db.get_trading_stats(self.symbol, today)
            
            if stats and len(stats) > 0:
                stat = stats[0]
                self.maker_buy_volume = stat['maker_buy_volume']
                self.maker_sell_volume = stat['maker_sell_volume']
                self.taker_buy_volume = stat['taker_buy_volume']
                self.taker_sell_volume = stat['taker_sell_volume']
                self.total_profit = stat['realized_profit']
                self.total_fees = stat['total_fees']
                
                logger.info("已從數據庫加載今日交易統計")
                logger.info(f"Maker買入量: {self.maker_buy_volume}, Maker賣出量: {self.maker_sell_volume}")
                logger.info(f"Taker買入量: {self.taker_buy_volume}, Taker賣出量: {self.taker_sell_volume}")
                logger.info(f"已實現利潤: {self.total_profit}, 總手續費: {self.total_fees}")
            else:
                logger.info("今日無交易統計記錄，將創建新記錄")
        except Exception as e:
            logger.error(f"加載交易統計時出錯: {e}")
    
    def _load_recent_trades(self):
        """從數據庫加載歷史成交記錄"""
        try:
            # 獲取訂單歷史
            trades = self.db.get_order_history(self.symbol, 1000)
            trades_count = len(trades) if trades else 0
            
            if trades_count > 0:
                for side, quantity, price, maker, fee in trades:
                    quantity = float(quantity)
                    price = float(price)
                    fee = float(fee)
                    
                    if side == 'Bid':  # 買入
                        self.buy_trades.append((price, quantity))
                        self.total_bought += quantity
                        if maker:
                            self.maker_buy_volume += quantity
                        else:
                            self.taker_buy_volume += quantity
                    elif side == 'Ask':  # 賣出
                        self.sell_trades.append((price, quantity))
                        self.total_sold += quantity
                        if maker:
                            self.maker_sell_volume += quantity
                        else:
                            self.taker_sell_volume += quantity
                    
                    self.total_fees += fee
                
                logger.info(f"已從數據庫載入 {trades_count} 條歷史成交記錄")
                logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
                logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
                logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")
                
                # 計算精確利潤
                self.total_profit = self._calculate_db_profit()
                logger.info(f"計算得出已實現利潤: {self.total_profit:.8f} {self.quote_asset}")
                logger.info(f"總手續費: {self.total_fees:.8f} {self.quote_asset}")
            else:
                logger.info("數據庫中沒有歷史成交記錄，嘗試從API獲取")
                self._load_trades_from_api()
                
        except Exception as e:
            logger.error(f"載入歷史成交記錄時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_trades_from_api(self):
        """從API加載歷史成交記錄"""
        from api.client import get_fill_history
        
        fill_history = get_fill_history(self.api_key, self.secret_key, self.symbol, 100)
        
        if isinstance(fill_history, dict) and "error" in fill_history:
            logger.error(f"載入成交記錄失敗: {fill_history['error']}")
            return
            
        if not fill_history:
            logger.info("沒有找到歷史成交記錄")
            return
        
        # 批量插入準備
        for fill in fill_history:
            price = float(fill.get('price', 0))
            quantity = float(fill.get('quantity', 0))
            side = fill.get('side')
            maker = fill.get('maker', False)
            fee = float(fill.get('fee', 0))
            fee_asset = fill.get('feeAsset', '')
            order_id = fill.get('orderId', '')
            
            # 準備訂單數據
            order_data = {
                'order_id': order_id,
                'symbol': self.symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'maker': maker,
                'fee': fee,
                'fee_asset': fee_asset,
                'trade_type': 'manual'
            }
            
            # 插入數據庫
            self.db.insert_order(order_data)
            
            if side == 'Bid':  # 買入
                self.buy_trades.append((price, quantity))
                self.total_bought += quantity
                if maker:
                    self.maker_buy_volume += quantity
                else:
                    self.taker_buy_volume += quantity
            elif side == 'Ask':  # 賣出
                self.sell_trades.append((price, quantity))
                self.total_sold += quantity
                if maker:
                    self.maker_sell_volume += quantity
                else:
                    self.taker_sell_volume += quantity
            
            self.total_fees += fee
        
        if fill_history:
            logger.info(f"已從API載入並存儲 {len(fill_history)} 條歷史成交記錄")
            
            # 更新總計
            logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
            logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
            logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")
            
            # 計算精確利潤
            self.total_profit = self._calculate_db_profit()
            logger.info(f"計算得出已實現利潤: {self.total_profit:.8f} {self.quote_asset}")
            logger.info(f"總手續費: {self.total_fees:.8f} {self.quote_asset}")
    
    def check_ws_connection(self):
        """檢查並恢復WebSocket連接"""
        ws_connected = self.ws and self.ws.is_connected()
        
        if not ws_connected:
            logger.warning("WebSocket連接已斷開或不可用，嘗試重新連接...")
            
            # 嘗試關閉現有連接
            if self.ws:
                try:
                    if hasattr(self.ws, 'running') and self.ws.running:
                        self.ws.running = False
                    if hasattr(self.ws, 'ws') and self.ws.ws:
                        try:
                            self.ws.ws.close()
                        except Exception:
                            pass
                    self.ws.close()
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"關閉現有WebSocket時出錯: {e}")
            
            # 創建新的連接
            try:
                logger.info("創建新的WebSocket連接...")
                self.ws = BackpackWebSocket(
                    self.api_key, 
                    self.secret_key, 
                    self.symbol, 
                    self.on_ws_message, 
                    auto_reconnect=True,
                    proxy=self.ws_proxy
                )
                self.ws.connect()
                
                # 等待連接建立
                wait_time = 0
                max_wait_time = 5
                while not self.ws.is_connected() and wait_time < max_wait_time:
                    time.sleep(0.5)
                    wait_time += 0.5
                    
                if self.ws.is_connected():
                    logger.info("WebSocket重新連接成功")
                    
                    # 重新初始化
                    self.ws.initialize_orderbook()
                    self.ws.subscribe_depth()
                    self.ws.subscribe_bookTicker()
                    self.subscribe_order_updates()
                else:
                    logger.warning("WebSocket重新連接嘗試中，將在下次迭代再次檢查")
                    
            except Exception as e:
                logger.error(f"創建新WebSocket連接時出錯: {e}")
                return False
        
        return self.ws and self.ws.is_connected()
    
    def on_ws_message(self, stream, data):
        """處理WebSocket消息回調"""
        if stream.startswith("account.orderUpdate."):
            event_type = data.get('e')
            
            # 「訂單成交」事件
            if event_type == 'orderFill':
                try:
                    side = data.get('S')
                    quantity = float(data.get('l', '0'))  # 此次成交數量
                    price = float(data.get('L', '0'))     # 此次成交價格
                    order_id = data.get('i')             # 訂單 ID
                    maker = data.get('m', False)         # 是否是 Maker
                    fee = float(data.get('n', '0'))      # 手續費
                    fee_asset = data.get('N', '')        # 手續費資產

                    logger.info(f"訂單成交: ID={order_id}, 方向={side}, 數量={quantity}, 價格={price}, Maker={maker}, 手續費={fee:.8f}")
                    
                    # 判斷交易類型
                    trade_type = 'market_making'  # 默認為做市行為
                    
                    # 安全地檢查訂單是否是重平衡訂單
                    try:
                        is_rebalance = self.db.is_rebalance_order(order_id, self.symbol)
                        if is_rebalance:
                            trade_type = 'rebalance'
                    except Exception as db_err:
                        logger.error(f"檢查重平衡訂單時出錯: {db_err}")
                    
                    # 準備訂單數據
                    order_data = {
                        'order_id': order_id,
                        'symbol': self.symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'maker': maker,
                        'fee': fee,
                        'fee_asset': fee_asset,
                        'trade_type': trade_type
                    }
                    
                    # 安全地插入數據庫
                    def safe_insert_order():
                        try:
                            self.db.insert_order(order_data)
                        except Exception as db_err:
                            logger.error(f"插入訂單數據時出錯: {db_err}")
                    
                    # 直接在當前線程中插入訂單數據，確保先寫入基本數據
                    safe_insert_order()
                    
                    # 更新買賣量和做市商成交量統計
                    if side == 'Bid':  # 買入
                        self.total_bought += quantity
                        self.buy_trades.append((price, quantity))
                        logger.info(f"買入成交: {quantity} {self.base_asset} @ {price} {self.quote_asset}")
                        
                        # 更新做市商成交量
                        if maker:
                            self.maker_buy_volume += quantity
                            self.session_maker_buy_volume += quantity
                        else:
                            self.taker_buy_volume += quantity
                            self.session_taker_buy_volume += quantity
                        
                        self.session_buy_trades.append((price, quantity))
                            
                    elif side == 'Ask':  # 賣出
                        self.total_sold += quantity
                        self.sell_trades.append((price, quantity))
                        logger.info(f"賣出成交: {quantity} {self.base_asset} @ {price} {self.quote_asset}")
                        
                        # 更新做市商成交量
                        if maker:
                            self.maker_sell_volume += quantity
                            self.session_maker_sell_volume += quantity
                        else:
                            self.taker_sell_volume += quantity
                            self.session_taker_sell_volume += quantity
                            
                        self.session_sell_trades.append((price, quantity))
                    
                    # 更新累計手續費
                    self.total_fees += fee
                    self.session_fees += fee
                        
                    # 在單獨的線程中更新統計數據，避免阻塞主回調
                    def safe_update_stats_wrapper():
                        try:
                            self._update_trading_stats()
                        except Exception as e:
                            logger.error(f"更新交易統計時出錯: {e}")
                    
                    self.executor.submit(safe_update_stats_wrapper)
                    
                    # 重新計算利潤（基於數據庫記錄）
                    # 也在單獨的線程中進行計算，避免阻塞
                    def update_profit():
                        try:
                            profit = self._calculate_db_profit()
                            self.total_profit = profit
                        except Exception as e:
                            logger.error(f"更新利潤計算時出錯: {e}")
                    
                    self.executor.submit(update_profit)
                    
                    # 計算本次執行的簡單利潤（不涉及數據庫查詢）
                    session_profit = self._calculate_session_profit()
                    
                    # 執行簡要統計
                    logger.info(f"累計利潤: {self.total_profit:.8f} {self.quote_asset}")
                    logger.info(f"本次執行利潤: {session_profit:.8f} {self.quote_asset}")
                    logger.info(f"本次執行手續費: {self.session_fees:.8f} {self.quote_asset}")
                    logger.info(f"本次執行淨利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")
                    
                    self.trades_executed += 1
                    logger.info(f"總買入: {self.total_bought} {self.base_asset}, 總賣出: {self.total_sold} {self.base_asset}")
                    logger.info(f"Maker買入: {self.maker_buy_volume} {self.base_asset}, Maker賣出: {self.maker_sell_volume} {self.base_asset}")
                    logger.info(f"Taker買入: {self.taker_buy_volume} {self.base_asset}, Taker賣出: {self.taker_sell_volume} {self.base_asset}")
                    
                except Exception as e:
                    logger.error(f"處理訂單成交消息時出錯: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _calculate_db_profit(self):
        """基於數據庫記錄計算已實現利潤（FIFO方法）"""
        try:
            # 獲取訂單歷史，注意這裡將返回一個列表
            order_history = self.db.get_order_history(self.symbol)
            if not order_history:
                return 0
            
            buy_trades = []
            sell_trades = []
            for side, quantity, price, maker, fee in order_history:
                if side == 'Bid':
                    buy_trades.append((float(price), float(quantity), float(fee)))
                elif side == 'Ask':
                    sell_trades.append((float(price), float(quantity), float(fee)))

            if not buy_trades or not sell_trades:
                return 0

            buy_queue = buy_trades.copy()
            total_profit = 0
            total_fees = 0

            for sell_price, sell_quantity, sell_fee in sell_trades:
                remaining_sell = sell_quantity
                total_fees += sell_fee

                while remaining_sell > 0 and buy_queue:
                    buy_price, buy_quantity, buy_fee = buy_queue[0]
                    matched_quantity = min(remaining_sell, buy_quantity)

                    trade_profit = (sell_price - buy_price) * matched_quantity
                    allocated_buy_fee = buy_fee * (matched_quantity / buy_quantity)
                    total_fees += allocated_buy_fee

                    net_trade_profit = trade_profit
                    total_profit += net_trade_profit

                    remaining_sell -= matched_quantity
                    if matched_quantity >= buy_quantity:
                        buy_queue.pop(0)
                    else:
                        remaining_fee = buy_fee * (1 - matched_quantity / buy_quantity)
                        buy_queue[0] = (buy_price, buy_quantity - matched_quantity, remaining_fee)

            self.total_fees = total_fees
            return total_profit

        except Exception as e:
            logger.error(f"計算數據庫利潤時出錯: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _update_trading_stats(self):
        """更新每日交易統計數據"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 計算額外指標
            volatility = 0
            if self.ws and hasattr(self.ws, 'historical_prices'):
                volatility = calculate_volatility(self.ws.historical_prices)
            
            # 計算平均價差
            avg_spread = 0
            if self.ws and self.ws.bid_price and self.ws.ask_price:
                avg_spread = (self.ws.ask_price - self.ws.bid_price) / ((self.ws.ask_price + self.ws.bid_price) / 2) * 100
            
            # 準備統計數據
            stats_data = {
                'date': today,
                'symbol': self.symbol,
                'maker_buy_volume': self.maker_buy_volume,
                'maker_sell_volume': self.maker_sell_volume,
                'taker_buy_volume': self.taker_buy_volume,
                'taker_sell_volume': self.taker_sell_volume,
                'realized_profit': self.total_profit,
                'total_fees': self.total_fees,
                'net_profit': self.total_profit - self.total_fees,
                'avg_spread': avg_spread,
                'trade_count': self.trades_executed,
                'volatility': volatility
            }
            
            # 使用專門的函數來處理數據庫操作
            def safe_update_stats():
                try:
                    success = self.db.update_trading_stats(stats_data)
                    if not success:
                        logger.warning("更新交易統計失敗，下次再試")
                except Exception as db_err:
                    logger.error(f"更新交易統計時出錯: {db_err}")
            
            # 直接在當前線程執行，避免過多的並發操作
            safe_update_stats()
                
        except Exception as e:
            logger.error(f"更新交易統計數據時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_average_buy_cost(self):
        """計算平均買入成本"""
        if not self.buy_trades:
            return 0
            
        total_buy_cost = sum(price * quantity for price, quantity in self.buy_trades)
        total_buy_quantity = sum(quantity for _, quantity in self.buy_trades)
        
        if not self.sell_trades or total_buy_quantity <= 0:
            return total_buy_cost / total_buy_quantity if total_buy_quantity > 0 else 0
        
        buy_queue = self.buy_trades.copy()
        consumed_cost = 0
        consumed_quantity = 0
        
        for _, sell_quantity in self.sell_trades:
            remaining_sell = sell_quantity
            
            while remaining_sell > 0 and buy_queue:
                buy_price, buy_quantity = buy_queue[0]
                matched_quantity = min(remaining_sell, buy_quantity)
                consumed_cost += buy_price * matched_quantity
                consumed_quantity += matched_quantity
                remaining_sell -= matched_quantity
                
                if matched_quantity >= buy_quantity:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_price, buy_quantity - matched_quantity)
        
        remaining_buy_quantity = total_buy_quantity - consumed_quantity
        remaining_buy_cost = total_buy_cost - consumed_cost
        
        if remaining_buy_quantity <= 0:
            if self.ws and self.ws.connected and self.ws.bid_price:
                return self.ws.bid_price
            return 0
        
        return remaining_buy_cost / remaining_buy_quantity
    
    def _calculate_session_profit(self):
        """計算本次執行的已實現利潤"""
        if not self.session_buy_trades or not self.session_sell_trades:
            return 0

        buy_queue = self.session_buy_trades.copy()
        total_profit = 0

        for sell_price, sell_quantity in self.session_sell_trades:
            remaining_sell = sell_quantity

            while remaining_sell > 0 and buy_queue:
                buy_price, buy_quantity = buy_queue[0]
                matched_quantity = min(remaining_sell, buy_quantity)

                # 計算這筆交易的利潤
                trade_profit = (sell_price - buy_price) * matched_quantity
                total_profit += trade_profit

                remaining_sell -= matched_quantity
                if matched_quantity >= buy_quantity:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_price, buy_quantity - matched_quantity)

        return total_profit

    def calculate_pnl(self):
        """計算已實現和未實現PnL"""
        # 總的已實現利潤
        realized_pnl = self._calculate_db_profit()
        
        # 本次執行的已實現利潤
        session_realized_pnl = self._calculate_session_profit()
        
        # 計算未實現利潤
        unrealized_pnl = 0
        net_position = self.total_bought - self.total_sold
        
        if net_position > 0:
            current_price = self.get_current_price()
            if current_price:
                avg_buy_cost = self._calculate_average_buy_cost()
                unrealized_pnl = (current_price - avg_buy_cost) * net_position
        
        # 返回總的PnL和本次執行的PnL
        return realized_pnl, unrealized_pnl, self.total_fees, realized_pnl - self.total_fees, session_realized_pnl, self.session_fees, session_realized_pnl - self.session_fees
    
    def get_current_price(self):
        """獲取當前價格（優先使用WebSocket數據）"""
        self.check_ws_connection()
        price = None
        if self.ws and self.ws.connected:
            price = self.ws.get_current_price()
        
        if price is None:
            ticker = get_ticker(self.symbol)
            if isinstance(ticker, dict) and "error" in ticker:
                logger.error(f"獲取價格失敗: {ticker['error']}")
                return None
            
            if "lastPrice" not in ticker:
                logger.error(f"獲取到的價格數據不完整: {ticker}")
                return None
            return float(ticker['lastPrice'])
        return price
    
    def get_market_depth(self):
        """獲取市場深度（優先使用WebSocket數據）"""
        self.check_ws_connection()
        bid_price, ask_price = None, None
        if self.ws and self.ws.connected:
            bid_price, ask_price = self.ws.get_bid_ask()
        
        if bid_price is None or ask_price is None:
            order_book = get_order_book(self.symbol)
            if isinstance(order_book, dict) and "error" in order_book:
                logger.error(f"獲取訂單簿失敗: {order_book['error']}")
                return None, None
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            if not bids or not asks:
                return None, None
            
            highest_bid = float(bids[0][0]) if bids else None
            lowest_ask = float(asks[0][0]) if asks else None
            
            return highest_bid, lowest_ask
        
        return bid_price, ask_price
    
    def calculate_dynamic_spread(self):
        """计算更合理的动态价差"""
        # 基础手续费
        MAKER_FEE = 0.0008  # 0.08%
        
        # 获取当前市场价差
        market_spread = self._get_market_spread()
        
        # 计算最小价差
        min_spread_for_fees = (MAKER_FEE * 2) * 100  # 基础手续费价差
        
        # 降低利润倍数，从1.5降到1.2
        base_profit_multiplier = 1.2
        
        # 计算基础价差，考虑市场实际情况
        min_profitable_spread = min_spread_for_fees * base_profit_multiplier
        
        # 使用市场价差作为参考
        market_based_spread = market_spread * 1.1  # 略高于市场价差
        
        # 选择合适的基础价差
        base_spread = min(
            max(min_profitable_spread, market_based_spread),
            0.15  # 设置最大价差上限为0.15%
        )
        
        # 获取市场深度信息
        depth_score = self._calculate_depth_score()
        
        # 根据市场深度调整价差
        if depth_score > 0.7:  # 深度好
            base_spread *= 0.9
        
        # 更温和的波动率调整
        volatility = self._calculate_volatility()
        if volatility > 0:
            if volatility > self.volatility_threshold:
                adjusted_spread = base_spread * (1 + volatility)  # 降低波动率影响
            else:
                adjusted_spread = base_spread * (1 - volatility * 0.3)  # 低波动时可以降低价差
        else:
            adjusted_spread = base_spread
        
        # 确保最终价差在合理范围内
        final_spread = max(
            min(adjusted_spread, 0.15),  # 最大不超过0.15%
            min_profitable_spread * 0.9   # 最小不低于基础盈利价差的90%
        )
        
        logger.info(f"市场价差: {market_spread:.4f}%")
        logger.info(f"基础盈利价差: {min_profitable_spread:.4f}%")
        logger.info(f"调整后价差: {final_spread:.4f}%")
        
        return final_spread

    def _get_market_spread(self):
        """获取当前市场价差"""
        if not self.ws.orderbook:
            return 0.1  # 默认值
            
        bids = self.ws.orderbook['bids']
        asks = self.ws.orderbook['asks']
        
        if not bids or not asks:
            return 0.1
            
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        spread_pct = ((best_ask - best_bid) / best_bid) * 100
        return spread_pct
    
    def calculate_prices(self):
        """计算买卖订单价格"""
        try:
            bid_price, ask_price = self.get_market_depth()
            if bid_price is None or ask_price is None:
                return None, None
            
            mid_price = (bid_price + ask_price) / 2
            
            # 使用新的价差计算方法
            spread_percentage = self.calculate_dynamic_spread()
            exact_spread = mid_price * (spread_percentage / 100)
            
            # 更温和的价格调整
            base_buy_price = mid_price - (exact_spread / 2)
            base_sell_price = mid_price + (exact_spread / 2)
            
            # 确保价格合理性
            if base_buy_price < bid_price * 0.995:  # 不要离最佳价格太远
                base_buy_price = bid_price * 0.995
            if base_sell_price > ask_price * 1.005:
                base_sell_price = ask_price * 1.005
            
            # 记录价格信息
            logger.info(f"市场中间价: {mid_price}")
            logger.info(f"买入价: {base_buy_price} (距离最佳买价: {((bid_price - base_buy_price) / bid_price * 100):.4f}%)")
            logger.info(f"卖出价: {base_sell_price} (距离最佳卖价: {((base_sell_price - ask_price) / ask_price * 100):.4f}%)")
            
            return base_buy_price, base_sell_price
        
        except Exception as e:
            logger.error(f"计算价格时出错: {str(e)}")
            return None, None
    
    def need_rebalance(self):
        """判斷是否需要重平衡倉位"""
        if self.total_bought == 0 and self.total_sold == 0:
            return False
            
        # 計算淨部位
        net_position = self.total_bought - self.total_sold
        if net_position == 0:
            return False
            
        # 獲取當前賬戶餘額
        balances = get_balance(self.api_key, self.secret_key)
        if isinstance(balances, dict) and "error" in balances:
            logger.error(f"獲取餘額失敗: {balances['error']}")
            # 如果無法獲取餘額，則使用舊方法
            imbalance_percentage = abs(net_position) / max(self.total_bought, self.total_sold) * 100
            return imbalance_percentage > self.rebalance_threshold
            
        # 計算總資產價值（以報價貨幣計算）
        total_assets = 0
        
        # 獲取當前價格，用於轉換基礎貨幣到報價貨幣
        current_price = self.get_current_price()
        if not current_price:
            logger.warning("無法獲取當前價格，使用舊方法計算不平衡度")
            imbalance_percentage = abs(net_position) / max(self.total_bought, self.total_sold) * 100
            return imbalance_percentage > self.rebalance_threshold
        
        # 累加所有資產價值
        for asset, details in balances.items():
            available = float(details.get('available', 0))
            locked = float(details.get('locked', 0))
            total = available + locked
            
            if asset == self.quote_asset:
                # 報價貨幣直接加入
                total_assets += total
            elif asset == self.base_asset:
                # 基礎貨幣轉換為報價貨幣的價值
                total_assets += total * current_price
        
        # 計算淨部位價值（以報價貨幣計）
        net_position_value = abs(net_position) * current_price
        
        # 計算風險暴露比例
        risk_exposure = (net_position_value / total_assets) * 100 if total_assets > 0 else 0
        
        logger.info(f"當前淨部位: {net_position} {self.base_asset} (價值: {net_position_value:.2f} {self.quote_asset})")
        logger.info(f"總資產價值: {total_assets:.2f} {self.quote_asset}")
        logger.info(f"風險暴露比例: {risk_exposure:.2f}%")
        
        return risk_exposure > self.rebalance_threshold
    
    def rebalance_position(self):
        """重平衡倉位"""
        logger.info("開始重新平衡倉位...")
        self.check_ws_connection()
        
        imbalance = self.total_bought - self.total_sold
        bid_price, ask_price = self.get_market_depth()
        
        if bid_price is None or ask_price is None:
            current_price = self.get_current_price()
            if current_price is None:
                logger.error("無法獲取價格，無法重新平衡")
                return
            bid_price = current_price * 0.998
            ask_price = current_price * 1.002
        
        if imbalance > 0:
            # 淨多頭，需要賣出
            quantity = round_to_precision(imbalance, self.base_precision)
            if quantity < self.min_order_size:
                logger.info(f"不平衡量 {quantity} 低於最小訂單大小 {self.min_order_size}，不進行重新平衡")
                return
            
            # 設定賣出價格
            price_factor = 1.0
            sell_price = round_to_tick_size(bid_price * price_factor, self.tick_size)
            logger.info(f"執行重新平衡: 賣出 {quantity} {self.base_asset} @ {sell_price}")
            
            # 構建訂單
            order_details = {
                "orderType": "Limit",
                "price": str(sell_price),
                "quantity": str(quantity),
                "side": "Ask",
                "symbol": self.symbol,
                "timeInForce": "GTC",
                "postOnly": True
            }
            
            # 嘗試執行訂單
            result = execute_order(self.api_key, self.secret_key, order_details)
            
            # 處理可能的錯誤
            if isinstance(result, dict) and "error" in result:
                error_msg = str(result['error'])
                logger.error(f"重新平衡賣單執行失敗: {error_msg}")
                
                # 如果因為訂單會立即成交而失敗，嘗試不使用postOnly
                if "POST_ONLY_TAKER" in error_msg or "Order would immediately match" in error_msg:
                    logger.info("嘗試使用非postOnly訂單進行重新平衡...")
                    order_details.pop("postOnly", None)
                    result = execute_order(self.api_key, self.secret_key, order_details)
                    
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"非postOnly賣單執行失敗: {result['error']}")
                    else:
                        logger.info(f"非postOnly賣單執行成功，價格: {sell_price}")
                        # 記錄這是一個重平衡訂單
                        if 'id' in result:
                            self.db.record_rebalance_order(result['id'], self.symbol)
            else:
                logger.info("重新平衡賣單已提交，作為maker")
                # 記錄這是一個重平衡訂單
                if 'id' in result:
                    self.db.record_rebalance_order(result['id'], self.symbol)
            
        elif imbalance < 0:
            # 淨空頭，需要買入
            quantity = round_to_precision(abs(imbalance), self.base_precision)
            if quantity < self.min_order_size:
                logger.info(f"不平衡量 {quantity} 低於最小訂單大小 {self.min_order_size}，不進行重新平衡")
                return
            
            # 設定買入價格
            price_factor = 1.0
            buy_price = round_to_tick_size(ask_price * price_factor, self.tick_size)
            logger.info(f"執行重新平衡: 買入 {quantity} {self.base_asset} @ {buy_price}")
            
            # 構建訂單
            order_details = {
                "orderType": "Limit",
                "price": str(buy_price),
                "quantity": str(quantity),
                "side": "Bid",
                "symbol": self.symbol,
                "timeInForce": "GTC",
                "postOnly": True
            }
            
            # 嘗試執行訂單
            result = execute_order(self.api_key, self.secret_key, order_details)
            
            # 處理可能的錯誤
            if isinstance(result, dict) and "error" in result:
                error_msg = str(result['error'])
                logger.error(f"重新平衡買單執行失敗: {error_msg}")
                
                # 如果因為訂單會立即成交而失敗，嘗試不使用postOnly
                if "POST_ONLY_TAKER" in error_msg or "Order would immediately match" in error_msg:
                    logger.info("嘗試使用非postOnly訂單進行重新平衡...")
                    order_details.pop("postOnly", None)
                    result = execute_order(self.api_key, self.secret_key, order_details)
                    
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"非postOnly買單執行失敗: {result['error']}")
                    else:
                        logger.info(f"非postOnly買單執行成功，價格: {buy_price}")
                        # 記錄這是一個重平衡訂單
                        if 'id' in result:
                            self.db.record_rebalance_order(result['id'], self.symbol)
            else:
                logger.info("重平衡買單已提交，作為maker")
                # 記錄這是一個重平衡訂單
                if 'id' in result:
                    self.db.record_rebalance_order(result['id'], self.symbol)
        
        logger.info("倉位重新平衡完成")
    
    def subscribe_order_updates(self):
        """訂閲訂單更新流"""
        if not self.ws or not self.ws.is_connected():
            logger.warning("無法訂閲訂單更新：WebSocket連接不可用")
            return False
        
        # 嘗試訂閲訂單更新流
        stream = f"account.orderUpdate.{self.symbol}"
        if stream not in self.ws.subscriptions:
            retry_count = 0
            max_retries = 3
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    success = self.ws.private_subscribe(stream)
                    if success:
                        logger.info(f"成功訂閲訂單更新: {stream}")
                        return True
                    else:
                        logger.warning(f"訂閲訂單更新失敗，嘗試重試... ({retry_count+1}/{max_retries})")
                except Exception as e:
                    logger.error(f"訂閲訂單更新時發生異常: {e}")
                
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # 重試前等待
            
            if not success:
                logger.error(f"在 {max_retries} 次嘗試後仍無法訂閲訂單更新")
                return False
        else:
            logger.info(f"已經訂閲了訂單更新: {stream}")
            return True
    
    def place_limit_orders(self):
        """下限价单"""
        self.check_ws_connection()
        self.cancel_existing_orders()
        
        # 获取买卖价格
        buy_price, sell_price = self.calculate_prices()
        if buy_price is None or sell_price is None:
            logger.error("無法計算訂單價格，跳過下單")
            return
            
        try:
            # 获取当前余额
            balances = get_balance(self.api_key, self.secret_key)
            if isinstance(balances, dict) and "error" in balances:
                logger.error(f"獲取餘額失敗: {balances['error']}")
                return
                
            base_balance = 0
            quote_balance = 0
            
            for asset, details in balances.items():
                if asset == self.base_asset:
                    base_balance = float(details.get('available', 0))
                elif asset == self.quote_asset:
                    quote_balance = float(details.get('available', 0))
                    
            logger.info(f"當前余額 - {self.base_asset}: {base_balance}, {self.quote_asset}: {quote_balance}")
            
            if base_balance == 0 and quote_balance == 0:
                logger.warning("賬戶余額為0，無法下單")
                return
                
            # 如果order_quantity为None，根据余额计算合适的订单数量
            if self.order_quantity is None:
                # 使用当前买入价格计算数量
                avg_price = buy_price  # 修改这里，直接使用买入价格
                
                # 提高资金利用率，根据市场深度动态调整
                order_book = get_order_book(self.symbol)
                if isinstance(order_book, dict) and "error" in order_book:
                    allocation_percent = 0.05  # 默认值
                else:
                    # 分析市场深度
                    total_bid_depth = sum(float(bid[1]) for bid in order_book['bids'][:5])
                    total_ask_depth = sum(float(ask[1]) for ask in order_book['asks'][:5])
                    avg_market_depth = (total_bid_depth + total_ask_depth) / 2
                    
                    # 根据市场深度动态调整资金使用比例
                    base_allocation = 0.08  # 基础分配比例
                    depth_factor = min(1.5, max(0.8, avg_market_depth / (base_balance if base_balance > 0 else 1)))
                    allocation_percent = base_allocation * depth_factor
                    
                    # 确保总资金使用不超过50%
                    total_allocation = allocation_percent * self.max_orders
                    if total_allocation > 0.5:
                        allocation_percent = 0.5 / self.max_orders
                
                # 计算买入和卖出订单的最大可用数量
                max_buy_quantity = quote_balance / avg_price / self.max_orders
                max_sell_quantity = base_balance / self.max_orders
                
                # 确保买入数量不超过可用资金
                buy_quantity = min(
                    max_buy_quantity,
                    quote_balance * allocation_percent / avg_price
                )
                buy_quantity = round_to_precision(buy_quantity, self.base_precision)
                
                # 确保卖出数量不超过可用余额
                sell_quantity = min(
                    max_sell_quantity,
                    base_balance * allocation_percent
                )
                sell_quantity = round_to_precision(sell_quantity, self.base_precision)
            else:
                # 使用固定的order_quantity
                # 计算买单数量
                buy_quantity = min(
                    self.order_quantity,
                    quote_balance / buy_price if buy_price > 0 else 0
                )
                buy_quantity = round_to_precision(buy_quantity, self.base_precision)
                
                # 计算卖单数量
                sell_quantity = min(
                    self.order_quantity,
                    base_balance
                )
                sell_quantity = round_to_precision(sell_quantity, self.base_precision)
            
            # 检查下单数量是否满足最小要求
            if buy_quantity < self.min_order_size:
                logger.warning(f"買單數量 {buy_quantity} 小於最小下單量 {self.min_order_size}")
                buy_quantity = 0
                
            if sell_quantity < self.min_order_size:
                logger.warning(f"賣單數量 {sell_quantity} 小於最小下單量 {self.min_order_size}")
                sell_quantity = 0
                
            # 记录下单信息
            logger.info(f"準備下單 - 買入: {buy_quantity}@{buy_price}, 賣出: {sell_quantity}@{sell_price}")
            
            # 如果买卖单数量都为0，直接返回
            if buy_quantity == 0 and sell_quantity == 0:
                logger.warning("沒有足夠的資金或資產進行下單")
                return
                
            # 下买单
            if buy_quantity > 0:
                try:
                    # 构建买单
                    buy_order = {
                        "orderType": "Limit",
                        "price": str(buy_price),
                        "quantity": str(buy_quantity),
                        "side": "Bid",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    }
                    
                    # 执行买单
                    result = execute_order(self.api_key, self.secret_key, buy_order)
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"買單執行失敗: {result['error']}")
                    else:
                        logger.info(f"買單已提交: {result}")
                        self.orders_placed += 1
                except Exception as e:
                    logger.error(f"買單提交失敗: {str(e)}")
                    
            # 下卖单
            if sell_quantity > 0:
                try:
                    # 构建卖单
                    sell_order = {
                        "orderType": "Limit",
                        "price": str(sell_price),
                        "quantity": str(sell_quantity),
                        "side": "Ask",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    }
                    
                    # 执行卖单
                    result = execute_order(self.api_key, self.secret_key, sell_order)
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"賣單執行失敗: {result['error']}")
                    else:
                        logger.info(f"賣單已提交: {result}")
                        self.orders_placed += 1
                except Exception as e:
                    logger.error(f"賣單提交失敗: {str(e)}")
                    
        except Exception as e:
            logger.error(f"下單過程發生錯誤: {str(e)}")
    
    def _adjust_quantity_by_market(self, base_quantity, side):
        """根据市场情况动态调整订单数量"""
        try:
            # 获取24小时成交量
            ticker = get_ticker(self.symbol)
            volume_24h = float(ticker.get('volume', 0))
            
            # 计算平均每小时成交量
            avg_hourly_volume = volume_24h / 24
            
            # 根据成交量调整订单大小
            volume_factor = min(1.5, max(0.5, avg_hourly_volume / base_quantity))
            
            # 获取订单簿深度
            order_book = get_order_book(self.symbol)
            depth_factor = 1.0
            
            if side == 'buy':
                total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:5])
                depth_factor = min(1.5, max(0.5, total_ask_volume / base_quantity))
            else:
                total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:5])
                depth_factor = min(1.5, max(0.5, total_bid_volume / base_quantity))
            
            # 综合调整因子
            adjustment_factor = (volume_factor + depth_factor) / 2
            
            # 调整数量并确保精度正确
            adjusted_quantity = base_quantity * adjustment_factor
            
            # 确保在最小订单大小和最大限制之间
            final_quantity = max(
                self.min_order_size, 
                min(adjusted_quantity, base_quantity * 2)
            )
            
            # 根据基础资产精度进行四舍五入
            final_quantity = round_to_precision(final_quantity, self.base_precision)
            
            # 额外检查确保不超过精度限制
            quantity_str = str(final_quantity)
            decimal_places = len(quantity_str.split('.')[-1]) if '.' in quantity_str else 0
            
            if decimal_places > self.base_precision:
                # 如果精度超出限制，则截断到允许的精度
                final_quantity = float(f"%.{self.base_precision}f" % final_quantity)
            
            logger.info(f"调整后订单数量: {final_quantity} (原始: {base_quantity}, 精度: {self.base_precision})")
            return final_quantity
                      
        except Exception as e:
            logger.error(f"调整订单数量时出错: {e}")
            # 发生错误时返回安全的数量
            return round_to_precision(base_quantity, self.base_precision)
    
    def cancel_existing_orders(self):
        """取消所有現有訂單"""
        open_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
        
        if isinstance(open_orders, dict) and "error" in open_orders:
            logger.error(f"獲取訂單失敗: {open_orders['error']}")
            return
        
        if not open_orders:
            logger.info("沒有需要取消的現有訂單")
            self.active_buy_orders = []
            self.active_sell_orders = []
            return
        
        logger.info(f"正在取消 {len(open_orders)} 個現有訂單")
        
        try:
            # 嘗試批量取消
            result = cancel_all_orders(self.api_key, self.secret_key, self.symbol)
            
            if isinstance(result, dict) and "error" in result:
                logger.error(f"批量取消訂單失敗: {result['error']}")
                logger.info("嘗試逐個取消...")
                
                # 初始化線程池
                with ThreadPoolExecutor(max_workers=5) as executor:
                    cancel_futures = []
                    
                    # 提交取消訂單任務
                    for order in open_orders:
                        order_id = order.get('id')
                        if not order_id:
                            continue
                        
                        future = executor.submit(
                            cancel_order, 
                            self.api_key, 
                            self.secret_key, 
                            order_id, 
                            self.symbol
                        )
                        cancel_futures.append((order_id, future))
                    
                    # 處理結果
                    for order_id, future in cancel_futures:
                        try:
                            res = future.result()
                            if isinstance(res, dict) and "error" in res:
                                logger.error(f"取消訂單 {order_id} 失敗: {res['error']}")
                            else:
                                logger.info(f"取消訂單 {order_id} 成功")
                                self.orders_cancelled += 1
                        except Exception as e:
                            logger.error(f"取消訂單 {order_id} 時出錯: {e}")
            else:
                logger.info("批量取消訂單成功")
                self.orders_cancelled += len(open_orders)
        except Exception as e:
            logger.error(f"取消訂單過程中發生錯誤: {str(e)}")
        
        # 等待一下確保訂單已取消
        time.sleep(1)
        
        # 檢查是否還有未取消的訂單
        remaining_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
        if remaining_orders and len(remaining_orders) > 0:
            logger.warning(f"警告: 仍有 {len(remaining_orders)} 個未取消的訂單")
        else:
            logger.info("所有訂單已成功取消")
        
        # 重置活躍訂單列表
        self.active_buy_orders = []
        self.active_sell_orders = []
    
    def check_order_fills(self):
        """檢查訂單成交情況"""
        open_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
        
        if isinstance(open_orders, dict) and "error" in open_orders:
            logger.error(f"獲取訂單失敗: {open_orders['error']}")
            return
        
        # 獲取當前所有訂單ID
        current_order_ids = set()
        if open_orders:
            for order in open_orders:
                order_id = order.get('id')
                if order_id:
                    current_order_ids.add(order_id)
        
        # 記錄更新前的訂單數量
        prev_buy_orders = len(self.active_buy_orders)
        prev_sell_orders = len(self.active_sell_orders)
        
        # 更新活躍訂單列表
        active_buy_orders = []
        active_sell_orders = []
        
        if open_orders:
            for order in open_orders:
                if order.get('side') == 'Bid':
                    active_buy_orders.append(order)
                elif order.get('side') == 'Ask':
                    active_sell_orders.append(order)
        
        # 檢查買單成交
        filled_buy_orders = []
        for order in self.active_buy_orders:
            order_id = order.get('id')
            if order_id and order_id not in current_order_ids:
                price = float(order.get('price', 0))
                quantity = float(order.get('quantity', 0))
                logger.info(f"買單已成交: {price} x {quantity}")
                filled_buy_orders.append(order)
        
        # 檢查賣單成交
        filled_sell_orders = []
        for order in self.active_sell_orders:
            order_id = order.get('id')
            if order_id and order_id not in current_order_ids:
                price = float(order.get('price', 0))
                quantity = float(order.get('quantity', 0))
                logger.info(f"賣單已成交: {price} x {quantity}")
                filled_sell_orders.append(order)
        
        # 更新活躍訂單列表
        self.active_buy_orders = active_buy_orders
        self.active_sell_orders = active_sell_orders
        
        # 輸出訂單數量變化，方便追蹤
        if prev_buy_orders != len(active_buy_orders) or prev_sell_orders != len(active_sell_orders):
            logger.info(f"訂單數量變更: 買單 {prev_buy_orders} -> {len(active_buy_orders)}, 賣單 {prev_sell_orders} -> {len(active_sell_orders)}")
        
        logger.info(f"當前活躍訂單: 買單 {len(self.active_buy_orders)} 個, 賣單 {len(self.active_sell_orders)} 個")
    
    def estimate_profit(self):
        """估算潛在利潤"""
        # 計算活躍買賣單的平均價格
        avg_buy_price = 0
        total_buy_quantity = 0
        for order in self.active_buy_orders:
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            avg_buy_price += price * quantity
            total_buy_quantity += quantity
        
        if total_buy_quantity > 0:
            avg_buy_price /= total_buy_quantity
        
        avg_sell_price = 0
        total_sell_quantity = 0
        for order in self.active_sell_orders:
            price = float(order.get('price', 0))
            quantity = float(order.get('quantity', 0))
            avg_sell_price += price * quantity
            total_sell_quantity += quantity
        
        if total_sell_quantity > 0:
            avg_sell_price /= total_sell_quantity
        
        # 計算總的PnL和本次執行的PnL
        realized_pnl, unrealized_pnl, total_fees, net_pnl, session_realized_pnl, session_fees, session_net_pnl = self.calculate_pnl()
        
        # 計算活躍訂單的潛在利潤
        if avg_buy_price > 0 and avg_sell_price > 0:
            spread = avg_sell_price - avg_buy_price
            spread_percentage = (spread / avg_buy_price) * 100
            min_quantity = min(total_buy_quantity, total_sell_quantity)
            potential_profit = spread * min_quantity
            
            logger.info(f"估算利潤: 買入均價 {avg_buy_price:.8f}, 賣出均價 {avg_sell_price:.8f}")
            logger.info(f"價差: {spread:.8f} ({spread_percentage:.2f}%), 潛在利潤: {potential_profit:.8f} {self.quote_asset}")
            logger.info(f"已實現利潤(總): {realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"總手續費(總): {total_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤(總): {net_pnl:.8f} {self.quote_asset}")
            logger.info(f"未實現利潤: {unrealized_pnl:.8f} {self.quote_asset}")
            
            # 打印本次執行的統計信息
            logger.info("\n---本次執行統計---")
            logger.info(f"本次執行已實現利潤: {session_realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"本次執行手續費: {session_fees:.8f} {self.quote_asset}")
            logger.info(f"本次執行凈利潤: {session_net_pnl:.8f} {self.quote_asset}")
            
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            
            logger.info(f"本次執行買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
            
        else:
            logger.info("無法估算潛在利潤: 缺少活躍的買/賣訂單")
            logger.info(f"已實現利潤(總): {realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"總手續費(總): {total_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤(總): {net_pnl:.8f} {self.quote_asset}")
            logger.info(f"未實現利潤: {unrealized_pnl:.8f} {self.quote_asset}")
            
            # 打印本次執行的統計信息
            logger.info("\n---本次執行統計---")
            logger.info(f"本次執行已實現利潤: {session_realized_pnl:.8f} {self.quote_asset}")
            logger.info(f"本次執行手續費: {session_fees:.8f} {self.quote_asset}")
            logger.info(f"本次執行凈利潤: {session_net_pnl:.8f} {self.quote_asset}")
            
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            
            logger.info(f"本次執行買入量: {session_buy_volume} {self.base_asset}, 賣出量: {session_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Maker買入: {self.session_maker_buy_volume} {self.base_asset}, Maker賣出: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"本次執行Taker買入: {self.session_taker_buy_volume} {self.base_asset}, Taker賣出: {self.session_taker_sell_volume} {self.base_asset}")
    
    def print_trading_stats(self):
        """打印交易統計報表"""
        try:
            logger.info("\n=== 做市商交易統計 ===")
            logger.info(f"交易對: {self.symbol}")
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 獲取今天的統計數據
            today_stats = self.db.get_trading_stats(self.symbol, today)
            
            if today_stats and len(today_stats) > 0:
                stat = today_stats[0]
                maker_buy = stat['maker_buy_volume']
                maker_sell = stat['maker_sell_volume']
                taker_buy = stat['taker_buy_volume']
                taker_sell = stat['taker_sell_volume']
                profit = stat['realized_profit']
                fees = stat['total_fees']
                net = stat['net_profit']
                avg_spread = stat['avg_spread']
                volatility = stat['volatility']
                
                total_volume = maker_buy + maker_sell + taker_buy + taker_sell
                maker_percentage = ((maker_buy + maker_sell) / total_volume * 100) if total_volume > 0 else 0
                
                logger.info(f"\n今日統計 ({today}):")
                logger.info(f"Maker買入量: {maker_buy} {self.base_asset}")
                logger.info(f"Maker賣出量: {maker_sell} {self.base_asset}")
                logger.info(f"Taker買入量: {taker_buy} {self.base_asset}")
                logger.info(f"Taker賣出量: {taker_sell} {self.base_asset}")
                logger.info(f"總成交量: {total_volume} {self.base_asset}")
                logger.info(f"Maker佔比: {maker_percentage:.2f}%")
                logger.info(f"平均價差: {avg_spread:.4f}%")
                logger.info(f"波動率: {volatility:.4f}%")
                logger.info(f"毛利潤: {profit:.8f} {self.quote_asset}")
                logger.info(f"總手續費: {fees:.8f} {self.quote_asset}")
                logger.info(f"凈利潤: {net:.8f} {self.quote_asset}")
            
            # 獲取所有時間的總計
            all_time_stats = self.db.get_all_time_stats(self.symbol)
            
            if all_time_stats:
                total_maker_buy = all_time_stats['total_maker_buy']
                total_maker_sell = all_time_stats['total_maker_sell']
                total_taker_buy = all_time_stats['total_taker_buy']
                total_taker_sell = all_time_stats['total_taker_sell']
                total_profit = all_time_stats['total_profit']
                total_fees = all_time_stats['total_fees']
                total_net = all_time_stats['total_net_profit']
                avg_spread = all_time_stats['avg_spread_all_time']
                
                total_volume = total_maker_buy + total_maker_sell + total_taker_buy + total_taker_sell
                maker_percentage = ((total_maker_buy + total_maker_sell) / total_volume * 100) if total_volume > 0 else 0
                
                logger.info("\n累計統計:")
                logger.info(f"Maker買入量: {total_maker_buy} {self.base_asset}")
                logger.info(f"Maker賣出量: {total_maker_sell} {self.base_asset}")
                logger.info(f"Taker買入量: {total_taker_buy} {self.base_asset}")
                logger.info(f"Taker賣出量: {total_taker_sell} {self.base_asset}")
                logger.info(f"總成交量: {total_volume} {self.base_asset}")
                logger.info(f"Maker佔比: {maker_percentage:.2f}%")
                logger.info(f"平均價差: {avg_spread:.4f}%")
                logger.info(f"毛利潤: {total_profit:.8f} {self.quote_asset}")
                logger.info(f"總手續費: {total_fees:.8f} {self.quote_asset}")
                logger.info(f"凈利潤: {total_net:.8f} {self.quote_asset}")
            
            # 添加本次執行的統計
            session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
            session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
            session_total_volume = session_buy_volume + session_sell_volume
            session_maker_volume = self.session_maker_buy_volume + self.session_maker_sell_volume
            session_maker_percentage = (session_maker_volume / session_total_volume * 100) if session_total_volume > 0 else 0
            session_profit = self._calculate_session_profit()
            
            logger.info(f"\n本次執行統計 (從 {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')} 開始):")
            logger.info(f"Maker買入量: {self.session_maker_buy_volume} {self.base_asset}")
            logger.info(f"Maker賣出量: {self.session_maker_sell_volume} {self.base_asset}")
            logger.info(f"Taker買入量: {self.session_taker_buy_volume} {self.base_asset}")
            logger.info(f"Taker賣出量: {self.session_taker_sell_volume} {self.base_asset}")
            logger.info(f"總成交量: {session_total_volume} {self.base_asset}")
            logger.info(f"Maker佔比: {session_maker_percentage:.2f}%")
            logger.info(f"毛利潤: {session_profit:.8f} {self.quote_asset}")
            logger.info(f"總手續費: {self.session_fees:.8f} {self.quote_asset}")
            logger.info(f"凈利潤: {(session_profit - self.session_fees):.8f} {self.quote_asset}")
                
            # 查詢前10筆最新成交
            recent_trades = self.db.get_recent_trades(self.symbol, 10)
            
            if recent_trades and len(recent_trades) > 0:
                logger.info("\n最近10筆成交:")
                for i, trade in enumerate(recent_trades):
                    maker_str = "Maker" if trade['maker'] else "Taker"
                    logger.info(f"{i+1}. {trade['timestamp']} - {trade['side']} {trade['quantity']} @ {trade['price']} ({maker_str}) 手續費: {trade['fee']:.8f}")
        
        except Exception as e:
            logger.error(f"打印交易統計時出錯: {e}")
    
    def _ensure_data_streams(self):
        """確保所有必要的數據流訂閲都是活躍的"""
        # 檢查深度流訂閲
        if "depth" not in self.ws.subscriptions:
            logger.info("重新訂閲深度數據流...")
            self.ws.initialize_orderbook()  # 重新初始化訂單簿
            self.ws.subscribe_depth()
        
        # 檢查行情數據訂閲
        if "bookTicker" not in self.ws.subscriptions:
            logger.info("重新訂閲行情數據...")
            self.ws.subscribe_bookTicker()
        
        # 檢查私有訂單更新流
        if f"account.orderUpdate.{self.symbol}" not in self.ws.subscriptions:
            logger.info("重新訂閲私有訂單更新流...")
            self.subscribe_order_updates()
    
    def analyze_market_conditions(self):
        """分析市场状况并调整参数"""
        try:
            # 获取当前时间
            current_time = time.time()
            
            # 如果距离上次分析时间不足5分钟，则跳过
            if hasattr(self, 'last_analysis_time') and current_time - self.last_analysis_time < 300:
                return
            
            self.last_analysis_time = current_time
            
            # 计算市场波动率
            volatility = self._calculate_volatility()
            
            # 计算市场趋势
            trend = self._calculate_trend()
            
            # 计算市场深度得分
            depth_score = self._calculate_depth_score()
            
            # 计算成交量水平
            volume_level = self._calculate_volume_level()
            
            # 计算当前点差水平
            spread_level = self._calculate_spread_level()
            
            # 更新市场状态
            self.market_state = {
                'volatility': volatility,
                'trend': trend,
                'depth_score': depth_score,
                'volume_level': volume_level,
                'spread_level': spread_level
            }
            
            # 根据市场状态调整参数
            self._adjust_parameters()
            
            logger.info("\n市场分析结果:")
            logger.info(f"波动率: {volatility:.4f}")
            logger.info(f"趋势: {trend:.4f}")
            logger.info(f"深度得分: {depth_score:.4f}")
            logger.info(f"成交量水平: {volume_level:.4f}")
            logger.info(f"点差水平: {spread_level:.4f}")
            
        except Exception as e:
            logger.error(f"市场分析出错: {str(e)}")
            
    def _calculate_volatility(self):
        """计算市场波动率"""
        try:
            # 使用client.py中的get_klines方法
            klines = get_klines(self.symbol, interval='1m', limit=30)
            if isinstance(klines, dict) and "error" in klines:
                logger.error(f"获取K线数据失败: {klines['error']}")
                return 0.0
                
            if not klines:
                return 0.0
                
            # 计算收盘价的标准差
            closes = [float(k[4]) for k in klines]  # k[4]是收盘价
            std_dev = np.std(closes)
            mean_price = np.mean(closes)
            
            # 返回波动率（标准差/均价）
            volatility = std_dev / mean_price if mean_price > 0 else 0.0
            logger.info(f"当前波动率: {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logger.error(f"计算波动率出错: {str(e)}")
            return 0.0
            
    def _calculate_trend(self):
        """计算市场趋势"""
        try:
            # 使用client.py中的get_klines方法
            klines = get_klines(self.symbol, interval='5m', limit=12)
            if isinstance(klines, dict) and "error" in klines:
                logger.error(f"获取K线数据失败: {klines['error']}")
                return 0.0
                
            if not klines:
                return 0.0
                
            # 计算简单移动平均线
            closes = [float(k[4]) for k in klines]  # k[4]是收盘价
            sma_5 = np.mean(closes[-5:])
            sma_12 = np.mean(closes)
            
            # 计算趋势强度 (-1到1之间)
            trend = (sma_5 - sma_12) / sma_12 if sma_12 > 0 else 0.0
            trend = max(min(trend, 1.0), -1.0)
            
            logger.info(f"当前趋势: {trend:.4f}")
            return trend
            
        except Exception as e:
            logger.error(f"计算趋势出错: {str(e)}")
            return 0.0
            
    def _calculate_depth_score(self):
        """计算市场深度得分"""
        try:
            if not self.ws.orderbook:
                return 0.0
                
            bids = self.ws.orderbook['bids']
            asks = self.ws.orderbook['asks']
            
            if not bids or not asks:
                return 0.0
                
            # 计算买卖盘的深度
            bid_depth = sum(float(qty) for price, qty in bids[:10])
            ask_depth = sum(float(qty) for price, qty in asks[:10])
            
            # 计算深度比率
            total_depth = bid_depth + ask_depth
            if total_depth == 0:
                return 0.0
                
            # 返回深度得分 (0到1之间)
            depth_ratio = min(bid_depth, ask_depth) / max(bid_depth, ask_depth)
            return depth_ratio
            
        except Exception as e:
            logger.error(f"计算深度得分出错: {str(e)}")
            return 0.0
            
    def _calculate_volume_level(self):
        """计算成交量水平"""
        try:
            # 获取最近的K线数据
            klines = get_klines(self.symbol, interval='1m', limit=30)
            if isinstance(klines, dict) and "error" in klines:
                logger.error(f"获取K线数据失败: {klines['error']}")
                return 0.0
                
            # 计算平均成交量
            volumes = [float(k[5]) for k in klines]
            avg_volume = np.mean(volumes)
            max_volume = max(volumes)
            
            # 返回成交量水平 (0到1之间)
            return avg_volume / max_volume if max_volume > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算成交量水平出错: {str(e)}")
            return 0.0
            
    def _calculate_spread_level(self):
        """计算当前点差水平"""
        try:
            if not self.ws.orderbook:
                return 0.0
                
            bids = self.ws.orderbook['bids']
            asks = self.ws.orderbook['asks']
            
            if not bids or not asks:
                return 0.0
                
            # 计算当前点差
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            # 计算点差比率
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0
            
            # 返回标准化的点差水平 (0到1之间)
            return min(spread / self.max_spread, 1.0)
            
        except Exception as e:
            logger.error(f"计算点差水平出错: {str(e)}")
            return 0.0
            
    def _adjust_parameters(self):
        """根据市场状态调整参数"""
        try:
            # 计算市场得分
            market_score = (
                self.market_state['depth_score'] * 0.3 +
                self.market_state['volume_level'] * 0.3 +
                (1 - self.market_state['volatility']) * 0.2 +
                (1 - self.market_state['spread_level']) * 0.2
            )
            
            # 根据市场得分调整参数
            self.base_spread_percentage = self._interpolate(
                self.param_ranges['base_spread'][0],
                self.param_ranges['base_spread'][1],
                market_score
            )
            
            self.max_orders = int(self._interpolate(
                self.param_ranges['orders'][0],
                self.param_ranges['orders'][1],
                market_score
            ))
            
            self.min_profit_multiplier = self._interpolate(
                self.param_ranges['profit_multiplier'][0],
                self.param_ranges['profit_multiplier'][1],
                market_score
            )
            
            self.max_position_size = self._interpolate(
                self.param_ranges['position_size'][0],
                self.param_ranges['position_size'][1],
                market_score
            )
            
            # 特殊情况处理
            if self.market_state['volatility'] > 0.05:  # 高波动
                self.base_spread_percentage *= 1.5
                self.max_orders = max(2, self.max_orders // 2)
                
            if abs(self.market_state['trend']) > 0.8:  # 明显趋势
                trend_direction = 1 if self.market_state['trend'] > 0 else -1
                self.aggressive_factor = self._interpolate(
                    self.param_ranges['aggressive'][0],
                    self.param_ranges['aggressive'][1],
                    0.7 + 0.3 * trend_direction
                )
            
            logger.info("\n参数调整结果:")
            logger.info(f"市场得分: {market_score:.4f}")
            logger.info(f"基础点差: {self.base_spread_percentage:.4f}%")
            logger.info(f"最大订单数: {self.max_orders}")
            logger.info(f"最小利润乘数: {self.min_profit_multiplier:.4f}")
            logger.info(f"最大仓位: {self.max_position_size:.4f}")
            logger.info(f"进攻因子: {self.aggressive_factor:.4f}")
            
        except Exception as e:
            logger.error(f"调整参数出错: {str(e)}")
            
    def _interpolate(self, min_val, max_val, score):
        """线性插值计算参数值"""
        return min_val + (max_val - min_val) * score

    def run(self, duration_seconds=3600, interval_seconds=60):
        """執行做市策略"""
        logger.info(f"開始運行做市策略: {self.symbol}")
        logger.info(f"運行時間: {duration_seconds} 秒, 間隔: {interval_seconds} 秒")
        
        # 重置本次執行的統計數據
        self.session_start_time = datetime.now()
        self.session_buy_trades = []
        self.session_sell_trades = []
        self.session_fees = 0.0
        self.session_maker_buy_volume = 0.0
        self.session_maker_sell_volume = 0.0
        self.session_taker_buy_volume = 0.0
        self.session_taker_sell_volume = 0.0
        
        start_time = time.time()
        iteration = 0
        last_report_time = start_time
        report_interval = 300  # 5分鐘打印一次報表
        
        try:
            # 先確保 WebSocket 連接可用
            connection_status = self.check_ws_connection()
            if connection_status:
                # 初始化訂單簿和數據流
                if not self.ws.orderbook["bids"] and not self.ws.orderbook["asks"]:
                    self.ws.initialize_orderbook()
                
                # 檢查並確保所有數據流訂閲
                if "depth" not in self.ws.subscriptions:
                    self.ws.subscribe_depth()
                if "bookTicker" not in self.ws.subscriptions:
                    self.ws.subscribe_bookTicker()
                if f"account.orderUpdate.{self.symbol}" not in self.ws.subscriptions:
                    self.subscribe_order_updates()
            
            while time.time() - start_time < duration_seconds:
                iteration += 1
                current_time = time.time()
                logger.info(f"\n=== 第 {iteration} 次迭代 ===")
                logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 检查连接并在必要时重连
                connection_status = self.check_ws_connection()
                
                # 如果连接成功，检查并确保所有流订阅
                if connection_status:
                    # 重新订阅必要的数据流
                    self._ensure_data_streams()
                
                # 分析市场状况并调整参数
                self.analyze_market_conditions()
                
                # 检查订单成交情况
                self.check_order_fills()
                
                # 检查是否需要重平衡仓位
                if self.need_rebalance():
                    self.rebalance_position()
                
                # 计算买卖价格
                self.place_limit_orders()
                
                # 估算利润
                self.estimate_profit()
                
                # 定期打印交易统计报表
                if current_time - last_report_time >= report_interval:
                    self.print_trading_stats()
                    last_report_time = current_time
                
                # 计算总的PnL和本次执行的PnL
                realized_pnl, unrealized_pnl, total_fees, net_pnl, session_realized_pnl, session_fees, session_net_pnl = self.calculate_pnl()
                
                logger.info("\n统计信息:")
                logger.info(f"总交易次数: {self.trades_executed}")
                logger.info(f"总下单次数: {self.orders_placed}")
                logger.info(f"总取消订单次数: {self.orders_cancelled}")
                logger.info(f"买入总量: {self.total_bought} {self.base_asset}")
                logger.info(f"卖出总量: {self.total_sold} {self.base_asset}")
                logger.info(f"Maker买入: {self.maker_buy_volume} {self.base_asset}, Maker卖出: {self.maker_sell_volume} {self.base_asset}")
                logger.info(f"Taker买入: {self.taker_buy_volume} {self.base_asset}, Taker卖出: {self.taker_sell_volume} {self.base_asset}")
                logger.info(f"总手续费: {total_fees:.8f} {self.quote_asset}")
                logger.info(f"已实现利润: {realized_pnl:.8f} {self.quote_asset}")
                logger.info(f"净利润: {net_pnl:.8f} {self.quote_asset}")
                logger.info(f"未实现利润: {unrealized_pnl:.8f} {self.quote_asset}")
                logger.info(f"WebSocket连接状态: {'已连接' if self.ws and self.ws.is_connected() else '未连接'}")
                
                # 打印本次执行的统计数据
                logger.info("\n---本次执行统计---")
                session_buy_volume = sum(qty for _, qty in self.session_buy_trades)
                session_sell_volume = sum(qty for _, qty in self.session_sell_trades)
                logger.info(f"买入量: {session_buy_volume} {self.base_asset}, 卖出量: {session_sell_volume} {self.base_asset}")
                logger.info(f"Maker买入: {self.session_maker_buy_volume} {self.base_asset}, Maker卖出: {self.session_maker_sell_volume} {self.base_asset}")
                time.sleep(interval_seconds)
        except Exception as e:
            logger.error(f"策略运行出错: {e}")
            raise