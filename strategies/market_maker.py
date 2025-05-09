"""
做市策略模塊
"""
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from api.client import (
    get_balance, execute_order, get_open_orders, cancel_all_orders, 
    cancel_order, get_market_limits, get_ticker, get_order_book, get_klines,
    get_borrow_lend_positions, get_fill_history
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
        base_spread_percentage=0.2,    # 降低基础价差以提高成交率
        order_quantity=0.05,           # 设置固定订单量为0.05 SOL
        max_orders=8,                  # 增加最大订单数到8个
        rebalance_threshold=8.0,      # 降低重平衡阈值到8%
        volatility_threshold=0.02,     # 波动率阈值
        min_profit_multiplier=1.2,     # 降低最小利润倍数到1.2
        max_position_size=30.0,        # 降低最大持仓规模到30%
        aggressive_factor=1.25,        # 提高进取因子来增加成交概率
        max_spread=0.35,               # 降低最大价差提高更新频率
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
        self.max_spread = max_spread
        self.ws_proxy = ws_proxy
        
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
            'base_spread': {'min': 0.17, 'max': 0.4},     # 调整价差范围
            'orders': {'min': 2, 'max': 5},               # 减少最大订单数
            'profit_multiplier': {'min': 1.1, 'max': 1.3}, # 降低利润倍数范围
            'position_size': {'min': 20.0, 'max': 40.0},   # 调整持仓规模范围
            'aggressive': {'min': 1.05, 'max': 1.2}        # 调整进取因子范围
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
        
        spread_pct = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else 0 # Handle potential division by zero
        return spread_pct
    
    def calculate_prices(self):
        """计算买卖订单价格 (重构以防止交叉并确保盈利)"""
        try:
            bid_price, ask_price = self.get_market_depth()
            if bid_price is None or ask_price is None:
                logger.warning("无法获取市场深度，跳过价格计算")
                return None, None

            # 获取市场深度信息和波动率
            depth_score = self._calculate_depth_score()
            volatility = self._calculate_volatility()

            # 计算当前市场价差和中间价
            market_spread = ((ask_price - bid_price) / bid_price) * 100 if bid_price > 0 else 0
            mid_price = (bid_price + ask_price) / 2

            # 计算最低盈利价差百分比
            MAKER_FEE_RATE = 0.0008  # 0.08%
            # 确保min_profit_multiplier有默认值或在初始化时设置
            profit_multiplier = getattr(self, 'min_profit_multiplier', 1.2)
            MIN_PROFITABLE_SPREAD_PCT = MAKER_FEE_RATE * 2 * 100 * profit_multiplier

            # -- 确定目标价差百分比 --
            target_spread_pct = self.base_spread_percentage

            # 根据市场价差调整 (如果市场价差更大，则参考市场价差，但有上限)
            target_spread_pct = max(target_spread_pct, min(market_spread * 1.1, self.max_spread))

            # 根据深度和波动率调整
            if depth_score < 0.3: target_spread_pct *= 1.1  # 深度差，扩大价差
            elif depth_score > 0.7: target_spread_pct *= 0.9 # 深度好，缩小价差
            if volatility > 0.001:
                # 波动率影响因子，可以考虑设置上限避免过度扩大
                volatility_factor = min(1 + volatility * 5, 2.0) # 例如，波动率影响最大翻倍
                target_spread_pct *= volatility_factor

            # 确保目标价差满足最低盈利要求，且不超过最大限制
            target_spread_pct = max(MIN_PROFITABLE_SPREAD_PCT, target_spread_pct)
            target_spread_pct = min(target_spread_pct, self.max_spread)

            # -- 基于中间价和目标价差计算价格 --
            half_spread_amount = (target_spread_pct / 100) * mid_price / 2
            calc_buy_price = mid_price - half_spread_amount
            calc_sell_price = mid_price + half_spread_amount

            # 四舍五入到tick size
            final_buy_price = round_to_tick_size(calc_buy_price, self.tick_size)
            final_sell_price = round_to_tick_size(calc_sell_price, self.tick_size)

            # -- 关键检查：防止价格交叉 --
            while final_sell_price <= final_buy_price:
                logger.warning(f"买卖价计算后交叉或相等 (买: {final_buy_price}, 卖: {final_sell_price})，基于卖价向上调整...")
                # 向上调整卖价一个tick
                final_sell_price = round_to_tick_size(final_sell_price + self.tick_size, self.tick_size)
                # 如果调整后仍然交叉（极不可能，除非tick_size为0或负数），则无法下单
                if final_sell_price <= final_buy_price:
                     logger.error("调整后买卖价仍然交叉，无法安全下单")
                     return None, None

            # -- 关键检查：防止价格穿透市场 --
            if final_buy_price >= ask_price:
                 logger.warning(f"计算的买价 ({final_buy_price}) 高于或等于市场卖价 ({ask_price})，调整买价至市场卖价下方一个tick")
                 final_buy_price = round_to_tick_size(ask_price - self.tick_size, self.tick_size)
                 # 再次检查交叉
                 if final_sell_price <= final_buy_price:
                     logger.error(f"调整买价后导致交叉 (买: {final_buy_price}, 卖: {final_sell_price})，无法下单")
                     return None, None

            if final_sell_price <= bid_price:
                 logger.warning(f"计算的卖价 ({final_sell_price}) 低于或等于市场买价 ({bid_price})，调整卖价至市场买价上方一个tick")
                 final_sell_price = round_to_tick_size(bid_price + self.tick_size, self.tick_size)
                 # 再次检查交叉
                 if final_sell_price <= final_buy_price:
                     logger.error(f"调整卖价后导致交叉 (买: {final_buy_price}, 卖: {final_sell_price})，无法下单")
                     return None, None


            # -- 最终检查：确保实际价差不过大 --
            final_spread_pct = ((final_sell_price - final_buy_price) / final_buy_price) * 100 if final_buy_price > 0 else 0
            # 允许一定的调整空间，例如不超过最大价差的150%
            if final_spread_pct > self.max_spread * 1.5:
                logger.warning(f"最终计算价差 ({final_spread_pct:.4f}%) 过大，超过最大允许价差 ({self.max_spread}%) 的1.5倍，跳过下单")
                return None, None
            elif final_spread_pct < MIN_PROFITABLE_SPREAD_PCT * 0.9: # 也检查价差是否过小
                 logger.warning(f"最终计算价差 ({final_spread_pct:.4f}%) 过小，低于最低盈利价差 ({MIN_PROFITABLE_SPREAD_PCT:.4f}%) 的90%，跳过下单")
                 return None, None


            # 记录最终价格信息
            logger.info(f"市场买价: {bid_price:.4f}, 市场卖价: {ask_price:.4f}, 市场价差: {market_spread:.4f}%")
            logger.info(f"中间价: {mid_price:.4f}")
            logger.info(f"最低盈利价差: {MIN_PROFITABLE_SPREAD_PCT:.4f}%")
            logger.info(f"目标价差: {target_spread_pct:.4f}%")
            logger.info(f"计算买价: {final_buy_price}, 计算卖价: {final_sell_price}")
            logger.info(f"最终价差: {final_spread_pct:.4f}%")
            logger.info(f"深度得分: {depth_score:.4f}, 波动率: {volatility:.4f}")

            return final_buy_price, final_sell_price

        except Exception as e:
            logger.error(f"计算价格时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _calculate_total_balance(self, include_borrow_positions=True):
        """
        计算账户总余额，包括现货和借贷仓位

        Args:
            include_borrow_positions (bool): 是否包含借贷仓位，默认为True

        Returns:
            tuple: 包含以下元素的元组：
                - base_balance (float): 基础货币余额（包括现货和借贷）
                - quote_balance (float): 报价货币余额（包括现货和借贷）
                - total_value_in_quote (float): 总资产价值（以报价货币计算）
                - error_message (str|None): 错误信息，如果没有错误则为None

        Note:
            - 返回的余额包括可用余额和锁定余额
            - 如果include_borrow_positions为True，还会包括借贷仓位
            - 借贷仓位中，借入为正，借出为负
        """
        try:
            # 获取现货余额
            balances = get_balance(self.api_key, self.secret_key)
            if isinstance(balances, dict) and "error" in balances:
                return 0, 0, 0, f"获取余额失败: {balances['error']}"

            # 初始化现货余额
            spot_base_balance = 0
            spot_quote_balance = 0
            for asset, details in balances.items():
                available = float(details.get('available', 0))
                locked = float(details.get('locked', 0))
                total = available + locked
                if asset == self.base_asset:
                    spot_base_balance = total
                elif asset == self.quote_asset:
                    spot_quote_balance = total

            # 初始化借贷调整
            borrow_lend_base_adjustment = 0
            borrow_lend_quote_adjustment = 0
            borrow_positions = []

            # 获取并处理借贷仓位
            if include_borrow_positions:
                borrow_positions = get_borrow_lend_positions(self.api_key, self.secret_key)
                if isinstance(borrow_positions, dict) and "error" in borrow_positions:
                    logger.warning(f"获取借贷仓位失败: {borrow_positions['error']}")
                    # 继续执行，但不包括借贷仓位
                elif isinstance(borrow_positions, list):
                    for position in borrow_positions:
                        position_symbol = position.get('symbol') # 获取仓位对应的资产符号
                        net_quantity_str = position.get('netQuantity') # 获取净数量
                        
                        if position_symbol and net_quantity_str:
                            try:
                                net_quantity = float(net_quantity_str)
                                if position_symbol == self.base_asset:
                                    borrow_lend_base_adjustment = net_quantity
                                    logger.debug(f"借贷调整 - 基础资产 ({self.base_asset}): {net_quantity}")
                                elif position_symbol == self.quote_asset:
                                    borrow_lend_quote_adjustment = net_quantity
                                    logger.debug(f"借贷调整 - 报价资产 ({self.quote_asset}): {net_quantity}")
                            except ValueError as e:
                                logger.error(f"处理借贷仓位数据时出错 ({position_symbol}): {e}")
                                continue # 跳过这个无效的仓位

            # 计算最终余额（现货余额 + 借贷调整）
            base_balance = spot_base_balance + borrow_lend_base_adjustment
            quote_balance = spot_quote_balance + borrow_lend_quote_adjustment

            # 获取当前价格用于计算总价值
            current_price = self.get_current_price()
            if not current_price:
                # 如果无法获取价格，总价值可能不准确，但仍返回计算出的余额
                logger.warning("无法获取当前价格，总资产价值可能不准确")
                total_value_in_quote = (base_balance * 0) + quote_balance # 假设基础资产价值为0
            else:
                # 计算总价值
                total_value_in_quote = (base_balance * current_price) + quote_balance

            logger.info(f"计算后余额 - 基础({self.base_asset}): {base_balance:.8f} (现货: {spot_base_balance:.8f}, 借贷: {borrow_lend_base_adjustment:.8f})")
            logger.info(f"计算后余额 - 报价({self.quote_asset}): {quote_balance:.8f} (现货: {spot_quote_balance:.8f}, 借贷: {borrow_lend_quote_adjustment:.8f})")
            logger.info(f"计算后总价值: {total_value_in_quote:.8f} {self.quote_asset}")

            return base_balance, quote_balance, total_value_in_quote, None

        except Exception as e:
            logger.error(f"计算总余额时出错: {str(e)}")
            import traceback
            traceback.print_exc() # 打印详细的回溯信息
            return 0, 0, 0, f"计算总余额时发生意外错误: {str(e)}"

    def need_rebalance(self):
        """判断是否需要重平衡仓位"""
        # 获取余额信息
        base_balance, quote_balance, total_assets, error = self._calculate_total_balance()
        if error:
            logger.error(error)
            return False
            
        # 获取当前价格
        current_price = self.get_current_price()
        if not current_price:
            logger.warning("无法获取当前价格，取消重平衡")
            return False
        
        # 计算基础货币和报价货币的价值
        base_value = base_balance * current_price
        quote_value = quote_balance
        
        # 计算价值比例
        total_value = base_value + quote_value
        base_ratio = (base_value / total_value) * 100 if total_value > 0 else 0
        quote_ratio = (quote_value / total_value) * 100 if total_value > 0 else 0
        
        # 计算价值偏差
        target_ratio = 50  # 目标是各占50%
        value_deviation = abs(base_ratio - target_ratio)
        
        # 获取当前市场波动率
        volatility = self._calculate_volatility()
        
        # 根据波动率动态调整重平衡阈值
        base_threshold = self.rebalance_threshold  # 基础阈值
        if volatility > 0.02:  # 高波动
            threshold = base_threshold * 0.8  # 降低阈值，更频繁重平衡
        else:
            threshold = base_threshold * 1.2  # 提高阈值，减少重平衡频率
        
        # 检查最近是否有重平衡操作
        current_time = time.time()
        min_rebalance_interval = 300  # 最小重平衡间隔（5分钟）
        
        if hasattr(self, 'last_rebalance_time') and \
           current_time - self.last_rebalance_time < min_rebalance_interval:
            logger.info("距离上次重平衡时间不足5分钟，暂不重平衡")
            return False
        
        # 记录重平衡判断的详细信息
        logger.info(f"基础货币价值: {base_value:.2f} {self.quote_asset} ({base_ratio:.2f}%)")
        logger.info(f"报价货币价值: {quote_value:.2f} {self.quote_asset} ({quote_ratio:.2f}%)")
        logger.info(f"总资产价值: {total_value:.2f} {self.quote_asset}")
        logger.info(f"价值偏差: {value_deviation:.2f}%")
        logger.info(f"当前波动率: {volatility:.4f}")
        logger.info(f"重平衡阈值: {threshold:.2f}%")
        
        # 更新最后重平衡时间
        if value_deviation > threshold:
            self.last_rebalance_time = current_time
            logger.info(f"需要重平衡：价值偏差 {value_deviation:.2f}% > 阈值 {threshold:.2f}%")
            return True
        
        return False

    def place_limit_orders(self):
        """下限价单"""
        try:
            # 1. 先检查WebSocket连接
            self.check_ws_connection()
            
            # 2. 获取当前活跃订单
            open_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
            if isinstance(open_orders, dict) and "error" in open_orders:
                logger.error(f"获取订单失败: {open_orders['error']}")
                return
                
            # REMOVED imbalance check block
            current_buy_orders = [order for order in open_orders if order.get('side') == 'Bid']
            current_sell_orders = [order for order in open_orders if order.get('side') == 'Ask']
            logger.info(f"当前活跃订单检查: {len(current_buy_orders)} 个买单, {len(current_sell_orders)} 个卖单") # Add log for visibility
            
            # 3. 获取最新市场价格并计算下单价格
            buy_price, sell_price = self.calculate_prices()
            if buy_price is None or sell_price is None:
                logger.warning("无法计算合适的订单价格，等待下次尝试")
                return
            
            # 4. 获取余额信息
            base_balance, quote_balance, total_assets, error = self._calculate_total_balance()
            if error:
                logger.error(f"获取余额失败: {error}")
                return
            
            # 5. 计算订单数量
            current_price = (buy_price + sell_price) / 2
            if total_assets <= 0: # Avoid division by zero
                logger.warning("总资产价值为0或负数，无法计算持仓比例")
                position_ratio = 0
            else:
                position_value = base_balance * current_price
                position_ratio = (position_value / total_assets) * 100
            
            # 根据持仓比例调整目标订单数
            # 使用动态获取的 max_orders 值
            current_max_orders = getattr(self, 'max_orders', 5) # Default to 5 if not set
            max_buy_orders = current_max_orders
            max_sell_orders = current_max_orders
            
            if position_ratio > self.max_position_size * 0.8:
                logger.warning(f"持仓比例({position_ratio:.2f}%) > 阈值({self.max_position_size * 0.8:.2f}%), 减少买单目标数量")
                max_buy_orders = max(1, current_max_orders // 2) # Reduce buy target, ensure at least 1 if max_orders > 0
            elif position_ratio < (100 - self.max_position_size * 0.8): # Check if quote ratio is too high (low base ratio)
                logger.warning(f"持仓比例({position_ratio:.2f}%) < 阈值({100 - self.max_position_size * 0.8:.2f}%), 减少卖单目标数量")
                max_sell_orders = max(1, current_max_orders // 2) # Reduce sell target, ensure at least 1
            
            logger.info(f"目标订单数: 买={max_buy_orders}, 卖={max_sell_orders} (基于持仓 {position_ratio:.2f}%) MaxOrders={current_max_orders}")
            
            # 6. 计算每个订单的数量
            base_order_size = round_to_precision(self.order_quantity or 0.05, self.base_precision)
            quote_value_per_order = base_order_size * current_price
            
            # 确保订单金额合理
            min_quote_value = 1.0 # 最小订单金额1 USDC (或从 market_limits 获取)
            while quote_value_per_order < min_quote_value and base_order_size < base_balance / 10: # Add safeguard against huge size increase
                base_order_size *= 1.5 # Increase slightly faster
                base_order_size = round_to_precision(base_order_size, self.base_precision) # Ensure precision
                quote_value_per_order = base_order_size * current_price
                if base_order_size == 0: # Break if rounding leads to zero
                    logger.warning("无法计算出大于0且满足最小金额的订单大小")
                    return
            
            final_order_size = max(base_order_size, self.min_order_size) # Ensure min order size is met
            final_order_size = round_to_precision(final_order_size, self.base_precision)
            logger.info(f"计算出的单笔订单大小: {final_order_size} {self.base_asset}")

            # 检查余额是否足够下至少一个最小订单
            if final_order_size > base_balance and final_order_size * buy_price > quote_balance:
                 logger.warning("基础和报价资产余额均不足以放置最小订单")
                 return
            if final_order_size == 0:
                logger.warning("计算出的最终订单大小为0，无法下单")
                return

            # 7. 下买单
            buy_orders_to_place = max_buy_orders - len(current_buy_orders)
            placed_buy_count = 0
            if buy_orders_to_place > 0:
                logger.info(f"需要放置 {buy_orders_to_place} 个买单 (目标: {max_buy_orders}, 当前: {len(current_buy_orders)}) ")
                for i in range(buy_orders_to_place):
                    # 检查余额是否足够
                    required_quote = final_order_size * buy_price
                    if required_quote * (1 + 0.01 * i) > quote_balance: # Check cumulative balance needed
                        logger.warning(f"报价货币余额不足，无法放置更多买单: 需要 ~{required_quote:.2f}, 剩余 {quote_balance:.2f}")
                        break
                    
                    order_details = {
                        "orderType": "Limit",
                        "price": str(buy_price),
                        "quantity": str(final_order_size),
                        "side": "Bid",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    }
                    
                    result = execute_order(self.api_key, self.secret_key, order_details)
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"买单 {i+1}/{buy_orders_to_place} 执行失败: {result['error']}")
                        # Dont break immediately, maybe next one works? Or maybe break is safer?
                        # break # Let's break for now to be safe
                    # --- CORRECTED SUCCESS CHECK --- 
                    elif isinstance(result, dict) and result.get('id') is not None and 'error' not in result:
                        logger.info(f"买单 {i+1}/{buy_orders_to_place} 已提交: {final_order_size}@{buy_price}, ID: {result.get('id')}, Status: {result.get('status')}")
                        quote_balance -= required_quote # Deduct estimated cost
                        self.orders_placed += 1
                        placed_buy_count += 1
                    else:
                        logger.warning(f"买单 {i+1}/{buy_orders_to_place} 提交结果未知或失败: {result}")
                        # break

                    time.sleep(0.1) # Small delay between orders
            else:
                 logger.info(f"无需放置买单 (目标: {max_buy_orders}, 当前: {len(current_buy_orders)})")

            # 8. 下卖单
            sell_orders_to_place = max_sell_orders - len(current_sell_orders)
            placed_sell_count = 0
            if sell_orders_to_place > 0:
                logger.info(f"需要放置 {sell_orders_to_place} 个卖单 (目标: {max_sell_orders}, 当前: {len(current_sell_orders)}) ")
                for i in range(sell_orders_to_place):
                    # 检查余额是否足够
                    required_base = final_order_size
                    if required_base * (1 + 0.01 * i) > base_balance: # Check cumulative balance needed
                        logger.warning(f"基础货币余额不足，无法放置更多卖单: 需要 ~{required_base:.4f}, 剩余 {base_balance:.4f}")
                        break
                    
                    order_details = {
                        "orderType": "Limit",
                        "price": str(sell_price),
                        "quantity": str(final_order_size),
                        "side": "Ask",
                        "symbol": self.symbol,
                        "timeInForce": "GTC",
                        "postOnly": True
                    }
                    
                    result = execute_order(self.api_key, self.secret_key, order_details)
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"卖单 {i+1}/{sell_orders_to_place} 执行失败: {result['error']}")
                        # break
                    # --- CORRECTED SUCCESS CHECK --- 
                    elif isinstance(result, dict) and result.get('id') is not None and 'error' not in result:
                        logger.info(f"卖单 {i+1}/{sell_orders_to_place} 已提交: {final_order_size}@{sell_price}, ID: {result.get('id')}, Status: {result.get('status')}")
                        base_balance -= required_base # Deduct amount
                        self.orders_placed += 1
                        placed_sell_count += 1
                    else:
                        logger.warning(f"卖单 {i+1}/{sell_orders_to_place} 提交结果未知或失败: {result}")
                        # break
                    
                    time.sleep(0.1)
            else:
                 logger.info(f"无需放置卖单 (目标: {max_sell_orders}, 当前: {len(current_sell_orders)})")

            logger.info(f"本轮下单完成: 放置了 {placed_buy_count} 个买单, {placed_sell_count} 个卖单")

        except Exception as e:
            logger.error(f"下单过程发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
                
            if not klines or not isinstance(klines, list):
                logger.warning("K线数据无效或为空")
                return 0.0
                
            try:
                # 检查K线数据格式
                if isinstance(klines[0], dict):
                    # 如果是字典格式，获取收盘价
                    if 'close' in klines[0]:
                        closes = [float(k['close']) for k in klines]
                    else:
                        logger.error("K线数据中缺少close字段")
                        return 0.0
                else:
                    logger.error("K线数据格式不正确")
                    return 0.0
                
                if not closes:
                    logger.warning("无有效收盘价数据")
                    return 0.0
                    
                std_dev = np.std(closes)
                mean_price = np.mean(closes)
                
                # 返回波动率（标准差/均价）
                volatility = (std_dev / mean_price) if mean_price > 0 else 0.0
                logger.info(f"当前波动率: {volatility:.4f}")
                return volatility
            except (IndexError, ValueError) as e:
                logger.error(f"处理K线数据时出错: {str(e)}")
                return 0.0
                
        except Exception as e:
            logger.exception(f"计算波动率出错: {str(e)}")
            return 0.0
            
    def _calculate_trend(self):
        """计算市场趋势"""
        try:
            # 使用client.py中的get_klines方法
            klines = get_klines(self.symbol, interval='5m', limit=12)
            if isinstance(klines, dict) and "error" in klines:
                logger.error(f"获取K线数据失败: {klines['error']}")
                return 0.0
                
            if not klines or not isinstance(klines, list):
                logger.warning("K线数据无效或为空")
                return 0.0
                
            try:
                # 检查K线数据格式
                if isinstance(klines[0], dict):
                    # 如果是字典格式，获取收盘价
                    if 'close' in klines[0]:
                        closes = [float(k['close']) for k in klines]
                    else:
                        logger.error("K线数据中缺少close字段")
                        return 0.0
                else:
                    logger.error("K线数据格式不正确")
                    return 0.0
                
                if len(closes) < 12:
                    logger.warning("K线数据不足")
                    return 0.0
                    
                sma_5 = np.mean(closes[-5:])
                sma_12 = np.mean(closes)
                
                # 计算趋势强度 (-1到1之间)
                trend = (sma_5 - sma_12) / sma_12 if sma_12 > 0 else 0.0
                trend = max(min(trend, 1.0), -1.0)
                
                logger.info(f"当前趋势: {trend:.4f}")
                return trend
            except (IndexError, ValueError) as e:
                logger.error(f"处理K线数据时出错: {str(e)}")
                return 0.0
                
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
                
            if not klines or not isinstance(klines, list):
                logger.warning("K线数据无效或为空")
                return 0.0
                
            try:
                # 检查K线数据格式
                if isinstance(klines[0], dict):
                    # 如果是字典格式，获取成交量
                    if 'volume' in klines[0]:
                        volumes = [float(k['volume']) for k in klines]
                    else:
                        logger.error("K线数据中缺少volume字段")
                        return 0.0
                else:
                    logger.error("K线数据格式不正确")
                    return 0.0
                
                if not volumes:
                    logger.warning("无有效成交量数据")
                    return 0.0
                    
                avg_volume = np.mean(volumes)
                max_volume = max(volumes)
                
                # 返回成交量水平 (0到1之间)
                volume_level = avg_volume / max_volume if max_volume > 0 else 0.0
                logger.info(f"当前成交量水平: {volume_level:.4f}")
                return volume_level
            except (IndexError, ValueError) as e:
                logger.error(f"处理成交量数据时出错: {str(e)}")
                return 0.0
                
        except Exception as e:
            logger.error(f"计算成交量水平出错: {str(e)}")
            return 0.0
            
    def _calculate_spread_level(self):
        """计算当前点差水平"""
        try:
            if not self.ws.orderbook:
                logger.warning("订单簿数据不可用")
                return 0.0
                
            bids = self.ws.orderbook.get('bids', [])
            asks = self.ws.orderbook.get('asks', [])
            
            if not bids or not asks:
                logger.warning("买卖盘数据不完整")
                return 0.0
                
            try:
                # 计算当前点差
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                # 计算点差比率
                spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0
                
                # 返回标准化的点差水平 (0到1之间)
                spread_level = min(spread / (self.max_spread / 100), 1.0)
                logger.info(f"当前点差水平: {spread_level:.4f}")
                return spread_level
                
            except (IndexError, ValueError, ZeroDivisionError) as e:
                logger.error(f"处理订单簿数据时出错: {str(e)}")
                return 0.0
                
        except Exception as e:
            logger.error(f"计算点差水平出错: {str(e)}")
            return 0.0
            
    def _adjust_parameters(self):
        """根据市场状态调整参数"""
        try:
            if not all(key in self.market_state for key in ['depth_score', 'volume_level', 'volatility', 'spread_level']):
                logger.warning("市场状态数据不完整")
                return
                
            # 计算市场得分
            market_score = (
                self.market_state['depth_score'] * 0.3 +
                self.market_state['volume_level'] * 0.3 +
                (1 - self.market_state['volatility']) * 0.2 +
                (1 - self.market_state['spread_level']) * 0.2
            )
            
            # 确保得分在0-1之间
            market_score = max(0.0, min(1.0, market_score))
            
            # 根据市场得分调整参数
            self.base_spread_percentage = self._interpolate(
                self.param_ranges['base_spread']['min'],
                self.param_ranges['base_spread']['max'],
                market_score
            )
            
            self.max_orders = int(self._interpolate(
                self.param_ranges['orders']['min'],
                self.param_ranges['orders']['max'],
                market_score
            ))
            
            self.min_profit_multiplier = self._interpolate(
                self.param_ranges['profit_multiplier']['min'],
                self.param_ranges['profit_multiplier']['max'],
                market_score
            )
            
            self.max_position_size = self._interpolate(
                self.param_ranges['position_size']['min'],
                self.param_ranges['position_size']['max'],
                market_score
            )
            
            # 特殊情况处理
            if self.market_state['volatility'] > 0.05:  # 高波动
                self.base_spread_percentage *= 1.5
                self.max_orders = max(2, self.max_orders // 2)
                
            if abs(self.market_state['trend']) > 0.8:  # 明显趋势
                trend_direction = 1 if self.market_state['trend'] > 0 else -1
                self.aggressive_factor = self._interpolate(
                    self.param_ranges['aggressive']['min'],
                    self.param_ranges['aggressive']['max'],
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

    def run(self, duration_seconds=3600, interval_seconds=30):  # 缩短默认间隔到30秒
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
        last_order_check_time = start_time
        last_market_analysis_time = start_time
        report_interval = 300  # 5分鐘打印一次報表
        order_check_interval = 10  # 10秒检查一次订单
        market_analysis_interval = 60  # 1分钟分析一次市场
        
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
                
                # 分析市场状况并调整参数（每分钟）
                if current_time - last_market_analysis_time >= market_analysis_interval:
                    self.analyze_market_conditions()
                    last_market_analysis_time = current_time
                
                # 更频繁地检查订单成交情况（每10秒）
                if current_time - last_order_check_time >= order_check_interval:
                    self.check_order_fills()
                    last_order_check_time = current_time
                
                # 检查是否需要重平衡仓位
                if self.need_rebalance():
                    self.rebalance_position()
                    # 在重平衡后跳过本次迭代的下单逻辑，给重平衡订单时间
                    continue 
                
                # 计算买卖价格并下单
                self.place_limit_orders()
                
                # 估算利润
                self.estimate_profit()
                
                # 定期打印交易统计报表（每5分钟）
                if current_time - last_report_time >= report_interval:
                    self.print_trading_stats()
                    last_report_time = current_time
                
                # 计算总的PnL和本次执行的PnL
                realized_pnl, unrealized_pnl, total_fees, net_pnl, session_realized_pnl, session_fees, session_net_pnl = self.calculate_pnl()
                
                # 只在每5次迭代时打印详细统计信息
                if iteration % 5 == 0:
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
                
                # 等待下一次迭代
                time.sleep(interval_seconds)
                
        except Exception as e:
            logger.error(f"策略运行出错: {e}")
            raise
        finally:
            # Ensure cleanup runs when duration expires or loop breaks/errors
            logger.info(f"策略運行循環結束 (迭代次數: {iteration})，執行清理...")
            self.cleanup()
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

    def rebalance_position(self):
        """重平衡仓位"""
        logger.info("开始重新平衡仓位...")
        self.check_ws_connection()
        
        # 获取当前余额和价格信息
        base_balance, quote_balance, total_assets, error = self._calculate_total_balance()
        if error:
            logger.error(f"获取余额失败: {error}")
            return
            
        current_price = self.get_current_price()
        if current_price is None:
            logger.error("无法获取当前价格，取消重平衡")
            return
            
        # 计算当前价值分布
        base_value = base_balance * current_price
        quote_value = quote_balance
        total_value = base_value + quote_value
        
        # 计算目标价值
        target_value = total_value / 2
        
        # 计算需要调整的价值
        if base_value > target_value:  # 需要卖出基础货币
            value_to_adjust = base_value - target_value
            quantity = round_to_precision(value_to_adjust / current_price, self.base_precision)
            is_sell = True
        else:  # 需要买入基础货币
            value_to_adjust = target_value - base_value
            quantity = round_to_precision(value_to_adjust / current_price, self.base_precision)
            is_sell = False
        
        # 考虑手续费，留出5%缓冲
        quantity = quantity * 0.95
        
        if quantity < self.min_order_size:
            logger.info(f"调整数量 {quantity} 低于最小订单大小 {self.min_order_size}，不进行重平衡")
            return
            
            
        # 获取市场深度
        bid_price, ask_price = self.get_market_depth()
        if bid_price is None or ask_price is None:
            logger.error("无法获取市场深度，取消重平衡")
            return
            
        # 计算更保守的批次大小
        num_batches = 5  # 分5次执行
        batch_size = round_to_precision(quantity / num_batches, self.base_precision)
        if batch_size < self.min_order_size:
            num_batches = max(1, int(quantity / self.min_order_size))
            batch_size = round_to_precision(quantity / num_batches, self.base_precision)
        
        logger.info(f"重平衡总量: {quantity} {self.base_asset}, 分 {num_batches} 批执行，每批 {batch_size}")
        
        # 执行重平衡
        if is_sell:  # 卖出
            sell_price = round_to_tick_size(bid_price * 1.0005, self.tick_size)  # 更积极的定价
            logger.info(f"执行重平衡: 卖出 {quantity} {self.base_asset} @ {sell_price}")
            
            remaining_quantity = quantity
            success_count = 0
            
            while remaining_quantity >= self.min_order_size and success_count < num_batches:
                current_batch = min(batch_size, remaining_quantity)
                current_batch = round_to_precision(current_batch, self.base_precision)
                
                # 动态调整价格
                new_bid_price, _ = self.get_market_depth()
                if new_bid_price:
                    sell_price = round_to_tick_size(new_bid_price * 1.0005, self.tick_size)
                
                order_details = {
                    "orderType": "Limit",
                    "price": str(sell_price),
                    "quantity": str(current_batch),
                    "side": "Ask",
                    "symbol": self.symbol,
                    "timeInForce": "GTC",
                    "postOnly": True
                }
                
                result = execute_order(self.api_key, self.secret_key, order_details)
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"重平衡卖单执行失败: {result['error']}")
                    break
                else:
                    logger.info(f"重平衡卖单已提交: 数量={current_batch}, 价格={sell_price}")
                    if 'id' in result:
                        self.db.record_rebalance_order(result['id'], self.symbol)
                    remaining_quantity -= current_batch
                    success_count += 1
                
                time.sleep(1)
                
        else:  # 买入
            buy_price = round_to_tick_size(ask_price * 0.9995, self.tick_size)  # 更积极的定价
            logger.info(f"执行重平衡: 买入 {quantity} {self.base_asset} @ {buy_price}")
            
            remaining_quantity = quantity
            success_count = 0
            
            while remaining_quantity >= self.min_order_size and success_count < num_batches:
                current_batch = min(batch_size, remaining_quantity)
                current_batch = round_to_precision(current_batch, self.base_precision)
                
                # 检查当前批次所需资金
                required_funds = current_batch * buy_price
                if required_funds > quote_balance * 0.95:  # 留5%余量
                    logger.warning(f"当前可用资金不足，调整批次大小")
                    current_batch = round_to_precision(quote_balance * 0.95 / buy_price, self.base_precision)
                    if current_batch < self.min_order_size:
                        logger.error("可用资金不足以执行最小订单，停止重平衡")
                        break
                
                # 动态调整价格
                _, new_ask_price = self.get_market_depth()
                if new_ask_price:
                    buy_price = round_to_tick_size(new_ask_price * 0.9995, self.tick_size)
                
                order_details = {
                    "orderType": "Limit",
                    "price": str(buy_price),
                    "quantity": str(current_batch),
                    "side": "Bid",
                    "symbol": self.symbol,
                    "timeInForce": "GTC",
                    "postOnly": True
                }
                
                result = execute_order(self.api_key, self.secret_key, order_details)
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"重平衡买单执行失败: {result['error']}")
                    break
                else:
                    logger.info(f"重平衡买单已提交: 数量={current_batch}, 价格={buy_price}")
                    if 'id' in result:
                        self.db.record_rebalance_order(result['id'], self.symbol)
                    remaining_quantity -= current_batch
                    success_count += 1
                    quote_balance -= required_funds
                
                time.sleep(1)
        
        logger.info(f"仓位重新平衡完成，成功执行 {success_count}/{num_batches} 批次")


    def cleanup(self):
        """清理資源：取消訂單並關閉WebSocket"""
        logger.info("開始執行清理程序...")
        try:
            # 取消所有现有订单
            logger.info("正在取消所有未完成订单...")
            self.cancel_existing_orders() # Use the existing method

            # 等待确认所有订单已取消
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Directly use get_open_orders from api.client
                    open_orders = get_open_orders(self.api_key, self.secret_key, self.symbol)
                    if isinstance(open_orders, dict) and "error" in open_orders:
                         logger.error(f"检查未结订单时出错: {open_orders['error']}")
                         # Break if error occurs during check
                         break
                    elif not open_orders or len(open_orders) == 0:
                        logger.info("所有订单已成功取消")
                        break
                    logger.warning(f"仍有 {len(open_orders)} 个未取消的订单，重试中...")
                    self.cancel_existing_orders() # Retry cancelling
                    retry_count += 1
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"检查或取消订单时发生异常: {e}")
                    break # Break on exception during check/cancel

            # 关闭WebSocket连接
            if hasattr(self, 'ws') and self.ws:
                self.ws.close()
                logger.info("WebSocket连接已关闭")

            # 打印最终统计信息
            self.print_trading_stats()

        except Exception as e:
            logger.error(f"清理过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("清理程序执行完毕。")