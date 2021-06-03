import datetime
import traceback
from enum import Enum
from typing import Optional, List, Union, Tuple, Type

import numpy as np

from .. import WebsocketsClient, RestClient
from ... import Template
from ....Res.ToolKit import BarData, TickData, TradeData, OrderBook, TradeInstruction, TradeSide, TradeReport, OrderState, OrderType

__all__ = ['HuobiOrderType', 'Market', 'Trade']


class HuobiOrderType(Enum):
    Buy_Market = 'buy-market'
    Buy_Limit = 'buy-limit'
    Buy_IOC = 'buy-ioc'
    Buy_Making = 'buy-limit-maker'

    Sell_Market = 'sell-market'
    Sell_Limit = 'sell-limit'
    Sell_IOC = 'sell-ioc'
    Sell_Making = 'sell-limit-maker'


class Market(Template.Market):
    def __init__(
            self,
            market_url,
            event_engine,
            logger,
            http_proxy: str = None,
            tick_level: str = 'step0',
            bar_span: datetime.timedelta = datetime.timedelta(seconds=60),
            tick_span: Optional[float] = None,
    ):
        super(Market, self).__init__(
            event_engine=event_engine,
            logger=logger,
        )

        self.market_client = WebsocketsClient.HuobiWebsocketsClient(
            name='Huobi.Spot.Market.Websockets.Client',
            url=market_url,
            proxy=http_proxy
        )

        self.market_level = tick_level
        self.synthetic_bar_span = bar_span
        self.tick_span = tick_span

        self.market_client.connect()

        self.last_bar = {}
        self.last_tick = {}

    def _parse_bar(self, message):
        topic: str = message['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]

        pre_bar: BarData = self.last_bar[ticker]['pre']['bar']
        active_bar: BarData = self.last_bar[ticker]['active']['bar']
        future_bar: BarData = self.last_bar[ticker]['next']['bar']

        if channel == 'trade':
            market_time = datetime.datetime.utcfromtimestamp(message['tick']['ts'] / 1000)
            self.last_bar[ticker]['market_time'] = market_time
            to_update = None
            side = 0
            for trade_data_dict in message['tick']['data']:
                trade_data = TradeData(
                    ticker=ticker,
                    trade_time=market_time,
                    trade_price=trade_data_dict['price'],
                    trade_volume=trade_data_dict['amount'],
                    side=trade_data_dict['direction']
                )
                side = trade_data.side.offset
                if active_bar is not None:
                    if market_time < active_bar.bar_start_time:
                        to_update = pre_bar
                    elif active_bar.bar_start_time <= market_time < active_bar.bar_start_time + self.synthetic_bar_span:
                        to_update = active_bar
                    elif active_bar.bar_start_time + self.synthetic_bar_span <= market_time:
                        to_update = future_bar
                    else:
                        continue

                    if np.isnan(to_update.open_price):
                        to_update.open_price = trade_data.price

                    to_update.close_price = trade_data.price
                    to_update.high_price = np.nanmax([trade_data.price, to_update.high_price])
                    to_update.low_price = np.nanmin([trade_data.price, to_update.low_price])

                    to_update.volume += trade_data.volume
                    to_update.notional += trade_data.notional

                    # noinspection PyUnresolvedReferences
                    to_update.net_notional_flow += side * trade_data.notional
                    # noinspection PyUnresolvedReferences
                    to_update.net_volume_flow += side * trade_data.volume
                    # to_update.net_trade_flow += trade_data.side.sign
                else:
                    continue

            if to_update is not None:
                to_update.trade_count += 1
                # noinspection PyUnresolvedReferences
                to_update.net_trade_flow += side

        elif channel == 'detail':
            market_time = datetime.datetime.utcfromtimestamp(message['ts'] / 1000)
            total_trade = message['tick']['count']
            total_volume = message['tick']['amount']
            total_notional = message['tick']['vol']
            if market_time < active_bar.bar_start_time:
                self.last_bar[ticker]['pre']['total_trade'] = max(total_trade, self.last_bar[ticker]['pre']['total_trade'])
                self.last_bar[ticker]['pre']['total_volume'] = max(total_volume, self.last_bar[ticker]['pre']['total_volume'])
                self.last_bar[ticker]['pre']['total_notional'] = max(total_notional, self.last_bar[ticker]['pre']['total_notional'])
                # active_bar.volume = self.last_bar[ticker]['active']['total_volume'] - self.last_bar[ticker]['pre']['total_volume']
                # active_bar.notional = self.last_bar[ticker]['active']['total_notional'] - self.last_bar[ticker]['pre']['total_notional']
                # active_bar.trade_count = self.last_bar[ticker]['active']['total_trade'] - self.last_bar[ticker]['pre']['total_trade']
            elif active_bar.bar_start_time <= market_time < active_bar.bar_start_time + self.synthetic_bar_span:
                self.last_bar[ticker]['active']['total_trade'] = max(total_trade, self.last_bar[ticker]['active']['total_trade'])
                self.last_bar[ticker]['active']['total_volume'] = max(total_volume, self.last_bar[ticker]['active']['total_volume'])
                self.last_bar[ticker]['active']['total_notional'] = max(total_notional, self.last_bar[ticker]['active']['total_notional'])
                active_bar.volume = self.last_bar[ticker]['active']['total_volume'] - self.last_bar[ticker]['pre']['total_volume']
                active_bar.notional = self.last_bar[ticker]['active']['total_notional'] - self.last_bar[ticker]['pre']['total_notional']
                active_bar.trade_count = self.last_bar[ticker]['active']['total_trade'] - self.last_bar[ticker]['pre']['total_trade']
            elif active_bar.bar_start_time + self.synthetic_bar_span <= market_time:
                self.last_bar[ticker]['next']['total_trade'] = max(total_trade, self.last_bar[ticker]['next']['total_trade'])
                self.last_bar[ticker]['next']['total_volume'] = max(total_volume, self.last_bar[ticker]['next']['total_volume'])
                self.last_bar[ticker]['next']['total_notional'] = max(total_notional, self.last_bar[ticker]['next']['total_notional'])

    def _parse_tick(self, message):
        topic: str = message['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]
        market_time = datetime.datetime.utcfromtimestamp(message['ts'] / 1000)
        timestamp = message['tick']['ts'] / 1000

        if channel == 'detail':
            if 'market_time' not in self.last_tick[ticker]['detail'] or self.last_tick[ticker]['detail']['market_time'] < market_time:
                self.last_tick[ticker]['detail'] = message['tick']
                self.last_tick[ticker]['detail']['market_time'] = market_time
        elif channel == 'depth':
            order_book = OrderBook(
                ticker=ticker,
                market_time=market_time,
                timestamp=timestamp
            )

            for entry in message['tick']['bids']:
                order_book.bid.add(price=entry[0], volume=entry[1])

            for entry in message['tick']['asks']:
                order_book.ask.add(price=entry[0], volume=entry[1])

            if 'order_book' not in self.last_tick[ticker]['depth'] or self.last_tick[ticker]['depth']['order_book'].market_time < market_time:
                self.last_tick[ticker]['depth']['order_book'] = order_book

        if 'close' in self.last_tick[ticker]['detail'] and 'order_book' in self.last_tick[ticker]['depth']:
            tick_data = TickData(
                ticker=ticker,
                market_time=market_time,
                last_price=self.last_tick[ticker]['detail']['close'],
                order_book=self.last_tick[ticker]['depth']['order_book'],
                # total_traded_volume=self.last_tick[ticker]['detail']['amount'],
                # total_traded_notional=self.last_tick[ticker]['detail']['vol'],
                # total_trade_count=self.last_tick[ticker]['detail']['count']
            )

            if self.tick_span is None or self.tick_span <= 0:
                self._on_market(market_data=tick_data)
            else:
                self.last_tick[ticker]['tick'] = tick_data

    def _parse_trade(self, message):
        topic: str = message['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]
        market_time = datetime.datetime.utcfromtimestamp(message['tick']['ts'] / 1000)
        timestamp = message['tick']['ts'] / 1000

        if channel == 'trade':
            for trade_data_dict in message['tick']['data']:
                trade_data = TradeData(
                    ticker=ticker,
                    trade_time=market_time,
                    timestamp=timestamp,
                    trade_price=trade_data_dict['price'],
                    trade_volume=trade_data_dict['amount'],
                    side=trade_data_dict['direction']
                )

                # self.last_trade[ticker]['trade'].put(trade_data, block=False)
                self._on_market(market_data=trade_data)

    def _parse_order_book(self, message):
        topic: str = message['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]
        market_time = datetime.datetime.utcfromtimestamp(message['tick']['ts'] / 1000)
        timestamp = message['tick']['ts'] / 1000

        if channel == 'depth':
            order_book = OrderBook(
                ticker=ticker,
                market_time=market_time,
                timestamp=timestamp
            )

            for entry in message['tick']['bids']:
                order_book.bid.add(price=entry[0], volume=entry[1])

            for entry in message['tick']['asks']:
                order_book.ask.add(price=entry[0], volume=entry[1])

            self._on_market(market_data=order_book)

    def _bar_thread(self, ticker: str):
        self.last_bar[ticker] = {}

        t = datetime.datetime.utcnow()
        t -= datetime.timedelta(seconds=t.second % self.synthetic_bar_span.seconds, microseconds=t.microsecond)
        bar_start_time = t + self.synthetic_bar_span

        pre_bar = BarData(ticker=ticker, bar_start_time=bar_start_time - self.synthetic_bar_span, bar_span=self.synthetic_bar_span)
        active_bar = BarData(ticker=ticker, bar_start_time=bar_start_time, bar_span=self.synthetic_bar_span)
        next_bar = BarData(ticker=ticker, bar_start_time=bar_start_time + self.synthetic_bar_span, bar_span=self.synthetic_bar_span)

        pre_bar.net_notional_flow = 0.
        pre_bar.net_volume_flow = 0.
        pre_bar.net_trade_flow = 0

        active_bar.net_notional_flow = 0.
        active_bar.net_volume_flow = 0.
        active_bar.net_trade_flow = 0

        next_bar.net_notional_flow = 0.
        next_bar.net_volume_flow = 0.
        next_bar.net_trade_flow = 0

        self.last_bar[ticker]['pre'] = {'total_volume': 0, 'total_notional': 0, 'total_trade': 0, 'bar': pre_bar}
        self.last_bar[ticker]['active'] = {'total_volume': 0, 'total_notional': 0, 'total_trade': 0, 'bar': active_bar}
        self.last_bar[ticker]['next'] = {'total_volume': 0, 'total_notional': 0, 'total_trade': 0, 'bar': next_bar}

        def push_bar(**_):
            active_bar = self.last_bar[ticker]['active']['bar']
            market_time = datetime.datetime.utcnow()
            error_range = datetime.timedelta(seconds=1)

            while active_bar.bar_start_time + self.synthetic_bar_span - error_range <= market_time <= active_bar.bar_start_time + self.synthetic_bar_span + error_range:
                self._on_market(market_data=active_bar)
                active_bar = self.last_bar[ticker]['next']['bar']
                next_bar = BarData(
                    ticker=ticker,
                    bar_start_time=active_bar.bar_start_time + self.synthetic_bar_span,
                    bar_span=self.synthetic_bar_span
                )

                next_bar.net_notional_flow = 0.
                next_bar.net_volume_flow = 0.
                next_bar.net_trade_flow = 0

                self.last_bar[ticker]['pre'] = self.last_bar[ticker]['active']
                self.last_bar[ticker]['active'] = self.last_bar[ticker]['next']
                self.last_bar[ticker]['next'] = {'total_volume': 0, 'total_notional': 0, 'total_trade': 0, 'bar': next_bar}

        self.market_client.subscribe(topic=f'market.{ticker}.trade.detail', callback=self._parse_bar)
        self.set_timer(interval=self.synthetic_bar_span, callback=push_bar)

    def _tick_thread(self, ticker: str):
        self.last_tick[ticker] = {'tick': None, 'detail': {}, 'depth': {}}

        self.market_client.subscribe(topic=f'market.{ticker}.detail', callback=self._parse_tick)
        self.market_client.subscribe(topic=f'market.{ticker}.depth.{self.market_level}', callback=self._parse_tick)

        if self.tick_span is not None and self.tick_span > 0:
            t = datetime.datetime.utcnow()
            t -= datetime.timedelta(microseconds=t.microsecond)

            def push_tick(**_):
                tick_data = self.last_tick[ticker]['tick']
                if tick_data is not None:
                    self._on_market(market_data=tick_data)
                    self.last_tick[ticker]['tick'] = None

            self.set_timer(interval=self.tick_span, callback=push_tick)

    def _trade_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'market.{ticker}.trade.detail', callback=self._parse_trade)

    def _order_book_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'market.{ticker}.depth.{self.market_level}', callback=self._parse_order_book)

    def _on_subscribe(self, ticker: str, dtype: str, **kwargs):
        if dtype == 'BarData':
            self._bar_thread(ticker=ticker)
        elif dtype == 'TickData':
            self._tick_thread(ticker=ticker)
        elif dtype == 'TradeData':
            self._trade_thread(ticker=ticker)
        elif dtype == 'OrderBook':
            self._order_book_thread(ticker=ticker)
        else:
            self.logger.warning(f'Invalid dtype {dtype}')


class Trade(Template.Trade):
    def __init__(
            self,
            event_engine,
            trade_url: str,
            rest_url: str,
            access_key: str,
            secret_key: str,
            mode: str = 'spot',  # or 'spot' for spot trade, 'margin' for margin trade
            http_proxy: str = None,
            logger=None,
    ):
        super().__init__(event_engine=event_engine, logger=logger)

        self.websockets_client = WebsocketsClient.HuobiWebsocketsClient(
            name='Huobi.Spot.Trade.Websockets.Client',
            access_key=access_key,
            secret_key=secret_key,
            url=trade_url,
            event_engine=event_engine,
            proxy=http_proxy
        )

        self.rest_client = RestClient.HuobiRestClient(
            name='Huobi.Spot.Trade.REST.Client',
            access_key=access_key,
            secret_key=secret_key,
            url=rest_url,
            proxy=http_proxy
        )

        self.websockets_client.on_open_jobs.append(self._on_ws_connect)
        self.websockets_client.on_reconnect_jobs.append(self._on_ws_connect)
        self.websockets_client.connect()

        if mode == 'spot':
            self.account_id = self._get_account_id()['spot']
            self.source = 'api'
        elif mode == 'margin':
            self.account_id = self._get_account_id()['margin']
            self.source = 'margin-api'

        self._position = Template.Position()

        self.query_order()

    def _on_ws_connect(self, websockets_client):
        websockets_client.auth(version='2.1')

    def _get_account_id(self):
        result = self.rest_client.request_signed(request='get', url=self.rest_client.url + "/v1/account/accounts")
        id_dict = {}

        if result['status'] == 'ok':
            for account_id in result['data']:
                if account_id['state'] != 'working':
                    self.logger.warning(f'{account_id["type"]} account state is {account_id["state"]}')

                id_dict[account_id['type']] = account_id['id']

        return id_dict

    def launch_order(self, order: TradeInstruction, **kwargs) -> str:
        """
        launch a trade order to huobi
        :param order: the given trade instruction
               the order must have following additional attributes:
               huobi_order_type: HuobiOrderType
               huobi_order_id: must set as None and being mutable
        :return: order id
        """
        self.websockets_client.subscribe(message={"action": "sub", "ch": f'orders#{order.Ticker}'}, callback=self._update_order)
        self.websockets_client.subscribe(message={"action": "sub", "ch": f'trade.clearing#{order.Ticker}#1'}, callback=self._update_trade)
        self._position.log_order(order=order)

        try:
            huobi_order_type = getattr(order, 'huobi_order_type')
        except AttributeError:
            huobi_order_type = self._to_HuobiOrderType(order_type=order.OrderType, trade_side=order.Side)
            self.logger.warning(f'Danger action! No huobi_order_type assigned to trade instruction! Using {huobi_order_type} as implied')

        params = {
            "account-id": self.account_id,
            "amount": order.Volume,
            "symbol": order.Ticker,
            "type": huobi_order_type.value,
            "source": self.source,
            # "client-order-id": trade_instruction.OrderID
        }
        if order.LimitPrice is not None:
            params["price"] = order.LimitPrice
        # noinspection PyBroadException
        try:
            self.logger.debug(f'Sending order {order}')
            result = self.rest_client.request_signed(request='post', url=self.rest_client.url + '/v1/order/orders/place', params=params)

            if result['status'] == 'ok':
                huobi_order_id = str(result['data'])
                order.huobi_order_id = huobi_order_id
                self.logger.debug(f'Order {order.OrderID} sent successful!')
                order.set_order_state(OrderState.Pending, market_datetime=datetime.datetime.utcnow())
                self._position.reset_order_id(from_id=order.OrderID, to_id=huobi_order_id)
                order.reset_order_id(order_id=huobi_order_id)
            else:
                self.logger.warning(f'Order {order.OrderID} sent failed! err-code: "{result["err-code"]}", err-msg: "{result["err-msg"]}"')
                order.set_order_state(OrderState.Rejected)
                self._on_error(order=order, **result)
        except Exception as _:
            self.logger.warning(traceback.format_exc())
            order.set_order_state(OrderState.UNKNOWN)
            self._on_error(order=order, traceback=traceback.format_exc())

        return order.OrderID

    def cancel_order(self, order_id: str, **kwargs):
        """
        post a cancel order instruction
        :param order_id: the given order id
        """
        # noinspection PyBroadException
        try:
            self.logger.debug(f'Canceling order {order_id}')
            # noinspection SpellCheckingInspection
            result = self.rest_client.request_signed(request='post', url=self.rest_client.url + f"/v1/order/orders/{order_id}/submitcancel")

            if result['status'] == 'ok':
                order_state = OrderState.Canceling
            else:
                self.logger.warning(f'Order {order_id} canceling failed! err-code: "{result["err-code"]}", err-msg: "{result["err-msg"]}"')
                order_state = OrderState.UNKNOWN
                self._on_error(order_id=order_id, **result)
        except Exception as _:
            self.logger.warning(traceback.format_exc())
            order_state = OrderState.UNKNOWN
            self._on_error(order_id=order_id, traceback=traceback.format_exc())

        logged_instruction = self._position.order.get(order_id)
        if logged_instruction is None:
            self.logger.warning(f'Order {order_id} not found in position')
        else:
            logged_instruction.set_order_state(order_state)

    def _update_order(self, msg: dict):
        data_dict = msg['data']
        order_id = str(data_dict['orderId'])
        event = data_dict['eventType']

        if order_id in self._position.order:
            order = self._position.order[order_id]
        else:
            order = None
            self.logger.warning(f'Order {order_id} not found! Perhaps it\'s not created by this client?')

        if event == 'trigger':
            raise NotImplementedError()
        elif event == 'deletion':
            raise NotImplementedError()
        # order listed
        elif event == 'creation':
            create_time = datetime.datetime.utcfromtimestamp(data_dict['orderCreateTime'] / 1000)
            if order:
                order.set_order_state(order_state=OrderState.Placed, market_datetime=create_time)
                self._on_order(order=order)
            else:
                order_type, order_side = self._from_HuobiOrderType(data_dict['type'])
                order = TradeInstruction(
                    ticker=data_dict['symbol'],
                    side=order_side,
                    order_type=order_type,
                    volume=float(data_dict['orderSize']),
                    limit_price=float(data_dict['orderPrice']),
                    order_id=order_id
                )

                self._position.log_order(order=order)
        # order traded, handled by _update_trade
        elif event == 'trade':
            pass
        # order canceled, handled by _update_trade
        elif event == 'cancellation':
            pass
        else:
            raise ValueError(f'Unregistered event {event}: {msg}')

    def _update_trade(self, msg: dict):
        data_dict = msg['data']
        order_id = str(data_dict['orderId'])
        event = data_dict['eventType']

        if order_id in self._position.order:
            order = self._position.order[order_id]
        else:
            order = None
            self.logger.warning(f'Order {order_id} not found! Perhaps it\'s not created by this client?')

        if event == 'trade':
            trade_price = float(data_dict['tradePrice'])
            trade_volume = float(data_dict['tradeVolume'])
            trade_id = str(data_dict['tradeId'])
            trade_time = datetime.datetime.utcfromtimestamp(data_dict['tradeTime'] / 1000)
            trade_fee = float(data_dict['transactFee'])

            report = TradeReport(
                ticker=data_dict['symbol'],
                side=TradeSide.__call__(data_dict['orderSide']),
                volume=trade_volume,
                notional=trade_volume * trade_price,
                trade_time=trade_time,
                order_id=order_id,
                trade_id=trade_id,
                fee=trade_fee
            )

            if order:
                self._position.log_report(report=report)
                self._on_order(order=order)

            self._position.log_report(report=report)
            self._on_report(report=report)
        elif event == 'cancellation':
            cancel_time = datetime.datetime.utcfromtimestamp(data_dict['orderCreateTime'] / 1000)
            remaining_amount = float(data_dict['remainAmt'])

            if order:
                order.set_order_state(order_state=OrderState.Canceled, market_datetime=cancel_time)
                if remaining_amount == order.Volume - order.FilledVolume:
                    self._on_order(order=order)

    # noinspection PyPep8Naming
    @classmethod
    def _to_HuobiOrderType(cls, order_type: OrderType, trade_side: TradeSide) -> HuobiOrderType:
        if order_type == OrderType.LimitOrder:
            if trade_side.value > 0:
                huobi_order_type = HuobiOrderType.Buy_Limit
            elif trade_side.value < 0:
                huobi_order_type = HuobiOrderType.Sell_Limit
            else:
                raise ValueError(f'Invalid TradeSide {trade_side}')
        elif order_type == OrderType.MarketOrder:
            if trade_side.value > 0:
                huobi_order_type = HuobiOrderType.Buy_Market
            elif trade_side.value < 0:
                huobi_order_type = HuobiOrderType.Sell_Market
            else:
                raise ValueError(f'Invalid TradeSide {trade_side}')
        elif order_type == OrderType.IOC:
            if trade_side.value > 0:
                huobi_order_type = HuobiOrderType.Buy_IOC
            elif trade_side.value < 0:
                huobi_order_type = HuobiOrderType.Sell_IOC
            else:
                raise ValueError(f'Invalid TradeSide {trade_side}')
        elif order_type == OrderType.LimitMarketMaking:
            if trade_side.value > 0:
                huobi_order_type = HuobiOrderType.Buy_Making
            elif trade_side.value < 0:
                huobi_order_type = HuobiOrderType.Sell_Making
            else:
                raise ValueError(f'Invalid TradeSide {trade_side}')
        else:
            raise ValueError(f'Invalid OrderType {order_type}')

        return huobi_order_type

    # noinspection PyPep8Naming
    @classmethod
    def _from_HuobiOrderType(cls, huobi_order_type: HuobiOrderType) -> Tuple[OrderType, TradeSide]:
        if huobi_order_type == huobi_order_type.Buy_Limit:
            order_type = OrderType.LimitOrder
            trade_side = TradeSide.LongOpen
        elif huobi_order_type == huobi_order_type.Sell_Limit:
            order_type = OrderType.LimitOrder
            trade_side = TradeSide.LongClose
        elif huobi_order_type == huobi_order_type.Buy_Market:
            order_type = OrderType.MarketOrder
            trade_side = TradeSide.LongOpen
        elif huobi_order_type == huobi_order_type.Sell_Market:
            order_type = OrderType.MarketOrder
            trade_side = TradeSide.LongClose
        elif huobi_order_type == huobi_order_type.Buy_IOC:
            order_type = OrderType.IOC
            trade_side = TradeSide.LongOpen
        elif huobi_order_type == huobi_order_type.Sell_IOC:
            order_type = OrderType.IOC
            trade_side = TradeSide.LongClose
        elif huobi_order_type == huobi_order_type.Buy_Making:
            order_type = OrderType.LimitMarketMaking
            trade_side = TradeSide.LongOpen
        elif huobi_order_type == huobi_order_type.Sell_Making:
            order_type = OrderType.LimitMarketMaking
            trade_side = TradeSide.LongClose
        else:
            raise ValueError(f'Invalid HuobiOrderType {huobi_order_type}')

        return order_type, trade_side

    # noinspection PyPep8Naming
    def _to_HuobiOrderState(self, order_state: OrderState) -> str:
        state_map = {
            OrderState.Pending: 'pre-submitted',
            OrderState.Placed: 'submitted',
            OrderState.PartFilled: 'partial-filled',
            OrderState.PartCanceled: 'partial-canceled',
            OrderState.Filled: 'filled',
            OrderState.Canceled: 'canceled'
        }

        if order_state in state_map.keys():
            return state_map[order_state]
        else:
            self.logger.warning(f'{order_state} huobi state str not found! returning unknown')
            return 'unknown'

    # noinspection PyPep8Naming
    def _from_HuobiOrderState(self, huobi_order_state: str) -> OrderState:
        state_map = {
            'pre-submitted': OrderState.Pending,
            'submitted': OrderState.Placed,
            'partial-filled': OrderState.PartFilled,
            'partial-canceled': OrderState.PartCanceled,
            'filled': OrderState.Filled,
            'canceled': OrderState.Canceled
        }

        if huobi_order_state in state_map.keys():
            return state_map[huobi_order_state]
        else:
            self.logger.warning(f'{huobi_order_state} OrderState not found! returning OrderState.UNKNOWN')
            return OrderState.UNKNOWN

    def cancel_open_orders(
            self,
            ticker: Optional[int] = None,
            trade_side: Optional[TradeSide] = None,
            size: Optional[int] = 2000
    ) -> Optional[dict]:
        """
        query all open orders
        :param ticker: the given symbol
        :param trade_side: 'buy', 'sell' :keyword 'both', default is 'both'
        :param size: maximum = 2000
        :return:
        """

        params = {}

        if ticker is not None:
            params['symbol'] = ticker
            params['account-id'] = self.account_id

        if trade_side:
            if trade_side.value > 0:
                params['side'] = 'buy'
            else:
                params['side'] = 'sell'

        if size:
            params['size'] = size

        # noinspection PyBroadException
        try:
            result = self.rest_client.request_signed(request='post', url=self.rest_client.url + "/v1/order/orders/batchCancelOpenOrders", params=params)
            if result['status'] == 'ok':
                self.logger.info(f'cancel success {result["data"]["success-count"]}; failed {result["data"]["failed-count"]}; next id {result["data"]["next-id"]}')
                return result['data']
            else:
                self.logger.warning(f'Cancel open orders failed! err-code: "{result["err-code"]}", err-msg: "{result["err-msg"]}"')
        except Exception as _:
            self.logger.warning(traceback.format_exc())

    def query_order(self, ticker: Optional[str] = None, trade_side: Optional[TradeSide] = None, size: Optional[int] = 2000) -> Optional[List[TradeInstruction]]:
        """
        query all open orders
        :param ticker:
        :param trade_side:
        :param size:
        :return:
        """
        params = {}
        trade_instruction_list = []

        if ticker is not None:
            params['symbol'] = ticker
            params['account-id'] = self.account_id

        if trade_side:
            if trade_side.sign > 0:
                params['side'] = 'buy'
            else:
                params['side'] = 'sell'

        if size:
            params['size'] = size

        # noinspection PyBroadException
        try:
            result = self.rest_client.request_signed(request='get', url=self.rest_client.url + "/v1/order/openOrders", params=params)
            if result['status'] == 'ok':
                instructions = result['data']
                for instruction_log in instructions:  # type: dict
                    order_type, trade_side = self._from_HuobiOrderType(huobi_order_type=HuobiOrderType(instruction_log['type']))
                    order_state = self._from_HuobiOrderState(instruction_log['state'])
                    ticker = instruction_log['symbol']
                    total_volume = float(instruction_log['amount'])
                    limit_price = float(instruction_log['price']) if 'price' in instruction_log else float('nan')
                    order_id = str(instruction_log['id'])
                    start_datetime = datetime.datetime.utcfromtimestamp(instruction_log['created-at'] / 1000)

                    logged_instruction = self._position.order.get(order_id)
                    if logged_instruction is None:
                        trade_instruction = TradeInstruction(
                            ticker=ticker,
                            side=trade_side,
                            order_type=order_type,
                            volume=total_volume,
                            limit_price=limit_price,
                            order_id=order_id
                        )

                        trade_instruction.set_order_state(order_state=OrderState.Placed, market_datetime=start_datetime)
                        trade_instruction.set_order_state(order_state=order_state, market_datetime=start_datetime)
                        self._position.log_order(order=trade_instruction)
                        self.query_trade(order_id=order_id)
                        logged_instruction = trade_instruction
                    else:
                        assert logged_instruction.Ticker == ticker
                        assert logged_instruction.Side == trade_side
                        assert logged_instruction.OrderType == order_type
                        assert logged_instruction.Volume == total_volume
                    trade_instruction_list.append(logged_instruction)
            else:
                self.logger.warning(f'Query open orders failed! err-code: "{result["err-code"]}", err-msg: "{result["err-msg"]}"')
        except Exception as _:
            self.logger.warning(traceback.format_exc())

        return trade_instruction_list

    def query_trade(self, order_id: Optional[Union[int, str]] = None) -> Optional[List[TradeReport]]:
        """
        query trade by OrderId from Huobi
        :param order_id: the given order id
        :return:
        """
        logged_instruction = self._position.order.get(order_id)
        trade_report_list = []
        if logged_instruction is None:
            self.logger.warning(f'Order {order_id} not found in position')
            logged_instruction, filled_volume = self.query_order()

        assert logged_instruction is not None

        # noinspection PyBroadException
        try:
            # noinspection SpellCheckingInspection
            result = self.rest_client.request_signed(request='get', url=self.rest_client.url + f"/v1/order/orders/{order_id}/matchresults")
            if result['status'] == 'ok':
                trades = result['data']
                for trade_log in trades:
                    _, trade_side = self._from_HuobiOrderType(huobi_order_type=HuobiOrderType(trade_log['type']))

                    trade_report = TradeReport(
                        ticker=trade_log['symbol'],
                        side=trade_side,
                        volume=float(trade_log['filled-amount']),
                        notional=float(trade_log['filled-amount']) * float(trade_log['price']),
                        trade_time=datetime.datetime.utcfromtimestamp(trade_log['created-at'] / 1000),
                        order_id=str(trade_log['order-id']),
                        trade_id=str(trade_log['trade-id']),
                        fee=float(trade_log['filled-fees'])
                    )

                    if trade_report.TradeID not in logged_instruction.Trades:
                        self._position.log_report(report=trade_report)
                        trade_report_list.append(trade_report)
            else:
                self.logger.warning(f'Query Order {order_id} trades failed! err-code: "{result["err-code"]}", err-msg: "{result["err-msg"]}"')
        except Exception as _:
            self.logger.warning(traceback.format_exc())

        return trade_report_list

    @property
    def position(self):
        return self._position.position

    @property
    def working_orders(self):
        return self._position.working

    @property
    def balance(self):
        raise NotImplementedError()
