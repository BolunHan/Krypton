import datetime
import traceback
import uuid
from enum import Enum
from typing import Optional, List, Union, Type, Dict

import numpy as np
import pandas as pd

from .. import WebsocketsClient, RestClient
from ... import Template
from ....Base import GlobalStatics
from ....Res.ToolKit import BarData, TickData, TradeData, OrderBook, TradeInstruction, TradeSide, TradeReport, OrderState, OrderType, AttrDict

__all__ = ['HuobiContractOrderPriceType', 'Market', 'Trade']


class HuobiContractOrderPriceType(Enum):
    LIMIT = 'limit'
    OPPONENT = 'opponent'
    POST_ONLY = 'post_only'
    OPTIMAL_5 = 'optimal_5'
    OPTIMAL_10 = 'optimal_10'
    OPTIMAL_20 = 'optimal_20'
    IOC = 'ioc'
    FOK = 'fok'


class Market(Template.Market):
    def __init__(
            self,
            market_url,
            event_engine,
            http_proxy: str = None,
            logger=None,
            tick_level: str = 'step0',
            bar_span: datetime.timedelta = datetime.timedelta(seconds=60),
            tick_span: Optional[float] = None,
    ):
        super(Market, self).__init__(
            event_engine=event_engine,
            logger=logger,
        )

        self.market_client = WebsocketsClient.HuobiWebsocketsClient(
            name='Huobi.Contract.Market.Websockets.Client',
            url=market_url,
            event_engine=event_engine,
            proxy=http_proxy
        )

        self.market_level = tick_level
        self.synthetic_bar_span = bar_span
        self.tick_span = tick_span

        self.market_client.connect()

        self.last_bar = {}
        self.last_tick = {}

    def _parse_bar(self, data_dict):
        topic: str = data_dict['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]

        pre_bar: BarData = self.last_bar[ticker]['pre']['bar']
        active_bar: BarData = self.last_bar[ticker]['active']['bar']
        future_bar: BarData = self.last_bar[ticker]['next']['bar']

        if channel == 'trade':
            market_time = datetime.datetime.utcfromtimestamp(data_dict['tick']['ts'] / 1000)
            self.last_bar[ticker]['market_time'] = market_time
            to_update = None
            side = 0
            for trade_data_dict in data_dict['tick']['data']:
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
            market_time = datetime.datetime.utcfromtimestamp(data_dict['ts'] / 1000)
            total_trade = data_dict['tick']['count']
            total_volume = data_dict['tick']['amount']
            total_notional = data_dict['tick']['vol']
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

    def _parse_tick(self, data_dict):
        topic: str = data_dict['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]
        market_time = datetime.datetime.utcfromtimestamp(data_dict['ts'] / 1000)

        if channel == 'detail':
            if 'market_time' not in self.last_tick[ticker]['detail'] or self.last_tick[ticker]['detail']['market_time'] < market_time:
                self.last_tick[ticker]['detail'] = data_dict['tick']
                self.last_tick[ticker]['detail']['market_time'] = market_time
        elif channel == 'depth':
            order_book = OrderBook(
                ticker=ticker,
                market_time=market_time
            )

            for entry in data_dict['tick']['bids']:
                order_book.bid.add(price=entry[0], volume=entry[1])

            for entry in data_dict['tick']['asks']:
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
                self._publish(market_data=tick_data)
            else:
                self.last_tick[ticker]['tick'] = tick_data

    def _parse_trade(self, data_dict):
        topic: str = data_dict['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]
        market_time = datetime.datetime.utcfromtimestamp(data_dict['tick']['ts'] / 1000)

        if channel == 'trade':
            for trade_data_dict in data_dict['tick']['data']:
                trade_data = TradeData(
                    ticker=ticker,
                    trade_time=market_time,
                    trade_price=trade_data_dict['price'],
                    trade_volume=trade_data_dict['amount'],
                    side=trade_data_dict['direction']
                )

                # self.last_trade[ticker]['trade'].put(trade_data, block=False)
                self._publish(market_data=trade_data)

    def _parse_order_book(self, data_dict):
        topic: str = data_dict['ch']
        ticker = topic.split('.')[1]
        channel = topic.split('.')[2]
        market_time = datetime.datetime.utcfromtimestamp(data_dict['tick']['ts'] / 1000)

        if channel == 'depth':
            order_book = OrderBook(
                ticker=ticker,
                market_time=market_time
            )

            for entry in data_dict['tick']['bids']:
                order_book.bid.add(price=entry[0], volume=entry[1])

            for entry in data_dict['tick']['asks']:
                order_book.ask.add(price=entry[0], volume=entry[1])

            self._publish(market_data=order_book)

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

        def push_bar():
            active_bar = self.last_bar[ticker]['active']['bar']
            market_time = datetime.datetime.utcnow()
            error_range = datetime.timedelta(seconds=1)

            while active_bar.bar_start_time + self.synthetic_bar_span - error_range <= market_time <= active_bar.bar_start_time + self.synthetic_bar_span + error_range:
                self._publish(market_data=active_bar)
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
        topic = self.event_engine.get_timer(interval=self.synthetic_bar_span, activate_time=t)
        self.event_engine.register_handler(topic=topic, handler=push_bar)

    def _tick_thread(self, ticker: str):
        self.last_tick[ticker] = {'tick': None, 'detail': {}, 'depth': {}}

        self.market_client.subscribe(topic=f'market.{ticker}.detail', callback=self._parse_tick)
        self.market_client.subscribe(topic=f'market.{ticker}.depth.{self.market_level}', callback=self._parse_tick)

        if self.tick_span is not None and self.tick_span > 0:
            t = datetime.datetime.utcnow()
            t -= datetime.timedelta(microseconds=t.microsecond)
            topic = self.event_engine.get_timer(interval=self.tick_span, activate_time=t)

            def push_tick():
                tick_data = self.last_tick[ticker]['tick']
                if tick_data is not None:
                    self._publish(market_data=tick_data)
                    self.last_tick[ticker]['tick'] = None

            self.event_engine.register_handler(topic=topic, handler=push_tick)

    def _trade_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'market.{ticker}.trade.detail', callback=self._parse_trade)

    def _order_book_thread(self, ticker: str):
        self.market_client.subscribe(topic=f'market.{ticker}.depth.{self.market_level}', callback=self._parse_order_book)

    def _topic(self, ticker: Optional[str] = None, dtype: Optional[Type] = None) -> str:
        return GlobalStatics.TOPIC.subscribe(ticker=ticker, dtype=dtype)

    def _subscribed(self, ticker: str, dtype: Type, **kwargs):
        topic = self._topic(ticker=ticker, dtype=dtype)
        if topic not in self.subscription:
            if dtype.__name__ == 'BarData':
                self._bar_thread(ticker=ticker)
            elif dtype.__name__ == 'TickData':
                self._tick_thread(ticker=ticker)
            elif dtype.__name__ == 'TradeData':
                self._trade_thread(ticker=ticker)
            elif dtype.__name__ == 'OrderBook':
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
            http_proxy: str = None,
            logger=None,
    ):
        super().__init__(event_engine=event_engine, logger=logger)

        self.websockets_client = WebsocketsClient.HuobiWebsocketsClient(
            name='Huobi.Contract.Trade.Websockets.Client',
            access_key=access_key,
            secret_key=secret_key,
            url=trade_url,
            event_engine=event_engine,
            proxy=http_proxy
        )

        self.rest_client = RestClient.HuobiRestClient(
            name='Huobi.Contract.Trade.REST.Client',
            access_key=access_key,
            secret_key=secret_key,
            url=rest_url,
            proxy=http_proxy
        )

        self.websockets_client.on_open_jobs.append(self._on_ws_connect)
        self.websockets_client.on_reconnect_jobs.append(self._on_ws_connect)
        self.websockets_client.connect()

        self._position = Template.Position()

        self._trading_contracts: Optional[List[AttrDict]] = None
        self.websockets_client.subscribe(message={'op': 'sub', 'cid': str(uuid.uuid4()), 'topic': 'public.*.contract_info'}, callback=self._update_contract)

    def _topic(self, order: TradeInstruction = None, report: TradeReport = None) -> str:
        if order:
            return GlobalStatics.TOPIC.on_order
        elif report:
            return GlobalStatics.TOPIC.on_report
        else:
            raise ValueError('Can not find topic')

    @staticmethod
    def _alias_suffix(contract_type: str):
        if contract_type == 'this_week':
            alias_suffix = 'CW'
        elif contract_type == 'next_week':
            alias_suffix = 'NW'
        elif contract_type == 'quarter' or contract_type == 'this_quarter':
            alias_suffix = 'CQ'
        elif contract_type == 'next_quarter':
            alias_suffix = 'NQ'
        else:
            raise ValueError(f'Invalid contract type {contract_type}')

        return alias_suffix

    def _on_ws_connect(self, websockets_client):
        websockets_client.auth(version='2')

    def _on_order(self, order: TradeInstruction):
        self.event_engine.put(topic=self._topic(order=order), order=order)

    def _on_report(self, report: TradeReport):
        self.event_engine.put(topic=self._topic(report=report), report=report)

    def _on_error(self, **kwargs):
        self.event_engine.put(topic=GlobalStatics.TOPIC.on_error, **kwargs)

    # noinspection PyPep8Naming
    def _from_HuobiOrderState(self, huobi_order_state: int) -> OrderState:
        state_map = {
            1: OrderState.Pending,
            2: OrderState.Pending,
            3: OrderState.Placed,
            4: OrderState.PartFilled,
            5: OrderState.PartCanceled,
            6: OrderState.Filled,
            7: OrderState.Canceled,
            11: OrderState.Canceling
        }

        if huobi_order_state in state_map.keys():
            return state_map[huobi_order_state]
        else:
            self.logger.warning(f'{huobi_order_state} OrderState not found! returning OrderState.UNKNOWN')
            return OrderState.UNKNOWN

    def _update_order(self, message):
        order_id = str(message['order_id'])
        market_datetime = datetime.datetime.utcfromtimestamp(message['ts'] / 1000)
        contract_code = message['contract_code']
        trade_side = TradeSide.from_offset(direction=message['direction'], offset=message['offset'])
        order_state = self._from_HuobiOrderState(message['status'])
        margin = float(message['margin_frozen'])

        if order_id in self._position.order:
            trade_instruction = self._position.order[order_id]
        else:
            self.logger.warning(f'Order {order_id} not found! Perhaps it\'s not created by this client?')
            leverage = float(message['lever_rate'])
            limit_price = float(message['price'])
            volume = float(message['volume'])
            order_type = OrderType.LimitOrder

            # total_trade_volume = float(message['trade_volume'])
            # total_trade_turnover = float(message['trade_turnover'])

            trade_instruction = TradeInstruction(
                ticker=contract_code,
                side=trade_side,
                order_type=order_type,
                volume=volume,
                limit_price=limit_price,
                order_id=order_id
            )

            trade_instruction.leverage = leverage
            trade_instruction.huobi_order_id = order_id

            self._position.log_order(order=trade_instruction)

        trade_instruction.margin = margin

        for trade_log in message['trade']:
            trade_id = str(trade_log['id'])
            trade_volume = float(trade_log['trade_volume'])
            contract_price = float(trade_log['trade_price'])
            trade_notional = float(trade_log['trade_turnover'])
            trade_fee = float(trade_log['trade_fee'])
            trade_time = datetime.datetime.utcfromtimestamp(message['created_at'] / 1000)
            fee_asset = str(trade_log['fee_asset'])

            trade_report = TradeReport(
                ticker=contract_code,
                side=trade_side,
                volume=trade_volume,
                price=contract_price,
                notional=trade_notional,
                trade_time=trade_time,
                order_id=order_id,
                trade_id=trade_id,
                fee=trade_fee
            )

            trade_report.fee_asset = fee_asset

            if trade_id not in self._position.trade[order_id]:
                self._position.log_report(report=trade_report)
                self._on_report(report=trade_report)

        if order_state in [OrderState.Placed, OrderState.Canceling, OrderState.Canceled, OrderState.PartCanceled]:
            trade_instruction.set_order_state(order_state=order_state, market_datetime=market_datetime)

        self._on_order(order=trade_instruction)

    def _update_contract(self, message, **kwargs):
        contract_log = message.get('data', [])
        event = message.get('event', None)

        if self._trading_contracts and event in ['snapshot', 'init']:
            return

        contract_info_list = []

        for contract_dict in contract_log:
            contract_status = int(contract_dict['contract_status'])
            if contract_status == 1:
                contract_info = AttrDict(
                    symbol=contract_dict['symbol'],
                    alias_suffix=self._alias_suffix(contract_type=contract_dict['contract_type']),
                    contract_code=contract_dict['contract_code'],
                    contract_type=contract_dict['contract_type'],
                    contract_size=float(contract_dict['contract_size']),
                    price_tick=float(contract_dict['price_tick']),
                    delivery_date=datetime.datetime.strptime(contract_dict['delivery_date'], '%Y%m%d').date(),
                    create_date=datetime.datetime.strptime(contract_dict['create_date'], '%Y%m%d').date()
                )
                contract_info['contract_alias'] = f'{contract_info["symbol"]}_{contract_info["alias_suffix"]}'
                contract_info_list.append(contract_info)

        self._trading_contracts = contract_info_list

        self.logger.debug('Contract info updated')

    def send_order(self, trade_instruction: TradeInstruction, leverage: int = 10, order_price_type: HuobiContractOrderPriceType = HuobiContractOrderPriceType.LIMIT, *args, **kwargs) -> Optional[str]:
        """
        send an order to huobi contract exchange
        :param trade_instruction: the given trade instruction
        :param leverage: leverage rate, 1, 5, 10 or 20. 20-times leverage order is forbidden if any 10-times leverage position exist. Default = 10
        :param order_price_type: a HuobiContractOrderPriceType indicating the order type of the given instruction, Default = Limit
        :return:
        """
        # symbol = self.query_contracts(trade_instruction.Ticker).symbol
        # self.websockets_client.subscribe(message={'op': 'sub', 'cid': str(uuid.uuid4()), 'topic': f'orders.{symbol}'}, callback=self._update_order)
        self._position.log_order(order=trade_instruction)

        if trade_instruction.Volume != int(trade_instruction.Volume):
            raise ValueError(f'Contract volume must be integer, not {trade_instruction.Volume}')

        params = {
            "volume": int(trade_instruction.Volume),
            "lever_rate": leverage,
            "order_price_type": order_price_type.value,
        }

        ticker = trade_instruction.Ticker

        if ticker in self.contracts['contract_alias'].values:
            contract_code = self.contracts.set_index('contract_alias').at[ticker, 'contract_code']
            # reset ticker to contract_code
            trade_instruction.TradeInstruction__ticker = contract_code
        elif ticker in self.contracts['contract_code'].values:
            contract_code = ticker
        else:
            raise ValueError(f'Invalid ticker {ticker}')

        params.update({'contract_code': contract_code})

        if trade_instruction.Side == TradeSide.LongOpen:
            params.update({'direction': 'buy', 'offset': 'open'})
        elif trade_instruction.Side == TradeSide.LongClose:
            params.update({'direction': 'sell', 'offset': 'close'})
        elif trade_instruction.Side == TradeSide.ShortOpen:
            params.update({'direction': 'sell', 'offset': 'open'})
        elif trade_instruction.Side == TradeSide.ShortClose:
            params.update({'direction': 'buy', 'offset': 'close'})
        else:
            raise ValueError(f'Invalid TradeSide {trade_instruction.Side}')

        if trade_instruction.LimitPrice is not None:
            params["price"] = trade_instruction.LimitPrice
        # noinspection PyBroadException
        try:
            self.logger.debug(f'Sending order {str(trade_instruction)}')
            result = self.rest_client.request_signed(request='post', url=self.rest_client.url + '/api/v1/contract_order', params=params)
            if result['status'] == 'ok':
                trade_instruction.huobi_order_id = str(result['data']['order_id_str'])
                trade_instruction.leverage = leverage

                self.logger.debug(f'Order {trade_instruction.OrderID} placement successful! Resetting order id to {trade_instruction.huobi_order_id}')
                trade_instruction.set_order_state(OrderState.Pending, market_datetime=datetime.datetime.utcfromtimestamp(result['ts'] / 1000))
                order_id = trade_instruction.huobi_order_id
                self._position.reset_order_id(from_id=trade_instruction.OrderID, to_id=trade_instruction.huobi_order_id)
            else:
                self.logger.warning(f'Order {trade_instruction.OrderID} placement failed! err_code: "{result["err_code"]}", err_msg: "{result["err_msg"]}"')
                trade_instruction.set_order_state(OrderState.UNKNOWN)
                order_id = None
                self._on_error(order=trade_instruction, **result)
        except Exception as _:
            self.logger.warning(traceback.format_exc())
            trade_instruction.set_order_state(OrderState.UNKNOWN)
            order_id = None
            self._on_error(order=trade_instruction)

        return order_id

    def cancel_order(self, order_id: str, **kwargs):
        """
        post a cancel order instruction
        :param order_id:
        :return:
        """

        trade_instruction = self._position.order.get(order_id)
        if trade_instruction is None:
            raise KeyError(f'order_id {order_id} not found')
        symbol = self.query_contracts(trade_instruction.Ticker).symbol

        params = {'symbol': symbol, 'order_id': order_id}

        # noinspection PyBroadException
        try:
            self.logger.debug(f'Canceling order {order_id}')
            # noinspection SpellCheckingInspection
            result = self.rest_client.request_signed(request='post', url=self.rest_client.url + "/api/v1/contract_cancel", params=params)
            if result['status'] == 'ok':
                if order_id in result['data']['successes']:
                    order_state = OrderState.Canceling
                else:
                    error_log = result['data']['errors'][0]
                    self.logger.warning(f'Order {order_id} canceling failed! err_code: "{error_log["err_code"]}", err_msg: "{error_log["err_msg"]}"')
                    order_state = OrderState.UNKNOWN
                    self._on_error(order=trade_instruction, **error_log)
            else:
                self.logger.warning(f'Order {order_id} canceling failed! err_code: "{result["err_code"]}", err_msg: "{result["err_msg"]}"')
                order_state = OrderState.UNKNOWN
                self._on_error(order=trade_instruction, **result)
        except Exception as _:
            self.logger.warning(traceback.format_exc())
            order_state = OrderState.UNKNOWN
            self._on_error(order=trade_instruction)

        trade_instruction.set_order_state(order_state)

    def query_orders(self, symbol: Optional[str] = None) -> Optional[List[TradeInstruction]]:
        """
        query all open orders
        :param symbol: 'BTC', 'ETH', etc.
        :return:
        """
        params = {}
        trade_instruction_list = []

        if symbol is not None:
            params['symbol'] = symbol

        # noinspection PyBroadException
        try:
            result = self.rest_client.request(request='post', url=self.rest_client.url + "/api/v1/contract_openorders", params=params)
            if result['status'] == 'ok':
                instructions = result['data']['orders']
                for instruction_log in instructions:  # type: dict
                    ticker = instruction_log['contract_code']
                    total_volume = float(instruction_log['volume'])
                    leverage = int(instruction_log['lever_rate'])
                    margin = float(instruction_log['margin_frozen'])
                    limit_price = float(instruction_log['price']) if 'price' in instruction_log else float('nan')
                    order_type = OrderType.LimitOrder
                    start_datetime = datetime.datetime.utcfromtimestamp(instruction_log['created_at'] / 1000)
                    order_id = str(instruction_log['order_id_str'])
                    order_state = self._from_HuobiOrderState(instruction_log['status'])
                    trade_side = TradeSide.from_offset(direction=instruction_log['direction'], offset=instruction_log['offset'])

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
                        trade_instruction.set_order_state(order_state=order_state)
                        trade_instruction.leverage = leverage
                        trade_instruction.margin = margin

                        self._position.log_order(order=trade_instruction)
                        self.query_trades(order_id=order_id)
                        logged_instruction = trade_instruction
                    else:
                        assert logged_instruction.Ticker == ticker
                        assert logged_instruction.Side == trade_side
                        assert logged_instruction.OrderType == order_type
                        assert logged_instruction.Volume == total_volume
                    trade_instruction_list.append(logged_instruction)
            else:
                self.logger.warning(f'Query open orders failed! err_code: "{result["err_code"]}", err_msg: "{result["err_msg"]}"')
        except Exception as _:
            self.logger.warning(traceback.format_exc())

        return trade_instruction_list

    def query_trades(self, order_id: str, symbol: Optional[str] = None):
        params = {'symbol': symbol, 'order_id': order_id}
        trade_report_list = []

        # noinspection PyBroadException
        try:
            # noinspection SpellCheckingInspection
            result = self.rest_client.request_signed(request='post', url=self.rest_client.url + "/api/v1/contract_order_detail", params=params)

            if result['status'] == 'ok':
                contract_code = str(result['contract_code'])
                trade_side = TradeSide.from_offset(direction=result['direction'], offset=result['offset'])
                fee_asset = str(result['fee_asset'])
                if result['data'] is None:
                    trades = []
                else:
                    trades = result['data']['trades']

                for trade_log in trades:
                    trade_id = str(trade_log['id'])

                    trade_report = TradeReport(
                        ticker=contract_code,
                        side=trade_side,
                        volume=float(trade_log['trade_volume']),
                        price=float(trade_log['trade_price']),
                        notional=float(trade_log['trade_turnover']),
                        trade_time=datetime.datetime.utcfromtimestamp(trade_log['created_at'] / 1000),
                        order_id=str(result['order_id']),
                        trade_id=trade_id,
                        fee=float(trade_log['trade_fee'])
                    )

                    trade_report.fee_asset = fee_asset
                    # trade_report.trade_turnover = float(trade_log['trade_turnover'])
                    # trade_report.role = str(trade_log['role'])
                    if trade_id not in self._position.trade[order_id]:
                        self._position.log_report(report=trade_report)
                        trade_report_list.append(trade_report)
            else:
                self.logger.warning(f'Query Order {order_id} trades failed! err_code: "{result["err_code"]}", err_msg: "{result["err_msg"]}"')
        except Exception as _:
            self.logger.warning(traceback.format_exc())

        return trade_report_list

    def query_contracts(self, *args, **kwargs) -> Optional[Union[AttrDict, List[AttrDict]]]:
        """
        Query trading contracts info
        """
        # noinspection PyBroadException

        if self._trading_contracts is None:
            return None

        result = self._trading_contracts[:]

        if args:
            result = [x for x in result if all([v in x.values() for v in args])]

        if kwargs:
            result = [x for x in result if all([x[k] == v for k, v in kwargs.items()])]

        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            return result

    def query_open_interest(self):
        # noinspection PyBroadException
        try:
            result = self.rest_client.request(request='get', url=self.rest_client.url + '/api/v1/contract_open_interest')
            if result['status'] == 'ok':
                open_interest = pd.DataFrame(index=pd.MultiIndex(levels=[[]] * 2, codes=[[]] * 2, name=['symbol', 'type']))

                for contract_dict in result['data']:
                    symbol = contract_dict['symbol']
                    alias_suffix = self._alias_suffix(contract_type=contract_dict['contract_type'])
                    contract_alias = f'{symbol}_{alias_suffix}'
                    contract_code = contract_dict['contract_code']
                    contract_type = contract_dict['contract_type']
                    volume = float(contract_dict['volume'])
                    notional = float(contract_dict['amount'])

                    open_interest.at[(symbol, alias_suffix), 'contract_code'] = contract_code
                    open_interest.at[(symbol, alias_suffix), 'contract_alias'] = contract_alias
                    open_interest.at[(symbol, alias_suffix), 'contract_type'] = contract_type
                    open_interest.at[(symbol, alias_suffix), 'volume'] = volume
                    open_interest.at[(symbol, alias_suffix), 'notional'] = notional

                return open_interest
            else:
                self.logger.warning(f'Query open interest failed! err_code: "{result["err_code"]}", err_msg: "{result["err_msg"]}"')
                return None
        except Exception as _:
            self.logger.warning(traceback.format_exc())
            raise Exception('Query open interest failed!')

    @property
    def contracts(self):
        trading_contracts = pd.DataFrame(index=pd.MultiIndex(levels=[[]] * 2, codes=[[]] * 2, name=['symbol', 'type']))

        if self._trading_contracts is None:
            message = self.rest_client.request(request='get', url=self.rest_client.url + '/api/v1/contract_contract_info')
            if message['status'] == 'ok':
                self._update_contract(message=message)
            else:
                self.logger.error(f'Query contracts info failed! err_code: "{message["err_code"]}", err_msg: "{message["err_msg"]}"')
                return trading_contracts

        for contract_log in self._trading_contracts:
            symbol = contract_log['symbol']
            alias_suffix = contract_log['alias_suffix']
            contract_code = contract_log['contract_code']
            contract_type = contract_log['contract_type']
            contract_size = contract_log['contract_size']
            price_tick = contract_log['price_tick']
            delivery_date = contract_log['delivery_date']
            create_date = contract_log['create_date']
            contract_alias = contract_log['contract_alias']

            trading_contracts.at[(symbol, alias_suffix), 'contract_code'] = contract_code
            trading_contracts.at[(symbol, alias_suffix), 'contract_alias'] = contract_alias
            trading_contracts.at[(symbol, alias_suffix), 'contract_type'] = contract_type
            trading_contracts.at[(symbol, alias_suffix), 'contract_size'] = contract_size
            trading_contracts.at[(symbol, alias_suffix), 'price_tick'] = price_tick
            trading_contracts.at[(symbol, alias_suffix), 'delivery_date'] = delivery_date
            trading_contracts.at[(symbol, alias_suffix), 'create_date'] = create_date

        return trading_contracts

    @property
    def working_orders(self) -> Dict[Union[str, int, uuid.UUID], TradeInstruction]:
        working_orders = {order_id: trade_instruction for order_id, trade_instruction in self._position.order.items() if trade_instruction.OrderState in [OrderState.Placed, OrderState.PartFilled, OrderState.Canceling]}

        return working_orders

    @property
    def position(self):
        return self._position.position

    @property
    def balance(self):
        raise NotImplementedError()
