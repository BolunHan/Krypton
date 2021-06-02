import copy
import datetime
import json
import os
import uuid
from enum import Enum
from typing import Union, Optional, Dict

from . import GLOBAL_LOGGER
from .MarketUtils import TradeSide, TradeData

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['OrderState', 'OrderType', 'TradeInstruction', 'TradeReport']


class OrderType(Enum):
    UNKNOWN = -2
    CancelOrder = -1
    Manual = 0
    LimitOrder = 1
    LimitMarketMaking = 1.1
    MarketOrder = 2
    FOK = 2.1
    FAK = 2.2
    IOC = 2.3

    def __hash__(self):
        return self.value


class OrderState(Enum):
    UNKNOWN = -3
    Rejected = -2
    Invalid = -1
    Pending = 0
    Placed = 1
    PartFilled = 2
    Filled = 3
    Canceling = 4
    PartCanceled = 5
    Canceled = 6

    def __hash__(self):
        return self.value


class TradeReport(object):
    def __init__(
            self, *,
            ticker: str,
            side: Union[int, float, str, TradeSide],
            volume: float,
            notional: float,
            trade_time: datetime.datetime,
            order_id: str,
            price: Optional[float] = None,
            trade_id: Optional[str] = None,
            multiplier: float = 1,
            fee: float = .0
    ):
        """
        store trade report data
        :param ticker: ticker (symbol) of the given asset (stock, future, option, crypto and etc.)
        :param side: TradeSide should be the same as TradeInstruction
        :param volume: Traded volume (the number of shares, contracts or crypto, etc.)
        :param notional: Traded notional (the amount of money) or premium of the option
        :param trade_time: datetime.datetime when trade was matched
        :param order_id: the id of its TradeInstruction
        :param price: the traded price. NOTED: trade price does not necessarily equals notional / volume. For example, currency swap, crypto swap (future) and debt
        :param trade_id: the id of itself
        :param multiplier: multiplier for contract or option
        :param fee: transition fee of this trade
        """
        assert volume >= 0, 'Trade volume must not be negative'
        self.__ticker = str(ticker)
        self.__side = TradeSide(side)
        self.__volume = volume
        self.__notional = notional
        self.__trade_time = trade_time
        self.__order_id = str(order_id)
        self.__price = price
        self.__trade_id = str(trade_id) if trade_id is not None else str(uuid.uuid4())
        self.__multiplier = float(multiplier)
        self.__fee = fee

    def __repr__(self):
        return '<TradeReport>{}'.format({key: item.name if key == 'side' else item for key, item in self.__dict__.items()})

    def __eq__(self, other):
        if isinstance(other, TradeReport):
            return self.__repr__() == other.__repr__()
        else:
            return False

    def __str__(self):
        return '<TradeReport>([{:%Y-%m-%d %H:%M:%S}] OrderID {} {} {} {:.2f} @ {:.2f} with TradeID {})'.format(self.TradeTime, self.OrderID, self.Ticker, self.Side.name, self.Volume, self.Price, self.TradeID)

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    def reset_order_id(self, order_id: Optional[str] = None, **kwargs) -> 'TradeReport':
        """
        reset order_id id to given string
        :param order_id: new order id, use UUID by default
        :return:
        """
        if not kwargs.pop('_ignore_warning', False):
            LOGGER.warning('TradeReport OrderID being reset manually! TradeInstruction.reset_order_id() is the recommended method to do so.')

        if order_id is not None:
            self.__order_id = str(order_id)
        else:
            self.__order_id = str(uuid.uuid4())

        return self

    def reset_trade_id(self, trade_id: Optional[str] = None) -> 'TradeReport':
        """
        reset trade id to given string
        :param trade_id:
        :return:
        """
        if trade_id is not None:
            self.__trade_id = str(trade_id)
        else:
            self.__trade_id = str(uuid.uuid4())

        return self

    def to_trade(self) -> TradeData:
        trade = TradeData(
            ticker=self.Ticker,
            trade_time=self.TradeTime,
            trade_price=self.Notional / self.Volume / self.__multiplier if self.Volume > 0 else 0,
            trade_volume=self.Volume,
            side=self.Side,
            multiplier=self.__multiplier
        )
        return trade

    def to_json(self) -> str:
        protected_decorator = '_{}'.format(self.__class__.__name__)
        data_dict = {
            key: value.strftime('%Y-%m-%d %H:%M:%S%f') if key == '{}__trade_time'.format(protected_decorator)
            else value.value if key == '{}__side'.format(protected_decorator)
            else value
            for key, value in self.__dict__.items()
        }

        return json.dumps(data_dict)

    def copy(self, **kwargs):
        new_trade = self.__class__(
            ticker=kwargs.pop('ticker', self.__ticker),
            side=kwargs.pop('side', self.__side),
            volume=kwargs.pop('volume', self.__volume),
            notional=kwargs.pop('notional', self.__notional),
            trade_time=kwargs.pop('trade_time', self.__trade_time),
            order_id=kwargs.pop('order_id', None),
            price=kwargs.pop('price', self.__price),
            trade_id=kwargs.pop('trade_id', f'{self.__trade_id}.copy'),
            multiplier=kwargs.pop('multiplier', self.__multiplier),
            fee=kwargs.pop('fee', self.__fee)
        )

        return new_trade

    @classmethod
    def from_json(cls, json_message: Union[str, bytes, bytearray, dict]) -> 'TradeReport':
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)
        protected_decorator = '_{}'.format(TradeReport.__name__)

        self = cls(
            ticker=json_dict.pop('{}__ticker'.format(protected_decorator)),
            side=TradeSide(json_dict.pop('{}__side'.format(protected_decorator))),
            volume=json_dict.pop('{}__volume'.format(protected_decorator)),
            notional=json_dict.pop('{}__notional'.format(protected_decorator)),
            trade_time=datetime.datetime.strptime(json_dict.pop('{}__trade_time'.format(protected_decorator)), '%Y-%m-%d %H:%M:%S%f'),
            order_id=json_dict.pop('{}__order_id'.format(protected_decorator)),
            trade_id=json_dict.pop('{}__trade_id'.format(protected_decorator))
        )

        for key, value in json_dict.items():
            setattr(self, key, value)

        return self

    @staticmethod
    def from_trade(trade_data: TradeData, order_id: str, trade_id: Optional[str] = None) -> 'TradeReport':
        report = TradeReport(
            ticker=trade_data.ticker,
            side=trade_data.side,
            volume=trade_data.volume,
            notional=trade_data.notional,
            trade_time=trade_data.trade_time,
            order_id=order_id,
            trade_id=trade_id
        )
        return report

    # noinspection PyPep8Naming
    @property
    def Ticker(self) -> str:
        return self.__ticker

    # noinspection PyPep8Naming
    @property
    def Side(self) -> TradeSide:
        return self.__side

    # noinspection PyPep8Naming
    @property
    def Volume(self) -> float:
        return self.__volume

    # noinspection PyPep8Naming
    @property
    def Notional(self) -> float:
        return self.__notional

    # noinspection PyPep8Naming
    @property
    def Price(self) -> float:
        if self.__price is not None:
            return self.__price
        elif self.__volume == 0:
            return .0
        else:
            return self.__notional / self.__volume / self.__multiplier

    # noinspection PyPep8Naming
    @property
    def TradeTime(self) -> datetime.datetime:
        return self.__trade_time

    # noinspection PyPep8Naming
    @property
    def OrderID(self) -> str:
        return self.__order_id

    # noinspection PyPep8Naming
    @property
    def TradeID(self) -> str:
        return self.__trade_id

    # noinspection PyPep8Naming
    @property
    def multiplier(self) -> float:
        return self.__multiplier

    @property
    def fee(self) -> float:
        return self.__fee


class TradeInstruction(object):
    def __init__(
            self, *,
            ticker: str,
            side: Union[int, float, str, TradeSide],
            order_type: OrderType = OrderType.Manual,
            volume: float = 0.0,
            limit_price: Optional[float] = None,
            order_id: Optional[str] = None,
            multiplier: float = 1
    ):
        if volume <= 0:
            raise Exception('Invalid trade volume!')

        self.__ticker = str(ticker)
        self.__side = TradeSide(side)
        self.__order_type = order_type
        self.__volume = float(volume)
        self.__limit_price = limit_price
        self.__order_id = str(order_id) if order_id is not None else str(uuid.uuid4())
        self.__multiplier = float(multiplier)

        self.__order_state: OrderState = OrderState.Pending
        self.__filled_volume: float = 0.0
        self.__filled_notional: float = 0.0
        self.__fee = .0
        self.__start_datetime: Optional[datetime.datetime] = None
        self.__cancel_datetime: Optional[datetime.datetime] = None
        self.__finish_datetime: Optional[datetime.datetime] = None
        self.__trades: Dict[str, TradeReport] = {}

    def __repr__(self):
        return '<TradeInstruction>{}'.format(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, TradeInstruction):
            return self.__repr__() == other.__repr__()
        else:
            return False

    def __str__(self):
        if self.LimitPrice is None or self.__order_type == OrderType.MarketOrder:
            return '<TradeInstruction>({} OrderID {} {} {} {:.2f} filled {:.2f} @ {:.2f} now {})'.format(self.OrderType.name, self.OrderID, self.Side.name, self.Ticker, self.Volume, self.__filled_volume, self.AveragePrice, self.OrderState.name)
        else:
            return '<TradeInstruction>({} OrderID {} {} {} {:.2f} limit {:.2f} filled {:.2f} @ {:.2f} now {})'.format(self.OrderType.name, self.OrderID, self.Side.name, self.Ticker, self.Volume, self.LimitPrice, self.__filled_volume, self.AveragePrice, self.OrderState.name)

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    def reset(self):
        self.__trades = {}

        self.__order_state: OrderState = OrderState.Pending
        self.__filled_volume: float = 0.0
        self.__filled_notional: float = 0.0
        self.__fee = .0
        self.__start_datetime: Optional[datetime.datetime] = None
        self.__cancel_datetime: Optional[datetime.datetime] = None
        self.__finish_datetime: Optional[datetime.datetime] = None
        self.__trades: Dict[str, TradeReport] = {}

    def reset_order_id(self, order_id: Optional[str] = None, **kwargs) -> 'TradeInstruction':
        """
        reset order id to given string
        :param order_id:
        :return:
        """
        if not kwargs.pop('_ignore_warning', False):
            LOGGER.warning('TradeInstruction OrderID being reset manually! Position.reset_order_id() is the recommended method to do so.')

        if order_id is not None:
            self.__order_id = str(order_id)
        else:
            self.__order_id = str(uuid.uuid4())

        for trade_report in self.__trades.values():
            trade_report.reset_order_id(order_id=self.__order_id, _ignore_warning=True)

        return self

    def set_order_state(self, order_state: OrderState, market_datetime: datetime.datetime = datetime.datetime.utcnow()) -> 'TradeInstruction':
        self.__order_state = order_state

        # assign a start_datetime if order placed
        if order_state == OrderState.Placed:
            self.__start_datetime: Optional[datetime.datetime] = market_datetime

        if order_state == OrderState.Canceled or OrderState == OrderState.PartCanceled:
            self.__cancel_datetime: Optional[datetime.datetime] = market_datetime
            self.__finish_datetime: Optional[datetime.datetime] = market_datetime

        return self

    def fill(self, trade_report: TradeReport) -> 'TradeInstruction':
        if trade_report.OrderID != self.OrderID:
            LOGGER.warning('Order ID not match! Instruction ID {}; Report ID {}'.format(self.OrderID, trade_report.OrderID))
            return self

        if trade_report.TradeID in self.__trades:
            LOGGER.warning('Duplicated trade received! Instruction {}; Report {}'.format(str(self), str(trade_report)))
            return self

        if trade_report.Volume != 0:
            # update multiplier
            if len(self.__trades) > 0:
                assert self.__multiplier == trade_report.multiplier, 'Multiplier not match!'
            else:
                self.__multiplier = trade_report.multiplier

            if trade_report.Volume + self.__filled_volume > self.__volume:
                LOGGER.warning('Fatal error!\nTradeInstruction: \n\t{}\nTradeReport:\n\t{}'.format(str(TradeInstruction), '\n\t'.join([str(x) for x in self.__trades.values()])))
                raise Exception('Fatal error! trade reports filled volume exceed order volume!')

            self.__filled_volume += abs(trade_report.Volume)
            self.__filled_notional += abs(trade_report.Notional)

        if self.__filled_volume == self.__volume:
            self.set_order_state(OrderState.Filled)
            self.__finish_datetime = trade_report.TradeTime
        elif self.__filled_volume > 0:
            self.set_order_state(OrderState.PartFilled)

        self.__trades[trade_report.TradeID] = trade_report

        return self

    def cancel_order(self) -> 'TradeInstruction':
        self.set_order_state(OrderState.Canceling)

        cancel_instruction = copy.copy(self)
        cancel_instruction.__order_type = OrderType.CancelOrder

        return cancel_instruction

    def canceled(self, canceled_datetime: datetime.datetime) -> 'TradeInstruction':
        LOGGER.warning(DeprecationWarning('[canceled] depreciated! Use [set_order_state] instead!'), stacklevel=2)

        self.set_order_state(OrderState.Canceled, canceled_datetime)
        return self

    def to_json(self, with_trade=True) -> str:
        protected_decorator = '_{}'.format(self.__class__.__name__)
        data_dict = {
            key: value.value if key == '{}__side'.format(protected_decorator)
            else value.value if key == '{}__order_type'.format(protected_decorator)
            else value.value if key == '{}__order_state'.format(protected_decorator)
            else value.strftime('%Y-%m-%d %H:%M:%S%f') if key == '{}__start_datetime'.format(protected_decorator) and getattr(self, '{}__start_datetime'.format(protected_decorator)) is not None
            else value.strftime('%Y-%m-%d %H:%M:%S%f') if key == '{}__cancel_datetime'.format(protected_decorator) and getattr(self, '{}__cancel_datetime'.format(protected_decorator)) is not None
            else value.strftime('%Y-%m-%d %H:%M:%S%f') if key == '{}__finish_datetime'.format(protected_decorator) and getattr(self, '{}__finish_datetime'.format(protected_decorator)) is not None
            else {trade_id: value[trade_id].to_json() for trade_id in value} if key == '{}__trades'.format(protected_decorator) and with_trade
            else None if key == '{}__trades'.format(protected_decorator) and not with_trade
            else value
            for key, value in self.__dict__.items()
        }

        return json.dumps(data_dict)

    @classmethod
    def from_json(cls, json_message: Union[str, bytes, bytearray, dict]) -> 'TradeInstruction':
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        protected_decorator = '_{}'.format(TradeInstruction.__name__)

        self = cls(
            ticker=json_dict.pop('{}__ticker'.format(protected_decorator)),
            side=TradeSide(json_dict.pop('{}__side'.format(protected_decorator))),
            order_type=OrderType(json_dict.pop('{}__order_type'.format(protected_decorator))),
            volume=json_dict.pop('{}__volume'.format(protected_decorator)),
            limit_price=json_dict.pop('{}__limit_price'.format(protected_decorator)),
            order_id=json_dict.pop('{}__order_id'.format(protected_decorator))
        )

        setattr(self, '{}__order_state'.format(protected_decorator), OrderState(json_dict.pop('{}__order_state'.format(protected_decorator))))

        trade_records = json_dict.pop('{}__trades'.format(protected_decorator))
        if trade_records is None:
            setattr(self, '{}__trades'.format(protected_decorator), {})
        else:
            setattr(self, '{}__trades'.format(protected_decorator), {trade_id: TradeReport.from_json(trade_records[trade_id]) for trade_id in trade_records})

        for key, value in json_dict.items():
            setattr(self, key, value)

        if getattr(self, '{}__start_datetime'.format(protected_decorator)) is not None:
            setattr(self, '{}__start_datetime'.format(protected_decorator), datetime.datetime.strptime(getattr(self, '{}__start_datetime'.format(protected_decorator)), '%Y-%m-%d %H:%M:%S%f'))

        if getattr(self, '{}__cancel_datetime'.format(protected_decorator)) is not None:
            setattr(self, '{}__cancel_datetime'.format(protected_decorator), datetime.datetime.strptime(getattr(self, '{}__cancel_datetime'.format(protected_decorator)), '%Y-%m-%d %H:%M:%S%f'))

        if getattr(self, '{}__finish_datetime'.format(protected_decorator)) is not None:
            setattr(self, '{}__finish_datetime'.format(protected_decorator), datetime.datetime.strptime(getattr(self, '{}__finish_datetime'.format(protected_decorator)), '%Y-%m-%d %H:%M:%S%f'))

        return self

    # noinspection PyPep8Naming
    @property
    def OrderID(self) -> str:
        return self.__order_id

    # noinspection PyPep8Naming
    @property
    def Ticker(self) -> str:
        return self.__ticker

    # noinspection PyPep8Naming
    @property
    def Side(self) -> TradeSide:
        return self.__side

    # noinspection PyPep8Naming
    @property
    def OrderType(self) -> OrderType:
        return self.__order_type

    # noinspection PyPep8Naming
    @property
    def Volume(self) -> float:
        return self.__volume

    # noinspection PyPep8Naming
    @property
    def LimitPrice(self) -> Optional[float]:
        return self.__limit_price

    # noinspection PyPep8Naming
    @property
    def StartTime(self) -> Optional[datetime.datetime]:
        return self.__start_datetime

    # noinspection PyPep8Naming
    @property
    def CancelTime(self) -> Optional[datetime.datetime]:
        return self.__cancel_datetime

    # noinspection PyPep8Naming
    @property
    def FinishTime(self) -> Optional[datetime.datetime]:
        return self.__finish_datetime

    # noinspection PyPep8Naming
    @property
    def OrderState(self) -> OrderState:
        return self.__order_state

    # noinspection PyPep8Naming
    @property
    def FilledVolume(self) -> float:
        return self.__filled_volume

    # noinspection PyPep8Naming
    @property
    def WorkingVolume(self) -> float:
        return self.__volume - self.__filled_volume

    # noinspection PyPep8Naming
    @property
    def FilledNotional(self) -> float:
        return self.__filled_notional

    # noinspection PyPep8Naming
    @property
    def AveragePrice(self) -> float:
        if self.__filled_volume != 0:
            return self.__filled_notional / self.__filled_volume / self.__multiplier
        else:
            return float('NaN')

    # noinspection PyPep8Naming
    @property
    def Trades(self) -> Dict[str, TradeReport]:
        return self.__trades

    # noinspection PyPep8Naming
    @property
    def Multiplier(self) -> float:
        return self.__multiplier

    @property
    def fee(self):
        return self.__fee

    @fee.setter
    def fee(self, value):
        self.__fee = float(value)
