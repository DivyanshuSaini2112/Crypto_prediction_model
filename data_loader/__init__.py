from .CoinMarketDataset import CoinMarketDataset
from .Bitmex import BitmexDataset
from .BinanceDataset import BinanceDataset
from datetime import datetime
import pandas as pd
import numpy as np

DATASETS = ['CoinMarket', 'Bitmex', 'Binance']
DATA_TYPES = ['train', 'validation', 'test']


def get_dataset(dataset_name, start_date, end_date, args):
    assert dataset_name in DATASETS, \
        f"Dataset '{dataset_name}' is not available. Choose from: {DATASETS}"

    if dataset_name == 'CoinMarket':
        main_features = ['High', 'Volume', 'Low', 'Close', 'Open', 'Mean']

        if start_date == "-1":
            start_date = None
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')

        if end_date == "-1":
            start_date = None
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        btc = CoinMarketDataset(main_features=main_features, start_date=start_date,
                                end_date=end_date, window_size=args.dataset_loader.window_size)
        dataset, profit_calculator = btc.get_dataset()

    elif dataset_name == 'Bitmex':
        btc = BitmexDataset(args)
        dataset, profit_calculator = btc.get_dataset()

    elif dataset_name == 'Binance':
        # Live data from Binance — works in India, no API key needed
        loader = BinanceDataset(args)
        dataset, profit_calculator = loader.get_dataset()

    return dataset, profit_calculator

