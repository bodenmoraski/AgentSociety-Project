from typing import List, Dict
from collections import defaultdict

class Market:
    def __init__(self, agents: List):
        self.agents = agents

    def collect_bids(self):
        """
        Collect buy and sell bids from all agents.
        Each agent should have a method to generate its buy and sell bids.
        Returns:
            buy_bids: List of buy bid dicts
            sell_bids: List of sell bid dicts
        """
        buy_bids = []
        sell_bids = []
        for agent in self.agents:
            buy, sell = agent.create_bids()
            buy_bids.extend(buy)
            sell_bids.extend(sell)
        return buy_bids, sell_bids

    def match_and_execute(self, buy_bids: List[Dict], sell_bids: List[Dict]):
        """
        Match buy and sell bids and execute trades.
        Unmatched bids expire after each round.
        Agents update good values in response to expired bids.
        """
        # Track which bids were fulfilled
        fulfilled_buy = set()
        fulfilled_sell = set()

        goods = set([bid["good"] for bid in buy_bids] + [bid["good"] for bid in sell_bids])
        for good in goods:
            good_buy_bids = [b for b in buy_bids if b["good"] == good]
            good_sell_bids = [s for s in sell_bids if s["good"] == good]
            good_buy_bids.sort(key=lambda b: b["price"], reverse=True)
            good_sell_bids.sort(key=lambda s: s["price"])

            i, j = 0, 0
            while i < len(good_buy_bids) and j < len(good_sell_bids):
                buy = good_buy_bids[i]
                sell = good_sell_bids[j]
                if buy["price"] >= sell["price"]:
                    trade_qty = min(buy["quantity"], sell["quantity"])
                    buyer = buy.get("buyer")
                    seller = sell.get("seller")
                    if buyer and seller:
                        seller.execute_trade(buyer, good, trade_qty, sell["price"])
                        buyer.update_good_value(good, sell["price"], success=True)
                        seller.update_good_value(good, sell["price"], success=True)
                        fulfilled_buy.add(id(buy))
                        fulfilled_sell.add(id(sell))
                    buy["quantity"] -= trade_qty
                    sell["quantity"] -= trade_qty
                    if buy["quantity"] == 0: i += 1
                    if sell["quantity"] == 0: j += 1
                else:
                    break

        # Expire unmatched buy bids
        for bid in buy_bids:
            if id(bid) not in fulfilled_buy:
                buyer = bid.get("buyer")
                if buyer:
                    buyer.update_good_value(bid["good"], bid["price"], success=False)

        # Expire unmatched sell bids
        for bid in sell_bids:
            if id(bid) not in fulfilled_sell:
                seller = bid.get("seller")
                if seller:
                    seller.update_good_value(bid["good"], bid["price"], success=False)

    def step(self):
        """
        Run one market clearing step:
        1. Collect bids from agents.
        2. Match and execute trades.
        """
        buy_bids, sell_bids = self.collect_bids()
        self.match_and_execute(buy_bids, sell_bids)
