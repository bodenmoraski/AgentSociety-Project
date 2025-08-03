"""
Core Agent Classes for Economic Experiments
"""

import random
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class EconomicAgent:
    # Identity
    id: int
    name: str

    # Economic state
    wealth: float
    goods: Dict[str, int]  # e.g., {"food": 10, "tools": 2}
    production_goods: List[str]  # e.g., ["food"]
    good_preferences: Dict[str, int]  # desired quantities
    good_values: Dict[str, float]  # marginal value per good
    consumption_needs: Dict[str, int]  # consumed per time step

    # Personality traits
    cooperativeness: float  # 0-1, willingness/skill to trade/negotiate
    innovation: float       # 0-1, likelihood to change strategy

    # Social traits
    network_centrality: float  # 0-1
    influence_power: float     # 0-1 (not used for trade yet)

    # Dynamic state
    trade_history: List[Tuple[int, str, int, float]] = field(default_factory=list)  # (partner_id, good, quantity, price)
    wealth_history: List[float] = field(default_factory=list)
    production_history: Dict[str, List[int]] = field(default_factory=dict)
    production_requirements: Dict[str, Dict[str, int]] = field(default_factory=dict)
    received_bids: List[Dict] = field(default_factory=list)  # Stores bids received in this time step

    def __post_init__(self):
        self.wealth_history.append(self.wealth)
        for good in self.production_goods:
            self.production_history[good] = []

    def produce(self):
        """Produce goods according to production_goods list and requirements."""
        for good in self.production_goods:
            requirements = self.production_requirements.get(good, {})
            # Check if agent has enough input goods
            can_produce = all(self.goods.get(input_good, 0) >= qty for input_good, qty in requirements.items())
            if can_produce:
                # Consume input goods
                for input_good, qty in requirements.items():
                    self.goods[input_good] -= qty
                produced = 1  # Produce 1 unit per time step
                self.goods[good] = self.goods.get(good, 0) + produced
                self.production_history[good].append(produced)
            else:
                self.production_history[good].append(0)  # Could not produce

    def consume(self):
        """Consume goods according to consumption_needs."""
        for good, amount in self.consumption_needs.items():
            self.goods[good] = max(0, self.goods.get(good, 0) - amount)

    def needs_to_buy(self) -> List[str]:
        """Return list of goods agent wants to buy (below preference)."""
        needs = []
        for good, desired in self.good_preferences.items():
            if self.goods.get(good, 0) < desired:
                needs.append(good)
        return needs

    def submit_bid(self, good: str, quantity: int, bid_type: str = "buy") -> Dict[str, float]:
        """
        Submit a bid for a good.
        bid_type: "buy" for purchase, "sell" for offering to sell.
        """
        value = self.good_values.get(good, 1.0)
        if bid_type == "buy":
            price = value * (0.8 + 0.4 * self.cooperativeness)
        else:  # "sell"
            price = value * (0.7 + 0.3 * self.cooperativeness)
        return {"good": good, "quantity": quantity, "price": price, "type": bid_type}

    def execute_trade(self, partner: 'EconomicAgent', good: str, quantity: int, price: float):
        """Exchange goods and money with another agent."""
        if self.goods.get(good, 0) >= quantity and partner.wealth >= price * quantity:
            # Seller: self, Buyer: partner
            self.goods[good] -= quantity
            self.wealth += price * quantity
            partner.goods[good] = partner.goods.get(good, 0) + quantity
            partner.wealth -= price * quantity
            self.trade_history.append((partner.id, good, quantity, price))
            partner.trade_history.append((self.id, good, quantity, price))
            self.wealth_history.append(self.wealth)
            partner.wealth_history.append(partner.wealth)

    def update_good_value(self, good: str, market_price: float, success: bool, adjustment_rate: float = 0.1):
        """
        Adjust perceived value for a good based on trade outcome.
        If purchase failed, increase value toward market price.
        If purchase succeeded, optionally decrease value toward market price.
        """
        current_value = self.good_values.get(good, 1.0)
        if success:
            # Lower value slightly if agent paid less than its own value
            if market_price < current_value:
                self.good_values[good] -= adjustment_rate * (current_value - market_price)
        else:
            # Raise value toward market price if agent failed to buy
            if market_price > current_value:
                self.good_values[good] += adjustment_rate * (market_price - current_value)

    # def select_best_bid(self, bids: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    #     """
    #     Given a list of bids (offers to sell), select the best one to fulfill.
    #     By default, chooses the lowest price that meets the desired quantity.
    #     """
    #     if not bids:
    #         return None
    #     # Sort bids by price (ascending), then by quantity (descending)
    #     sorted_bids = sorted(bids, key=lambda b: (b["price"], -b["quantity"]))
    #     # Optionally, filter for bids that meet a minimum quantity
    #     for bid in sorted_bids:
    #         if bid["quantity"] > 0:
    #             return bid
    #     return None

    # def clear_received_bids(self):
    #     """Clear received bids at the start of each time step."""
    #     self.received_bids.clear()

    # def receive_bid(self, bid: Dict):
    #     """Add a received bid to the agent's list."""
    #     self.received_bids.append(bid)

    def create_bids(self):
        """
        Simulate one time step for this agent:
        1. Produce goods.
        2. Consume goods.
        3. Create buy and sell bids.
        4. Send bids to neighbors.
        """
        # 1. Produce goods
        self.produce()

        # 2. Consume goods
        self.consume()

        # 3. Create buy bids
        buy_bids = []
        needs = self.needs_to_buy()
        for good in needs:
            quantity_needed = self.good_preferences[good] - self.goods.get(good, 0)
            if quantity_needed > 0:
                bid = self.submit_bid(good, quantity_needed, bid_type="buy")
                bid["buyer"] = self  # Attach reference to self for trade execution
                buy_bids.append(bid)

        # 3. Create sell bids
        sell_bids = []
        for good, amount in self.goods.items():
            surplus = amount - self.good_preferences.get(good, 0)
            if surplus > 0:
                bid = self.submit_bid(good, surplus, bid_type="sell")
                bid["seller"] = self  # Attach reference to self for trade execution
                sell_bids.append(bid)

        # 4. Send buy and sell bids to market
        return buy_bids, sell_bids
        
    # def process_received_bids(self):
    #     """
    #     Decide which received bids to fulfill (e.g., sell to buyers).
    #     You can implement logic to select the best bids, fulfill as many as possible, etc.
    #     """
    #     for bid in self.received_bids:
    #         if bid["type"] == "buy":
    #             # Check if agent can fulfill the buy bid
    #             if self.goods.get(bid["good"], 0) >= bid["quantity"] and bid["price"] >= self.good_values.get(bid["good"], 1.0) * 0.7:
    #                 # Find the buyer agent (you may need to pass a reference or id)
    #                 buyer = bid.get("buyer")
    #                 if buyer:
    #                     self.execute_trade(buyer, bid["good"], bid["quantity"], bid["price"])
    #                     self.update_good_value(bid["good"], bid["price"], success=True)
    #     self.clear_received_bids()

def load_agent_from_json(json_path: str) -> 'EconomicAgent':
    """
    Load a single agent configuration from a JSON file and create an EconomicAgent instance.
    The JSON file should be a dict containing agent attributes.
    """
    with open(json_path, 'r') as f:
        cfg = json.load(f)

    agent = EconomicAgent(
        id=cfg["id"],
        name=cfg.get("name", f"Agent{cfg['id']}"),
        wealth=cfg.get("wealth", 0.0),
        goods=cfg.get("goods", {}),
        production_goods=cfg.get("production_goods", []),
        good_preferences=cfg.get("good_preferences", {}),
        good_values=cfg.get("good_values", {}),
        consumption_needs=cfg.get("consumption_needs", {}),
        cooperativeness=cfg.get("cooperativeness", 0.5),
        innovation=cfg.get("innovation", 0.0),
        network_centrality=cfg.get("network_centrality", 0.0),
        influence_power=cfg.get("influence_power", 0.0),
        production_requirements=cfg.get("production_requirements", {})
    )
    return agent