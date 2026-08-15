"""
Unit tests for Capa 1: MarketSimulator price curves, batch execution,
and town consumption mechanics against the official Kaggriculture specification.
"""

import pytest
from models.constants import (
    BASE_MARKET_INVENTORY,
    MARKET_PARAMS,
    PRICE_FLOOR,
    ProductType,
    ShopType,
)
from engine.market_simulator import MarketSimulator, BatchSellResult, BatchBuyResult


class TestMarketFormulas:
    """Validates mathematical price curves against official reference tables."""

    @pytest.fixture
    def sim(self) -> MarketSimulator:
        return MarketSimulator()

    def test_base_equilibrium_prices(self, sim: MarketSimulator):
        """At I0 = 10,000, price must equal base price exactly for all resources."""
        for res_name, spec in MARKET_PARAMS.items():
            price = sim.compute_price(res_name, BASE_MARKET_INVENTORY)
            assert price == spec.base, f"Failed base price for {res_name}: expected {spec.base}, got {price}"

    def test_official_table_wheat(self, sim: MarketSimulator):
        # Base=25, I0=10000, T=400
        assert sim.compute_price("WHEAT", 10000 - 400) == 45
        assert sim.compute_price("WHEAT", 10000 + 400) == 20
        assert sim.compute_price("WHEAT", 10000 + 800) == 19

    def test_official_table_carrot(self, sim: MarketSimulator):
        # Base=35, I0=10000, T=450
        assert sim.compute_price("CARROT", 10000 - 450) == 42
        assert sim.compute_price("CARROT", 10000 + 450) == 10
        assert sim.compute_price("CARROT", 10000 + 900) == 1

    def test_official_table_tomato(self, sim: MarketSimulator):
        # Base=60, I0=10000, T=200
        assert sim.compute_price("TOMATO", 10000 - 200) == 84
        assert sim.compute_price("TOMATO", 10000 + 200) == 24
        assert sim.compute_price("TOMATO", 10000 + 400) == 9

    def test_official_table_strawberry(self, sim: MarketSimulator):
        # Base=120, I0=10000, T=100
        assert sim.compute_price("STRAWBERRY", 10000 - 100) == 204
        assert sim.compute_price("STRAWBERRY", 10000 + 100) == 1
        assert sim.compute_price("STRAWBERRY", 10000 + 200) == 1

    def test_official_table_melon(self, sim: MarketSimulator):
        # Base=250, I0=10000, T=300
        assert sim.compute_price("MELON", 10000 - 300) == 300
        assert sim.compute_price("MELON", 10000 + 300) == 1
        assert sim.compute_price("MELON", 10000 + 600) == 1

    def test_official_table_egg(self, sim: MarketSimulator):
        # Base=50, I0=10000, T=332
        assert sim.compute_price("EGG", 10000 - 332) == 70
        assert sim.compute_price("EGG", 10000 + 332) == 40
        assert sim.compute_price("EGG", 10000 + 664) == 39

    def test_official_table_milk(self, sim: MarketSimulator):
        # Base=160, I0=10000, T=122
        assert sim.compute_price("MILK", 10000 - 122) == 256
        assert sim.compute_price("MILK", 10000 + 122) == 1
        assert sim.compute_price("MILK", 10000 + 244) == 1

    def test_official_table_wool(self, sim: MarketSimulator):
        # Base=200, I0=10000, T=105
        assert sim.compute_price("WOOL", 10000 - 105) == 240
        assert sim.compute_price("WOOL", 10000 + 105) == 1
        assert sim.compute_price("WOOL", 10000 + 210) == 1

    def test_official_table_fertilizer(self, sim: MarketSimulator):
        # Base=100, I0=10000, T=200
        assert sim.compute_price("FERTILIZER", 10000 - 200) == 140
        assert sim.compute_price("FERTILIZER", 10000 + 200) == 60
        assert sim.compute_price("FERTILIZER", 10000 + 400) == 20


class TestBatchOrdersAndInvariants:
    """Validates order execution mechanics, price floor responsiveness, and buy/sell symmetry."""

    @pytest.fixture
    def sim(self) -> MarketSimulator:
        return MarketSimulator()

    def test_zero_net_buy_sell_invariant(self, sim: MarketSimulator):
        """Buying 1 unit followed immediately by selling 1 unit must net exactly 0."""
        initial_inv = 10000
        buy_res = sim.simulate_batch_buy("WHEAT", initial_inv, 1)
        # Cost paid is buy_res.total_cost at inv=9999
        sell_res = sim.simulate_batch_sell("WHEAT", buy_res.final_inventory, 1)
        # Revenue received is sell_res.total_revenue at pre-sell inv=9999
        assert buy_res.total_cost == sell_res.total_revenue
        assert sell_res.final_inventory == initial_inv

    def test_batch_sell_step_by_step_equivalence(self, sim: MarketSimulator):
        """Simulate_batch_sell must match manual step-by-step price summation."""
        qty = 25
        initial_inv = 9980
        batch_res = sim.simulate_batch_sell("CARROT", initial_inv, qty)

        # Manual iteration
        manual_rev = 0
        cur_inv = initial_inv
        for _ in range(qty):
            p = sim.compute_price("CARROT", cur_inv)
            manual_rev += p
            if p > PRICE_FLOOR:
                cur_inv += 1

        assert batch_res.total_revenue == manual_rev
        assert batch_res.final_inventory == cur_inv
        assert batch_res.marginal_price == sim.compute_price("CARROT", cur_inv - 1 if cur_inv > initial_inv else cur_inv)

    def test_floor_inventory_responsiveness(self, sim: MarketSimulator):
        """When price hits $1, additional sells do not increase market inventory."""
        # For Strawberry, selling past glutamate collapses price to $1 quickly
        initial_inv = 10500
        assert sim.compute_price("STRAWBERRY", initial_inv) == 1

        res = sim.simulate_batch_sell("STRAWBERRY", initial_inv, 50)
        assert res.marginal_price == 1
        assert res.total_revenue == 50  # 50 units * $1
        assert res.final_inventory == initial_inv  # inventory must NOT increase past floor!

    def test_find_optimal_sell_batch(self, sim: MarketSimulator):
        """Ensures optimal batch finder halts before price drops below threshold."""
        # Wheat at 10000 base=25. If threshold is 23, find max units.
        max_units = sim.find_optimal_sell_batch("WHEAT", 10000, 500, min_acceptable_price=23)
        res = sim.simulate_batch_sell("WHEAT", 10000, max_units)
        assert res.marginal_price >= 23
        # Next unit should fall below 23
        next_p = sim.compute_price("WHEAT", res.final_inventory)
        assert next_p <= 23


class TestTownDemandCalculations:
    """Validates town center and shop consumption rates."""

    def test_daily_town_drain(self):
        shops = ["BAKERY", "PIZZA_SHOP", "YARN_STORE"]
        drain = MarketSimulator.get_daily_town_drain(shops)

        # Town center gives +1 to all except fertilizer
        # Bakery gives: EGG +6, WHEAT +6
        # Pizza gives: MILK +6, TOMATO +6, WHEAT +6
        # Yarn store gives: WOOL +12 (2x * 6)
        assert drain["WHEAT"] == 1 + 6 + 6  # 13
        assert drain["EGG"] == 1 + 6         # 7
        assert drain["MILK"] == 1 + 6        # 7
        assert drain["TOMATO"] == 1 + 6      # 7
        assert drain["WOOL"] == 1 + 12       # 13
        assert drain["CARROT"] == 1          # 1 (only town center)
        assert drain["FERTILIZER"] == 0      # 0 (town center doesn't consume fertilizer)

    def test_market_trajectory_recovery(self):
        sim = MarketSimulator()
        shops = ["BAKERY", "BAKERY", "PIZZA_SHOP"]  # Heavy wheat demand (6+6+6 + 1 = 19/day)
        init_inv = {"WHEAT": 10200}

        projections = sim.project_market_trajectory(
            initial_inventory=init_inv,
            unlocked_shops=shops,
            days_ahead=12,
        )

        wheat_proj = projections["WHEAT"]
        assert len(wheat_proj) == 12
        # Inventory should decrease each day and price should recover towards base
        assert wheat_proj[0].inventory == 10200 - 19
        assert wheat_proj[11].inventory == 10200 - (19 * 12)
        # At day 11, inventory has drained 228 units from 10200 down to 9972 (< 10000), crossing equilibrium
        assert wheat_proj[11].price > wheat_proj[0].price
