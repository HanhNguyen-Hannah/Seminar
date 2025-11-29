# agents.py - FIXED VERSION WITH IMPROVEMENTS
from mesa import Agent
import numpy as np

class BaseFirm(Agent):
    """
    Generic firm agent for multi-tier supply chain.
    Tier in {"supplier","plant","dc","retailer"}.
    """

    def __init__(self, unique_id, model, tier,
                 base_stock=50, capacity=10, lead_time=1,
                 holding_cost_per_unit=1.0, backlog_cost_per_unit=5.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.tier = tier

        # Inventory, capacity, pipeline, backlog
        self.inventory = int(base_stock)
        self.base_stock = int(base_stock)
        self.capacity = int(capacity)
        self.available_capacity = int(capacity)
        self.lead_time = int(lead_time)

        # Pipeline: list of {"qty","remaining","from"}
        self.in_transit = []
        self.backlog = 0

        # KPIs & histories
        self.total_demand = 0
        self.fulfilled_demand = 0
        self.order_history = []
        self.holding_cost = 0.0
        self.backlog_cost = 0.0
        self.backlog_history = []
        self.demand_received_history = []

        # Disruption / recovery
        self.recovery_timer = 0
        self.is_disrupted = False  # Track disruption state

        # Store originals to revert after recovery
        self._orig_lead_time = int(lead_time)
        self._orig_capacity = int(capacity)

        # RNG consistent with model seed
        if hasattr(self.model, "_seed") and self.model._seed is not None:
            self.np_random = np.random.default_rng(self.model._seed + unique_id)
        else:
            self.np_random = np.random.default_rng()

        # Cost parameters
        self.holding_cost_per_unit = float(holding_cost_per_unit)
        self.backlog_cost_per_unit = float(backlog_cost_per_unit)

    # -------------------------
    # Per-step actions
    # -------------------------
    def reset_step_state(self):
        """
        Reset available_capacity at start of step.
        Capacity is already reduced during disruption, will be restored upon recovery.
        """
        self.available_capacity = self.capacity

    def step_order(self):
        if self.tier == "retailer":
            demand = int(self.np_random.poisson(self.model.retailer_demand_mean))
            self.total_demand += demand
            self.demand_received_history.append(demand)

        pipeline = sum(e["qty"] for e in self.in_transit)
        net_inv = self.inventory + pipeline - self.backlog
        
        # Demand forecast
        if len(self.order_history) >= 3:
            recent_demand_proxy = np.mean(self.order_history[-3:])
        else:
            recent_demand_proxy = self.model.retailer_demand_mean  # Use model param
        
        # FIX: Bound the recent_demand_proxy to prevent runaway
        recent_demand_proxy = min(recent_demand_proxy, self.base_stock * 2)
        
        # Safety stock inflation (bounded)
        safety_factor = 1.0
        if self.backlog > 0:
            safety_factor = 1.0 + min(0.3, self.backlog / max(1, self.base_stock))
        
        # Inventory adjustment (bounded)
        inv_deficit = self.base_stock - net_inv
        inv_deficit = max(-self.base_stock, min(self.base_stock * 2, inv_deficit))  # BOUND
        
        overreaction = 1.2  # Reduced from 1.4
        
        # Pipeline adjustment (bounded)
        desired_pipeline = recent_demand_proxy * (self.lead_time + 1) * safety_factor
        pipeline_gap = desired_pipeline - pipeline
        pipeline_gap = max(-self.base_stock, min(self.base_stock, pipeline_gap))  # BOUND
        
        # Combined order
        base_order = inv_deficit * overreaction
        pipeline_order = pipeline_gap * 0.4  # Reduced from 0.6
        order_qty = max(0, int(base_order + pipeline_order))
        
        # FIX: HARD UPPER BOUND - no order > 3x base_stock
        order_qty = min(order_qty, self.base_stock * 3)
        
        # Noise (reduced)
        noise_factor = 1.0 + self.np_random.uniform(-0.1, 0.1)
        order_qty = max(0, int(order_qty * noise_factor))
        
        self.order_history.append(order_qty)
        return order_qty

    def step_receive(self):
        """
        Process in_transit arrivals (decrement remaining).
        Arrivals reduce backlog first, then increase inventory.
        RETAILERS: Also fulfill customer demand from inventory here.
        """
        # Process arrivals
        arrived = 0
        new_transit = []
        for e in self.in_transit:
            if e["remaining"] <= 1:
                arrived += e["qty"]
            else:
                new_transit.append({
                    "qty": e["qty"], 
                    "remaining": e["remaining"] - 1, 
                    "from": e.get("from")
                })
        self.in_transit = new_transit

        # Settle arrivals: backlog first, then inventory
        if arrived > 0:
            if self.backlog > 0:
                settle = min(self.backlog, arrived)
                self.backlog -= settle
                arrived -= settle
                if self.tier == "retailer":
                    self.fulfilled_demand += settle
            if arrived > 0:
                self.inventory += arrived

        # RETAILERS: Fulfill current customer demand from inventory
        if self.tier == "retailer":
            unfulfilled = self.total_demand - self.fulfilled_demand
            if unfulfilled > 0 and self.inventory > 0:
                can_fulfill = min(self.inventory, unfulfilled)
                self.inventory -= can_fulfill
                self.fulfilled_demand += can_fulfill
                remaining_unmet = unfulfilled - can_fulfill
                if remaining_unmet > 0:
                    self.backlog += remaining_unmet

        # Cost bookkeeping
        self.holding_cost += max(0, self.inventory) * self.holding_cost_per_unit
        self.backlog_cost += max(0, self.backlog) * self.backlog_cost_per_unit
        self.backlog_history.append(self.backlog)

    def step_produce(self):
        """
        Only SUPPLIERS produce from raw materials.
        Plants transform via order-receive-ship cycle (no explicit production).
        """
        if self.tier == "supplier" and self.available_capacity > 0:
            pipeline = sum(e["qty"] for e in self.in_transit)
            net_inv = self.inventory + pipeline - self.backlog
            desired = max(0, int(self.base_stock - net_inv))
            produce_qty = min(self.available_capacity, desired)
            if produce_qty > 0:
                self.inventory += produce_qty
                self.available_capacity -= produce_qty

    def step_recover(self):
        """
        Decrement recovery timer. When done, notify model.
        """
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.is_disrupted = False
                try:
                    self.model._agent_recovered(self)
                except Exception as e:
                    print(f"Warning: recovery callback failed: {e}")

    # -------------------------
    # Utilities
    # -------------------------
    def receive_shipment(self, qty, lead_time=None, from_uid=None):
        """Create an in_transit entry for incoming shipment."""
        if qty <= 0:
            return
        lt = self.lead_time if lead_time is None else int(lead_time)
        self.in_transit.append({
            "qty": int(qty), 
            "remaining": int(lt), 
            "from": from_uid
        })

    def __repr__(self):
        pipeline = sum(e["qty"] for e in self.in_transit)
        return f"<Firm {self.unique_id} {self.tier} Inv:{self.inventory} Bk:{self.backlog} Pipe:{pipeline}>"
    
    def record_demand_received(self, qty):
        """Record demand received from downstream customers."""
        self.demand_received_history.append(qty)