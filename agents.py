# agents.py
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
        self.capacity = int(capacity)                # nominal capacity per step
        self.available_capacity = int(capacity)      # reset each step
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
        self.backlog_history = []  # 1 if backlog>0 at step, else 0

        # Disruption / recovery
        self.recovery_timer = 0
        self.recovery_ramp_fraction = 1.0

        # store originals to revert after temporary changes
        self._orig_lead_time = int(lead_time)
        self._orig_capacity = int(capacity)

        # RNG consistent with model seed
        if hasattr(self.model, "_seed") and self.model._seed is not None:
            self.np_random = np.random.default_rng(self.model._seed + unique_id)
        else:
            self.np_random = np.random.default_rng()

        # cost parameters
        self.holding_cost_per_unit = float(holding_cost_per_unit)
        self.backlog_cost_per_unit = float(backlog_cost_per_unit)

    # -------------------------
    # Per-step actions (called by model)
    # -------------------------
    def reset_step_state(self):
        """
        Reset available_capacity at start of step.
        During recovery, apply ramp fraction (<1).
        """
        if self.recovery_timer > 0:
            self.available_capacity = max(0, int(self.capacity * self.recovery_ramp_fraction))
        else:
            self.available_capacity = self.capacity

    def step_order(self):
        """
        Retailer generates demand; all agents compute order-up-to quantity.
        Return order_qty.
        """
        # Retailer demand
        if self.tier == "retailer":
            demand = int(self.np_random.poisson(self.model.retailer_demand_mean))
            self.total_demand += demand
            # immediate fulfillment
            fulfilled = min(self.inventory, demand)
            self.inventory -= fulfilled
            unmet = demand - fulfilled
            self.backlog += unmet
            self.fulfilled_demand += fulfilled

        # compute pipeline (on-order)
        pipeline = sum(e["qty"] for e in self.in_transit)
        net_inv = self.inventory + pipeline - self.backlog
        order_qty = max(0, int(self.base_stock - net_inv))
        self.order_history.append(order_qty)
        return order_qty

    def step_receive(self):
        """
        Process in_transit arrivals (decrement remaining).
        Arrivals reduce backlog first, then increase inventory.
        Update costs and backlog_history.
        """
        arrived = 0
        new_transit = []
        for e in self.in_transit:
            if e["remaining"] <= 1:
                arrived += e["qty"]
            else:
                new_transit.append({"qty": e["qty"], "remaining": e["remaining"] - 1, "from": e.get("from")})
        self.in_transit = new_transit

        if arrived > 0:
            # satisfy backlog first
            if self.backlog > 0:
                settle = min(self.backlog, arrived)
                self.backlog -= settle
                arrived -= settle
                if self.tier == "retailer":
                    self.fulfilled_demand += settle
            if arrived > 0:
                self.inventory += arrived

        # costs bookkeeping
        self.holding_cost += max(0, self.inventory) * self.holding_cost_per_unit
        self.backlog_cost += max(0, self.backlog) * self.backlog_cost_per_unit

        # record backlog presence
        self.backlog_history.append(1 if self.backlog > 0 else 0)

    def step_produce(self):
        """
        Produce up to available_capacity, but target production to reach base_stock.
        This prevents unbounded overproduction.
        """
        if self.tier in ("supplier", "plant") and self.available_capacity > 0:
            # compute desired production to move inventory towards base_stock (consider pipeline and backlog)
            pipeline = sum(e["qty"] for e in self.in_transit)
            net_inv = self.inventory + pipeline - self.backlog
            desired = max(0, int(self.base_stock - net_inv))
            produce_qty = min(self.available_capacity, desired)
            if produce_qty > 0:
                self.inventory += produce_qty
                # consume available_capacity for this step
                self.available_capacity -= produce_qty

    def step_recover(self):
        """
        Decrement recovery timer. When recovery done, inform model via callback.
        """
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            # optional ramp-up logic could be here (e.g., increase ramp fraction gradually)
            if self.recovery_timer == 0:
                # when finished, reset ramp fraction to full and let model revert other attributes
                self.recovery_ramp_fraction = 1.0
                # notify model about recovery
                try:
                    self.model._agent_recovered(self)
                except Exception:
                    pass

    # -------------------------
    # Utilities
    # -------------------------
    def receive_shipment(self, qty, lead_time=None, from_uid=None):
        """
        Create an in_transit entry for incoming shipment.
        """
        if qty <= 0:
            return
        lt = self.lead_time if lead_time is None else int(lead_time)
        self.in_transit.append({"qty": int(qty), "remaining": int(lt), "from": from_uid})

    def __repr__(self):
        pipeline = sum(e["qty"] for e in self.in_transit)
        return f"<Firm {self.unique_id} {self.tier} Inv:{self.inventory} Bk:{self.backlog} Pipe:{pipeline}>"
