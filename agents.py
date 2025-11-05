from mesa import Agent
import numpy as np


class BaseFirm(Agent):
    """
    Agent representing a firm in a multi-tier supply chain.
    Can act as Supplier, Plant, DC, or Retailer.
    """

    def __init__(self, unique_id, model, tier, base_stock=50, capacity=10, lead_time=1):
        super().__init__(unique_id, model)
        self.tier = tier

        # --- State variables ---
        self.inventory = base_stock
        self.base_stock = base_stock
        self.capacity = capacity
        self.available_capacity = capacity
        self.lead_time = lead_time
        self.on_order = 0
        self.backlog = 0
        self.in_transit = []  # list of tuples (qty, remaining_time)
        self.recovery_timer = 0  # for post-disruption recovery

        # --- KPI tracking ---
        self.total_demand = 0
        self.fulfilled_demand = 0
        self.holding_cost = 0
        self.backlog_cost = 0
        self.order_history = []
        self.recovery_steps = []

        # --- Random generator ---
        if hasattr(self.model, "_seed") and self.model._seed is not None:
            self.np_random = np.random.default_rng(self.model._seed + unique_id)
        else:
            self.np_random = np.random.default_rng()

    # ---------------------------------------------------------------------
    # Step methods
    # ---------------------------------------------------------------------

    def step_order(self):
        """
        Retailer generates demand and computes orders based on base-stock policy.
        Other tiers just compute base-stock replenishment.
        """
        if self.tier == "retailer":
            demand = self.np_random.poisson(5)
            self.total_demand += demand

            # Fulfill as much demand as possible
            fulfilled = min(self.inventory, demand)
            self.inventory -= fulfilled
            unmet = demand - fulfilled
            self.backlog += unmet
            self.fulfilled_demand += fulfilled

        # Base-stock policy
        net_inventory = self.inventory + self.on_order - self.backlog
        order_qty = max(0, self.base_stock - net_inventory)

        # Record order and place it
        self.order_history.append(order_qty)
        self.on_order += order_qty
        self.model.place_order(self, order_qty)

        return order_qty  # optional, useful for logging

    def step_receive(self):
        """
        Update inventory from in_transit shipments.
        Update on_order and compute holding/backlog costs.
        """
        arrived = 0
        new_transit = []

        for qty, t in self.in_transit:
            if t <= 1:
                arrived += qty
            else:
                new_transit.append((qty, t - 1))

        self.in_transit = new_transit

        if arrived > 0:
            self.inventory += arrived
            self.on_order = max(0, self.on_order - arrived)

        # --- Compute costs ---
        h = 1  # holding cost per unit
        p = 5  # backlog cost per unit
        self.holding_cost += max(self.inventory, 0) * h
        self.backlog_cost += max(self.backlog, 0) * p

    def step_produce(self):
        """
        Produce goods if supplier or plant.
        Limited by available_capacity.
        """
        if self.tier in ("supplier", "plant") and self.available_capacity > 0:
            # Produce up to available capacity
            produced = min(self.capacity, self.available_capacity)
            self.inventory += produced

    def step_recover(self):
        """
        Handles recovery after disruption.
        When recovery_timer hits 0, restore full capacity and log recovery step.
        """
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.available_capacity = self.capacity
                # log recovery step for later analysis
                self.recovery_steps.append(self.model.time)

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    def receive_shipment(self, qty, lead_time=None):
        """
        Add incoming shipment to in_transit list.
        """
        if lead_time is None:
            lead_time = self.lead_time
        self.in_transit.append((qty, lead_time))

    def __repr__(self):
        return (
            f"<Firm {self.unique_id} | Tier: {self.tier} | "
            f"Inv: {self.inventory} | Backlog: {self.backlog} | On order: {self.on_order}>"
        )
