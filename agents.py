# agents.py

from mesa import Agent
import numpy as np

class BaseFirm(Agent):
    """
    Agent is stimulating a firm in a multi-tier supply chain.
    Agent can be Supplier, Plant, DC, Retailer.
    """
    def __init__(self, unique_id, model, tier, base_stock=50, capacity=10, lead_time=1):
        super().__init__(unique_id, model)
        self.tier = tier
        self.inventory = base_stock
        self.base_stock = base_stock
        self.capacity = capacity
        self.available_capacity = capacity  # can be reduce in disruption
        self.lead_time = lead_time
        self.on_order = 0
        self.backlog = 0
        self.in_transit = []  # [(qty, remaining_time)]
        self.recovery_timer = 0  # count recovery time after disruption

        # KPI tracking
        self.total_demand = 0           # total demand (retailer)
        self.fulfilled_demand = 0       # fulfilled_demand
        self.holding_cost = 0           # holding inventory cost
        self.backlog_cost = 0           # backlog cost
        self.order_history = []         # save order amount each step
        self.recovery_steps = []        # save steps after recover

        # ---------------------------
        # NumPy random generator using the model's seed for reproducibility
        # ---------------------------
        # If the model has a defined seed (_seed is not None),
        # initialize a separate RNG (Random Number Generator) for this agent
        # using (model_seed + unique_id). This ensures that:
        #   1) Each agent has its own independent random stream.
        #   2) Results are reproducible when running with the same seed.
        # If no seed is defined in the model, use a fully random generator.
        if hasattr(self.model, "_seed") and self.model._seed is not None:
            self.np_random = np.random.default_rng(self.model._seed + unique_id)
        else:
            self.np_random = np.random.default_rng()

    # ---------------------------
    # Step methods
    # ---------------------------
    def step_order(self):
        """
        Retailer demand ~ Poisson; order = base-stock - current inventory.
        """
        if self.tier == "retailer":
            demand = self.np_random.poisson(5) 
            self.total_demand += demand

            # fulfill demand by inventory (if enough)
            fulfilled = min(self.inventory, demand)
            self.inventory -= fulfilled
            unmet = demand - fulfilled
            self.backlog += unmet
            self.fulfilled_demand += fulfilled

        # Calculate order from base-stock
        net_inventory = self.inventory + self.on_order - self.backlog
        order_qty = max(0, self.base_stock - net_inventory)
        self.order_history.append(order_qty)

        # place
        self.model.place_order(self, order_qty)

    def step_receive(self):
        """
        Update inventory from in_transit shipments
        Compute holding and backlog costs
        """
        arrived = 0
        new_transit = []
        for qty, t in self.in_transit:
            if t <= 1:
                arrived += qty
            else:
                new_transit.append((qty, t-1))
        self.in_transit = new_transit
        if arrived > 0:
            self.inventory += arrived
            self.on_order -= arrived

        
        h = 1  # cost per unit inventory
        p = 5  # cost per unit backlog
        self.holding_cost += self.inventory * h
        self.backlog_cost += self.backlog * p

    def step_produce(self):
        """
        Production if tier is supplier or plant.
        """
        if self.tier in ("supplier", "plant") and self.available_capacity > 0:
            produced = min(self.available_capacity, self.capacity)
            self.inventory += produced

    def step_recover(self):
        """
        Recover after disruption
        """
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.available_capacity = self.capacity
                # reset lead_time if temporary increased
