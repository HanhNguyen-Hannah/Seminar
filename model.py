from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
from agents import BaseFirm

class MultiTierModel(Model):
    """
    Multi-tier supply chain model with suppliers, plants, DCs, and retailers.
    Supports disruption scenarios and KPI computation.
    """

    def __init__(self, assumptions=None, seed=None, dual_sourcing=False,
                 disruption_at_step=10, scenario="capacity_only"):
        super().__init__(seed=seed)
        self.schedule = SimultaneousActivation(self)
        self.G = nx.DiGraph()
        self.dual_sourcing = dual_sourcing
        self.time = 0
        self._seed = seed
        self.scenario = scenario

        # --- Default assumptions ---
        if assumptions is None:
            assumptions = {}

        self.n_suppliers = assumptions.get("n_suppliers", 3)
        self.n_plants = assumptions.get("n_plants", 3)
        self.n_dcs = assumptions.get("n_dcs", 2)
        self.n_retailers = assumptions.get("n_retailers", 5)

        self.base_stock = assumptions.get(
            "base_stock", {"supplier": 100, "plant": 80, "dc": 60, "retailer": 30}
        )
        self.capacity = assumptions.get(
            "capacity", {"supplier": 20, "plant": 15, "dc": 10, "retailer": 0}
        )
        self.lead_time = assumptions.get(
            "lead_time", {"supplier": 2, "plant": 5, "dc": 5, "retailer": 2}
        )
        self.capacity_loss_frac = assumptions.get("capacity_loss_frac", 0.5)
        self.recovery_duration = assumptions.get("recovery_duration", 5)
        self.holding_cost = assumptions.get("holding_cost", 1)
        self.backlog_cost = assumptions.get("backlog_cost", 5)
        self.retailer_demand_mean = assumptions.get("retailer_demand_mean", 5)

        # --- Disruption tracking ---
        self.disruption_at_step = disruption_at_step
        self.disruption_occurred = False
        self.disruption_step = None
        self.recovery_step = None
        self.fill_rate_before_disruption = None

        # --- Create agents ---
        self.agents = []
        uid = 0
        self.suppliers = self._create_agents("supplier", self.n_suppliers, uid)
        uid += self.n_suppliers
        self.plants = self._create_agents("plant", self.n_plants, uid)
        uid += self.n_plants
        self.dcs = self._create_agents("dc", self.n_dcs, uid)
        uid += self.n_dcs
        self.retailers = self._create_agents("retailer", self.n_retailers, uid)

        # --- Build network ---
        self._build_network()

        # --- DataCollector ---
        self.datacollector = DataCollector(
            model_reporters={
                "time": lambda m: m.time,
                "fill_rate": lambda m: m.compute_fill_rate(),
                "total_cost": lambda m: m.compute_total_cost(),
            },
            agent_reporters={
                "inventory": "inventory",
                "backlog": "backlog",
                "holding_cost": "holding_cost",
                "backlog_cost": "backlog_cost",
            },
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    def _create_agents(self, tier, n, start_uid):
        agents = []
        for i in range(n):
            a = BaseFirm(
                unique_id=start_uid + i,
                model=self,
                tier=tier,
                base_stock=self.base_stock[tier],
                capacity=self.capacity[tier],
                lead_time=self.lead_time[tier],
                holding_cost_per_unit=self.holding_cost,
                backlog_cost_per_unit=self.backlog_cost
            )
            self.schedule.add(a)
            self.G.add_node(a.unique_id, agent=a)
            agents.append(a)
            self.agents.append(a)
        return agents

    def _build_network(self):
        # Supplier -> Plant
        for plant in self.plants:
            upstream = random.sample(
                self.suppliers, k=min(len(self.suppliers), 2 if self.dual_sourcing else 1)
            )
            for s in upstream:
                self.G.add_edge(s.unique_id, plant.unique_id)
        # Plant -> DC
        for dc in self.dcs:
            upstream = random.sample(self.plants, k=1)
            for p in upstream:
                self.G.add_edge(p.unique_id, dc.unique_id)
        # DC -> Retailer
        for r in self.retailers:
            upstream = random.sample(self.dcs, k=1)
            for d in upstream:
                self.G.add_edge(d.unique_id, r.unique_id)

    def place_order(self, buyer: BaseFirm, qty: int):
        """Split order among upstream nodes."""
        if qty <= 0:
            return
        upstream_nodes = list(self.G.predecessors(buyer.unique_id))
        if not upstream_nodes:
            buyer.receive_shipment(qty, buyer.lead_time)
            return
        per_supplier = int(np.ceil(qty / len(upstream_nodes)))
        for uid in upstream_nodes:
            supplier_agent = self.G.nodes[uid]["agent"]
            shipped = min(per_supplier, supplier_agent.inventory)
            if shipped > 0:
                supplier_agent.inventory -= shipped
                buyer.receive_shipment(shipped, supplier_agent.lead_time)
            # Correct backlog
            supplier_agent.backlog += (per_supplier - shipped)

    # -----------------------------
    # Disruption scenarios
    # -----------------------------
    def step_disruption(self):
        if not self.disruption_occurred and self.time == self.disruption_at_step:
            self.fill_rate_before_disruption = self.compute_fill_rate()
            candidates = [a for a in self.agents if a.tier in ("supplier", "plant")]
            victim = self.random.choice(candidates)

            if self.scenario == "capacity_only":
                victim.available_capacity = max(0, int(victim.capacity * (1 - self.capacity_loss_frac)))
            elif self.scenario == "lead_time_surge":
                victim.lead_time += 2  ## CAN CHANGE LEADTIME
            elif self.scenario == "demand_spike":
                self.retailer_demand_mean *= 2  ## CAN CHANGE DEMAND SPIKE

            victim.recovery_timer = self.recovery_duration
            self.disruption_occurred = True
            self.disruption_step = self.time
            print(f"Disruption at step {self.time} | Scenario: {self.scenario} | Victim: {victim}")

    # -----------------------------
    # Step
    # -----------------------------
    def step(self):
        self.step_disruption()
        for a in self.agents:
            a.step_order()
        for a in self.agents:
            a.step_receive()
            a.step_produce()
        for a in self.agents:
            a.step_recover()
        self.datacollector.collect(self)
        self.time += 1
        self.schedule.step()

    # -----------------------------
    # KPIs
    # -----------------------------
    def compute_fill_rate(self):
        total_demand = sum(r.total_demand for r in self.retailers)
        total_fulfilled = sum(r.fulfilled_demand for r in self.retailers)
        return total_fulfilled / total_demand if total_demand > 0 else np.nan

    def compute_total_cost(self):
        return sum(a.holding_cost + a.backlog_cost for a in self.agents)

    def compute_bullwhip(self):
        """Compute bullwhip once after simulation."""
        retailer_flat = [x for r in self.retailers for x in r.order_history]
        upstream_flat = [x for a in self.agents if a.tier in ("supplier","plant") for x in a.order_history]
        if len(retailer_flat) < 2 or np.var(retailer_flat) == 0:
            return np.nan
        return np.var(upstream_flat) / np.var(retailer_flat)

    def compute_time_to_recover(self):
        if self.disruption_step is None:
            return np.nan
        baseline_fill_rate = getattr(self, "fill_rate_before_disruption", 0.9)
        df = self.datacollector.get_model_vars_dataframe()
        recovery_steps = df.index[df["fill_rate"] >= baseline_fill_rate]
        recovery_steps = [s for s in recovery_steps if s >= self.disruption_step]
        if recovery_steps:
            self.recovery_step = recovery_steps[0]
            return self.recovery_step - self.disruption_step + 1
        return np.nan
