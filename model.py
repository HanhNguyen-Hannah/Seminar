from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import networkx as nx
import random
import numpy as np
from agents import BaseFirm

class MultiTierModel(Model):
    """
    Multi-tier supply chain model with suppliers, plants, DCs, retailers.
    Supports disruptions and KPI computation.
    """
    def __init__(self, assumptions=None, seed=None, dual_sourcing=False):
        super().__init__(seed=seed)
        self.schedule = SimultaneousActivation(self)
        self.G = nx.DiGraph()
        self.dual_sourcing = dual_sourcing
        self.time = 0

        # ---------------------------
        # Read assumptions
        # ---------------------------
        if assumptions is None:
            assumptions = {}

        self.n_suppliers = assumptions.get("n_suppliers", 3)
        self.n_plants = assumptions.get("n_plants", 3)
        self.n_dcs = assumptions.get("n_dcs", 2)
        self.n_retailers = assumptions.get("n_retailers", 5)

        self.base_stock = assumptions.get("base_stock", {"supplier":100,"plant":80,"dc":60,"retailer":30})
        self.capacity = assumptions.get("capacity", {"supplier":20,"plant":15,"dc":10,"retailer":0})
        self.lead_time = assumptions.get("lead_time", {"supplier":1,"plant":2,"dc":1,"retailer":1})

        self.disruption_prob = assumptions.get("disruption_prob", 0.05)
        self.capacity_loss_frac = assumptions.get("capacity_loss_frac", 0.5)
        self.recovery_duration = assumptions.get("recovery_duration", 5)

        # ---------------------------
        # Create agents
        # ---------------------------
        self.agents = []
        uid = 0

        # Suppliers
        self.suppliers = []
        for i in range(self.n_suppliers):
            a = BaseFirm(uid, self, tier="supplier",
                         base_stock=self.base_stock["supplier"],
                         capacity=self.capacity["supplier"],
                         lead_time=self.lead_time["supplier"])
            self._add_agent(a); self.suppliers.append(a); uid += 1

        # Plants
        self.plants = []
        for i in range(self.n_plants):
            a = BaseFirm(uid, self, tier="plant",
                         base_stock=self.base_stock["plant"],
                         capacity=self.capacity["plant"],
                         lead_time=self.lead_time["plant"])
            self._add_agent(a); self.plants.append(a); uid += 1

        # DCs
        self.dcs = []
        for i in range(self.n_dcs):
            a = BaseFirm(uid, self, tier="dc",
                         base_stock=self.base_stock["dc"],
                         capacity=self.capacity["dc"],
                         lead_time=self.lead_time["dc"])
            self._add_agent(a); self.dcs.append(a); uid += 1

        # Retailers
        self.retailers = []
        for i in range(self.n_retailers):
            a = BaseFirm(uid, self, tier="retailer",
                         base_stock=self.base_stock["retailer"],
                         capacity=self.capacity["retailer"],
                         lead_time=self.lead_time["retailer"])
            self._add_agent(a); self.retailers.append(a); uid += 1

        # ---------------------------
        # Build upstream links
        # ---------------------------
        # Supplier -> Plant
        for plant in self.plants:
            upstream = random.sample(self.suppliers, k=min(len(self.suppliers), 1 if not dual_sourcing else 2))
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

        # ---------------------------
        # DataCollector
        # ---------------------------
        self.datacollector = DataCollector(
            model_reporters={"time": lambda m: m.time},
            agent_reporters={}
        )

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _add_agent(self, agent):
        self.schedule.add(agent)
        self.G.add_node(agent.unique_id, agent=agent)
        self.agents.append(agent)

    def place_order(self, buyer: BaseFirm, qty: int):
        """
        Simplified ordering logic: split order among upstream neighbors.
        """
        if qty <= 0:
            return
        upstream_nodes = list(self.G.predecessors(buyer.unique_id))
        if not upstream_nodes:
            # external supply
            buyer.in_transit.append((qty, buyer.lead_time))
            return
        per_supplier = int(np.ceil(qty / len(upstream_nodes)))
        for uid in upstream_nodes:
            supplier_agent = self.G.nodes[uid]['agent']
            shipment_qty = min(per_supplier, supplier_agent.inventory)
            if shipment_qty > 0:
                supplier_agent.inventory -= shipment_qty
                buyer.on_order += shipment_qty
                buyer.in_transit.append((shipment_qty, supplier_agent.lead_time))
            else:
                # future shipment
                buyer.in_transit.append((per_supplier, supplier_agent.lead_time + 1))

    # ---------------------------
    # Step methods
    # ---------------------------
    def step_disruptions(self):
        if self.random.random() < self.disruption_prob:
            candidates = [a for a in self.agents if a.tier in ("supplier", "plant")]
            victim = self.random.choice(candidates)
            victim.available_capacity = max(0, int(victim.capacity * (1 - self.capacity_loss_frac)))
            victim.recovery_timer = self.recovery_duration
            victim.lead_time += 1  # lead-time surge

    def step(self):
        # 1) disruption
        self.step_disruptions()
        # 2) order
        for a in self.agents:
            a.step_order()
        # 3) production / receive
        for a in self.agents:
            a.step_receive()
            a.step_produce()
        # 4) recover
        for a in self.agents:
            a.step_recover()
        # 5) collect data
        self.datacollector.collect(self)
        self.time += 1
        self.schedule.step()

    # ---------------------------
    # KPI methods
    # ---------------------------
    def compute_fill_rate(self):
        total_demand = sum(r.total_demand for r in self.retailers)
        total_fulfilled = sum(r.fulfilled_demand for r in self.retailers)
        return total_fulfilled / total_demand if total_demand > 0 else np.nan

    def compute_total_cost(self):
        return sum(a.holding_cost + a.backlog_cost for a in self.agents)

    def compute_bullwhip(self):
        retailer_orders = [r.order_history for r in self.retailers]
        retailer_orders_flat = [x for history in retailer_orders for x in history]

        upstream_orders = []
        for a in self.agents:
            if a.tier in ("supplier", "plant"):
                upstream_orders.extend(a.order_history)

        if len(retailer_orders_flat) < 2 or len(upstream_orders) < 2:
            return np.nan
        var_retail = np.var(retailer_orders_flat)
        var_upstream = np.var(upstream_orders)
        return var_upstream / var_retail if var_retail > 0 else np.nan

    def compute_time_to_recover(self, baseline_fill_rate=0.9):
        fill_rate = self.compute_fill_rate()
        if fill_rate >= baseline_fill_rate:
            return self.time
        return np.nan
