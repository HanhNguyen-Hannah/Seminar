# model.py
from mesa import Model
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
from agents import BaseFirm

class DummySchedule:
    """Small schedule to satisfy DataCollector (agents list and step counter)."""
    def __init__(self):
        self.agents = []
        self.steps = 0

class MultiTierModel(Model):
    """
    Multi-tier supply chain ABM.
    Default topology: 2 suppliers -> 1 plant -> 1 DC -> 3 retailers
    Supports strategies: dual sourcing, safety stock (factor), flexible capacity, dynamic reallocation (SRF).
    Single disruption per run at disruption_at_step.
    """

    def __init__(self, assumptions=None, seed=None, strategies=None,
                 disruption_at_step=10, scenario="capacity_loss"):
        super().__init__(seed=seed)
        self.schedule = DummySchedule()
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # default configs
        assumptions = assumptions or {}
        strategies = strategies or {}

        # network sizes
        self.n_suppliers = assumptions.get("n_suppliers", 2)
        self.n_plants = assumptions.get("n_plants", 1)
        self.n_dcs = assumptions.get("n_dcs", 1)
        self.n_retailers = assumptions.get("n_retailers", 3)

        # base params
        self.base_stock = assumptions.get("base_stock",
                                          {"supplier": 100, "plant": 80, "dc": 60, "retailer": 30})
        self.capacity = assumptions.get("capacity",
                                        {"supplier": 20, "plant": 15, "dc": 10, "retailer": 0})
        self.lead_time = assumptions.get("lead_time",
                                        {"supplier": 1, "plant": 5, "dc": 5, "retailer": 2})
        self.capacity_loss_frac = assumptions.get("capacity_loss_frac", 0.5)
        self.recovery_duration = assumptions.get("recovery_duration", 5)
        self.holding_cost = assumptions.get("holding_cost", 1.0)
        self.backlog_cost = assumptions.get("backlog_cost", 5.0)
        self.retailer_demand_mean = assumptions.get("retailer_demand_mean", 5.0)
        self._retailer_demand_baseline = self.retailer_demand_mean

        # strategies
        self.strategy_dual = strategies.get("dual_sourcing", False)
        self.safety_stock_factor = strategies.get("safety_stock_factor", 1.0)
        self.flexible_capacity = strategies.get("flexible_capacity", False)
        self.dynamic_reallocation = strategies.get("dynamic_reallocation", False)
        self.dual_share = strategies.get("dual_share", (0.8, 0.2))  # primary/secondary

        # disruption config
        self.disruption_at_step = disruption_at_step
        self.scenario = scenario  # 'capacity_loss','lead_time_surge','demand_spike'
        self.time = 0
        self.disruption_done = False
        self.disruption_step = None
        self.fill_rate_before_disruption = None
        self._victim = None
        self.recovery_step = None

        # agents & network
        self.G = nx.DiGraph()
        self.agents = []
        self.suppliers, self.plants, self.dcs, self.retailers = [], [], [], []

        # apply safety stock factor
        for tier in self.base_stock:
            self.base_stock[tier] = int(self.base_stock[tier] * self.safety_stock_factor)

        # create agents
        uid = 0
        self.suppliers = self._create_agents("supplier", self.n_suppliers, uid); uid += self.n_suppliers
        self.plants = self._create_agents("plant", self.n_plants, uid); uid += self.n_plants
        self.dcs = self._create_agents("dc", self.n_dcs, uid); uid += self.n_dcs
        self.retailers = self._create_agents("retailer", self.n_retailers, uid); uid += self.n_retailers

        # build network edges
        self._build_network()

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "time": lambda m: m.time,
                "fill_rate": lambda m: m.compute_fill_rate(),
                "total_cost": lambda m: m.compute_total_cost(),
                "bullwhip": lambda m: m.compute_bullwhip()
            },
            agent_reporters={
                "inventory": "inventory",
                "backlog": "backlog",
                "holding_cost": "holding_cost",
                "backlog_cost": "backlog_cost"
            }
        )

    # -----------------------------
    # Creation & network
    # -----------------------------
    def _create_agents(self, tier, n, start_uid):
        created = []
        for i in range(n):
            a = BaseFirm(unique_id=start_uid + i, model=self, tier=tier,
                         base_stock=self.base_stock[tier],
                         capacity=self.capacity[tier],
                         lead_time=self.lead_time[tier],
                         holding_cost_per_unit=self.holding_cost,
                         backlog_cost_per_unit=self.backlog_cost)
            # flexible capacity (increase nominal capacity)
            if self.flexible_capacity and a.tier in ("supplier", "plant"):
                # example: +25% capacity
                a.capacity = max(1, int(a.capacity * 1.25))
                a._orig_capacity = a.capacity
            a._orig_capacity = a.capacity
            a._orig_lead_time = a.lead_time   
            self.G.add_node(a.unique_id, agent=a)
            created.append(a)
            self.agents.append(a)
            self.schedule.agents.append(a)
        return created

    def _build_network(self):
        # Supplier -> Plant (allow up to 2 suppliers per plant if dual sourcing)
        for plant in self.plants:
            k = min(len(self.suppliers), 2 if self.strategy_dual else 1)
            upstream = random.sample(self.suppliers, k=k)
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

    # -----------------------------
    # Step orchestration
    # -----------------------------
    def step(self):
        # 0) reset agent step state

        for a in self.agents:
            a.reset_step_state()

        t = self.time
        if t == self.disruption_step:
            df = self.datacollector.get_model_vars_dataframe()
            self.fill_rate_baseline = df["fill_rate"].iloc[0:self.disruption_step].mean()
            self.disruption_done = True

        # 1) trigger disruption
        self._maybe_trigger_disruption()

        # 2) collect 
        self.schedule.steps = self.time
        self.datacollector.collect(self)

        # 3) ordering decisions
        buyer_orders = {}
        for a in self.agents:
            q = a.step_order()
            buyer_orders[a.unique_id] = q

        # 4) process orders and allocate shipments
        self._process_orders_and_allocate(buyer_orders)

        # 5) receive shipments và update cost
        for a in self.agents:
            a.step_receive()

        # 6) production
        for a in self.agents:
            a.step_produce()

        # 7) recovery countdown
        for a in self.agents:
            a.step_recover()

        self.time += 1


    # -----------------------------
    # Disruption handling
    # -----------------------------
    def _maybe_trigger_disruption(self):
        if (not self.disruption_done) and (self.time == self.disruption_at_step):
            # choose victim among suppliers and plants
            candidates = [a for a in self.agents if a.tier in ("supplier", "plant")]
            if not candidates:
                return
            victim = random.choice(candidates)
            self._victim = victim
            self.fill_rate_before_disruption = self.compute_fill_rate()
            # apply scenario
            if self.scenario == "capacity_loss":
                # reduce capacity via ramp fraction
                victim.recovery_ramp_fraction = max(0.0, 1.0 - self.capacity_loss_frac)
                # set available capacity now accordingly
                victim.available_capacity = int(victim.capacity * victim.recovery_ramp_fraction)
                victim.recovery_timer = self.recovery_duration
            elif self.scenario == "lead_time_surge":
                victim.lead_time += 1
                victim.recovery_timer = self.recovery_duration
            elif self.scenario == "demand_spike":
                self.retailer_demand_mean = self._retailer_demand_baseline * 4.0
                # model uses victim.recovery_timer as a system-level recovery window
                victim.recovery_timer = self.recovery_duration
            self.disruption_done = True
            self.disruption_step = self.time
            # print small log
            print(f"[Disruption] t={self.time} scenario={self.scenario} victim={victim}")

    def _agent_recovered(self, agent):
        """
        Called when an agent finishes recovery (its recovery_timer reached zero).
        Revert any temporary modifications (lead_time).
        """
        if self._victim is None or agent.unique_id != self._victim.unique_id:
            return
        if self._victim is not None and self._victim.recovery_timer == 0:
         # reset demand baseline
            if self.scenario == "demand_spike":
                self.retailer_demand_mean = self._retailer_demand_baseline
            agent.lead_time = agent._orig_lead_time
            agent.available_capacity = agent._orig_capacity
        self.recovery_step = self.time

    # -----------------------------
    # Orders processing and allocation
    # -----------------------------
    def _process_orders_and_allocate(self, buyer_orders):
        """
        1) Map buyer orders to upstream suppliers (or external infinite source).
        2) Create request buckets for all upstream nodes.
        3) Suppliers allocate according to SRF (dynamic_reallocation) or proportional.
        """
        # prepare request buckets for all agent ids (safe)
        request_buckets = {a.unique_id: [] for a in self.agents}

        # map buyer orders to upstream nodes
        for buyer_uid, qty in buyer_orders.items():
            if qty <= 0:
                continue
            buyer = self._agent_by_uid(buyer_uid)
            preds = list(self.G.predecessors(buyer_uid))
            if not preds:
                # external infinite source: immediate shipment with buyer lead_time
                buyer.receive_shipment(qty, lead_time=buyer.lead_time, from_uid=None)
                continue
            # if single upstream or duals not enabled, send whole request to single predecessor
            if (len(preds) == 1) or (not self.strategy_dual):
                # send full request to that one
                supplier_uid = preds[0]
                request_buckets[supplier_uid].append((buyer_uid, qty))
            else:
                # dual sourcing: split according to dual_share across available preds (use first two)
                # ensure ordering deterministic: sort preds
                pids = preds[:2]
                s1_share, s2_share = self.dual_share
                s1_qty = int(np.floor(qty * s1_share))
                s2_qty = qty - s1_qty
                request_buckets[pids[0]].append((buyer_uid, s1_qty))
                request_buckets[pids[1]].append((buyer_uid, s2_qty))

        # now allocation from each upstream node's inventory
        for supplier_uid, reqs in request_buckets.items():
            if not reqs:
                continue
            supplier = self._agent_by_uid(supplier_uid)
            total_req = sum(q for (_, q) in reqs)
            if total_req <= supplier.inventory:
                # satisfy all requests
                for (buyer_uid, q) in reqs:
                    if q <= 0:
                        continue
                    buyer = self._agent_by_uid(buyer_uid)
                    supplier.inventory -= q
                    buyer.receive_shipment(q, lead_time=supplier.lead_time, from_uid=supplier_uid)
            else:
                # shortage: apply dynamic reallocation (SRF) or proportional allocation
                if self.dynamic_reallocation:
                    # sort by buyer backlog descending (service-recovery-first)
                    reqs_sorted = sorted(reqs, key=lambda x: self._agent_by_uid(x[0]).backlog, reverse=True)
                    remaining = supplier.inventory
                    for (buyer_uid, q) in reqs_sorted:
                        if remaining <= 0:
                            break
                        alloc = min(q, remaining)
                        if alloc > 0:
                            buyer = self._agent_by_uid(buyer_uid)
                            supplier.inventory -= alloc
                            buyer.receive_shipment(alloc, lead_time=supplier.lead_time, from_uid=supplier_uid)
                            remaining -= alloc
                    # remaining unmet requests produce backlog at buyers (retailers will have backlog already)
                else:
                    # proportional allocation based on requested share
                    remaining = supplier.inventory
                    for (buyer_uid, q) in reqs:
                        if remaining <= 0:
                            break
                        share = q / total_req if total_req > 0 else 0
                        alloc = min(int(np.floor(supplier.inventory * share)), remaining)
                        if alloc > 0:
                            buyer = self._agent_by_uid(buyer_uid)
                            supplier.inventory -= alloc
                            buyer.receive_shipment(alloc, lead_time=supplier.lead_time, from_uid=supplier_uid)
                            remaining -= alloc
                    # allocate any leftover one-by-one
                    if remaining > 0:
                        for (buyer_uid, q) in reqs:
                            if remaining <= 0:
                                break
                            buyer = self._agent_by_uid(buyer_uid)
                            buyer.receive_shipment(1, lead_time=supplier.lead_time, from_uid=supplier_uid)
                            supplier.inventory -= 1
                            remaining -= 1

    def _agent_by_uid(self, uid):
        # small network: linear search OK; can optimize with dict if needed
        for a in self.agents:
            if a.unique_id == uid:
                return a
        return None

    # -----------------------------
    # KPI computations
    # -----------------------------
    def compute_fill_rate(self):
        total_demand = sum(r.total_demand for r in self.retailers)
        total_fulfilled = sum(r.fulfilled_demand for r in self.retailers)
        return total_fulfilled / total_demand if total_demand > 0 else np.nan

    def compute_total_cost(self):
        return sum(a.holding_cost + a.backlog_cost for a in self.agents)

    def compute_bullwhip(self):
        retailer_orders = [x for r in self.retailers for x in r.order_history]
        upstream_orders = [x for a in self.agents if a.tier in ("supplier", "plant") for x in a.order_history]
        if len(retailer_orders) < 2 or np.var(retailer_orders) == 0:
            return np.nan
        return float(np.var(upstream_orders) / np.var(retailer_orders)) if np.var(retailer_orders) > 0 else np.nan

    def compute_backlog_duration(self):
        """
        Average backlog duration across retailers (number of steps with backlog>0).
        """
        durations = [sum(r.backlog_history) for r in self.retailers]
        return float(np.mean(durations)) if durations else np.nan

    def compute_time_to_recover(self, target_frac=0.95):
        """
        TTR = steps until fill_rate >= target_frac * fill_rate avg in first 10 steps.
        """
        if not self.disruption_done or self.disruption_step is None:
            return np.nan

        baseline = getattr(self, "fill_rate_baseline", None)
        if baseline is None or np.isnan(baseline):
            return np.nan

        df = self.datacollector.get_model_vars_dataframe()
        target = target_frac * baseline

        recovery_steps = df.index[(df["fill_rate"] >= target) & (df.index >= self.disruption_step)].tolist()
        if recovery_steps:
            return int(recovery_steps[0] - self.disruption_step + 1)
        return np.nan


