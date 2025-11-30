# model.py
from mesa import Model
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
from agents import BaseFirm

class DummySchedule:
    """Small schedule to satisfy DataCollector."""
    def __init__(self):
        self.all_agents = []
        self.steps = 0

class MultiTierModel(Model):
    """
    Multi-tier supply chain ABM with disruption scenarios and resilience strategies.
    
    """

    def __init__(self, assumptions=None, seed=None, strategies=None,
                 disruption_at_step=10, scenario="capacity_loss", n_steps=100):
        super().__init__(seed=seed)
        self.schedule = DummySchedule()
        self._seed = seed
        self.n_steps = n_steps  # Store for TTR calculation
        random.seed(seed)
        np.random.seed(seed)

        # Default configs
        assumptions = assumptions or {}
        strategies = strategies or {}

        # Network sizes
        self.n_suppliers = assumptions.get("n_suppliers", 2)
        self.n_plants = assumptions.get("n_plants", 1)
        self.n_dcs = assumptions.get("n_dcs", 1)
        self.n_retailers = assumptions.get("n_retailers", 3)

        # Base params
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

        # Strategies
        self.strategy_dual = strategies.get("dual_sourcing", False)
        self.safety_stock_factor = strategies.get("safety_stock_factor", 1.2)
        self.flexible_capacity = strategies.get("flexible_capacity", False)
        self.dynamic_reallocation = strategies.get("dynamic_reallocation", False)

        # Disruption config
        self.disruption_at_step = disruption_at_step
        self.scenario = scenario
        self.time = 0
        self.disruption_done = False
        self.disruption_step = None
        self.fill_rate_baseline = None
        self._victim = None
        self.victim_history = []
        self.recovery_step = None

        # Agents & network
        self.G = nx.DiGraph()
        self.all_agents = []
        self.suppliers, self.plants, self.dcs, self.retailers = [], [], [], []

        # Apply safety stock factor
        for tier in self.base_stock:
            self.base_stock[tier] = int(self.base_stock[tier] * self.safety_stock_factor)

        # Create agents
        uid = 0
        self.suppliers = self._create_agents("supplier", self.n_suppliers, uid)
        uid += self.n_suppliers
        self.plants = self._create_agents("plant", self.n_plants, uid)
        uid += self.n_plants
        self.dcs = self._create_agents("dc", self.n_dcs, uid)
        uid += self.n_dcs
        self.retailers = self._create_agents("retailer", self.n_retailers, uid)

        # Build network
        self._build_network()

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "time": lambda m: m.time,
                "fill_rate": lambda m: m.compute_fill_rate(),
                "total_cost": lambda m: m.compute_total_cost(),
                "bullwhip_total": lambda m: m.compute_bullwhip().get("total", np.nan),
                "bullwhip_retailer": lambda m: m.compute_bullwhip().get("retailer", np.nan),
                "bullwhip_dc": lambda m: m.compute_bullwhip().get("dc", np.nan),
                "bullwhip_plant": lambda m: m.compute_bullwhip().get("plant", np.nan),
                "bullwhip_supplier": lambda m: m.compute_bullwhip().get("supplier", np.nan),
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
            
            # Flexible capacity: increase nominal capacity
            if self.flexible_capacity and a.tier in ("supplier", "plant"):
                a.capacity = max(1, int(a.capacity * 1.5))
                a._orig_capacity = a.capacity
            
            self.G.add_node(a.unique_id, agent=a)
            created.append(a)
            self.all_agents.append(a)
            self.schedule.all_agents.append(a)
        return created

    def _build_network(self):
        """
        Build deterministic network topology using round-robin assignment.
        Ensures all nodes at every tier are connected.
        """
        # Supplier -> Plant: Loop plants
        for i, plant in enumerate(self.plants):
            if self.strategy_dual and len(self.suppliers) >= 2:
                # Dual sourcing: assign 2 suppliers per plant
                s1_idx = (i * 2) % len(self.suppliers)
                s2_idx = (i * 2 + 1) % len(self.suppliers)
                self.G.add_edge(self.suppliers[s1_idx].unique_id, plant.unique_id)
                self.G.add_edge(self.suppliers[s2_idx].unique_id, plant.unique_id)
            else:
                # Single sourcing: round-robin
                s = self.suppliers[i % len(self.suppliers)]
                self.G.add_edge(s.unique_id, plant.unique_id)
        
        # Plant -> DC: Loop plants
        for i, plant in enumerate(self.plants):
            dc = self.dcs[i % len(self.dcs)]
            self.G.add_edge(plant.unique_id, dc.unique_id)
        
        # DC -> Retailer: Loop retailers
        for i, r in enumerate(self.retailers):
            dc = self.dcs[i % len(self.dcs)]
            self.G.add_edge(dc.unique_id, r.unique_id)

    # -----------------------------
    # Step orchestration
    # -----------------------------
    def step(self):
        # 0) Reset agent step state
        for a in self.all_agents:
            a.reset_step_state()

        # Calculate baseline BEFORE disruption
        if self.time == self.disruption_at_step - 1:
            df = self.datacollector.get_model_vars_dataframe()
            if len(df) >= 5:
                # Use last 5 steps as baseline
                self.fill_rate_baseline = df["fill_rate"].iloc[-5:].mean()
            else:
                self.fill_rate_baseline = 0.95

        # 1) Trigger disruption
        self._maybe_trigger_disruption()

        # 2) Collect data
        self.schedule.steps = self.time
        self.datacollector.collect(self)

        # 3) Ordering decisions
        buyer_orders = {}
        for a in self.all_agents:
            q = a.step_order()
            buyer_orders[a.unique_id] = q
            if a.tier == "retailer":
                pass

        # 4) Process orders and allocate
        self._process_orders_and_allocate(buyer_orders)

        # 5) Receive shipments and fulfill demand
        for a in self.all_agents:
            a.step_receive()

        # 6) Production (suppliers only)
        for a in self.all_agents:
            a.step_produce()

        # 7) Recovery countdown
        for a in self.all_agents:
            a.step_recover()

        # Track victim state
        if self._victim is not None:
            self.victim_history.append({
                "step": self.time,
                "inventory": self._victim.inventory,
                "backlog": self._victim.backlog,
                "capacity": self._victim.capacity,
                "lead_time": self._victim.lead_time
            })

        self.time += 1

    # -----------------------------
    # Disruption handling
    # -----------------------------
    def _maybe_trigger_disruption(self):
        if (not self.disruption_done) and (self.time == self.disruption_at_step):
            # Select victim based on scenario
            if self.scenario == "capacity_loss":
                candidates = [a for a in self.all_agents if a.tier in ("supplier", "plant")]
            elif self.scenario == "demand_spike":
                candidates = [a for a in self.all_agents if a.tier == "retailer"]
            elif self.scenario == "lead_time_surge":
                candidates = [a for a in self.all_agents if a.tier in ("supplier", "plant", "dc")]
            else:
                candidates = self.all_agents

            if not candidates:
                return
            
            victim = random.choice(candidates)
            self._victim = victim

            # Apply disruption
            if self.scenario == "capacity_loss":
                victim.is_disrupted = True
                victim.capacity = max(1, int(victim.capacity * (1 - self.capacity_loss_frac)))
                victim.available_capacity = victim.capacity
                victim.recovery_timer = self.recovery_duration
                
            elif self.scenario == "lead_time_surge":
                # Affect entire tier
                affected_tier = victim.tier
                self._affected_agents = [a for a in self.all_agents if a.tier == affected_tier]
                
                for agent in self._affected_agents:
                    agent._orig_lead_time = agent.lead_time
                    agent.lead_time += 3
                    agent.is_disrupted = True
                    agent.recovery_timer = self.recovery_duration
                
            elif self.scenario == "demand_spike":
                # Affect all retailers
                self.retailer_demand_mean = self._retailer_demand_baseline * 5
                
                for r in self.retailers:
                    r.is_disrupted = True
                    r.recovery_timer = self.recovery_duration

            self.disruption_done = True
            self.disruption_step = self.time
            
            # Print disruption info
            if self.scenario == "capacity_loss":
                print(f"[Disruption] t={self.time} scenario={self.scenario} " +
                    f"victim={victim.tier}-{victim.unique_id} " +
                    f"(capacity: {victim._orig_capacity} -> {victim.capacity})")
            elif self.scenario == "lead_time_surge":
                print(f"[Disruption] t={self.time} scenario={self.scenario} " +
                    f"affected_tier={affected_tier} " +
                    f"(all {len(self._affected_agents)} {affected_tier}s: lead_time +3)")
            elif self.scenario == "demand_spike":
                print(f"[Disruption] t={self.time} scenario={self.scenario} " +
                    f"(all {len(self.retailers)} retailers: demand x5)")

    def _agent_recovered(self, agent):
        """Called when recovery_timer reaches zero."""
        
        # Revert disruption effects
        agent.lead_time = agent._orig_lead_time
        agent.capacity = agent._orig_capacity
        agent.available_capacity = agent.capacity
        agent.is_disrupted = False

        # For demand_spike, restore demand when first retailer recovers
        if self.scenario == "demand_spike" and self.recovery_step is None:
            self.retailer_demand_mean = self._retailer_demand_baseline

        # Log recovery once (first agent to recover)
        if self.recovery_step is None:
            self.recovery_step = self.time
            
            if self.scenario == "capacity_loss":
                print(f"[Recovery] t={self.time} {agent.tier}-{agent.unique_id} " +
                    f"restored (capacity: {agent.capacity})")
            elif self.scenario == "lead_time_surge":
                print(f"[Recovery] t={self.time} {agent.tier} tier " +
                    f"restored (lead_time back to normal)")
            elif self.scenario == "demand_spike":
                print(f"[Recovery] t={self.time} all retailers " +
                    f"restored (demand: {self.retailer_demand_mean:.1f})")

    # -----------------------------
    # Order processing
    # -----------------------------
    def _process_orders_and_allocate(self, buyer_orders):
        # Request buckets
        request_buckets = {a.unique_id: [] for a in self.all_agents}

        demand_this_step = {a.unique_id: 0 for a in self.all_agents}

        # Map orders to upstream
        for buyer_uid, qty in buyer_orders.items():
            if qty <= 0:
                continue
            buyer = self._agent_by_uid(buyer_uid)
            preds = list(self.G.predecessors(buyer_uid))
            
            if not preds:
                # External infinite source
                buyer.receive_shipment(qty, lead_time=buyer.lead_time, from_uid=None)
                continue
            
            # Single or dual sourcing
            if (len(preds) == 1) or (not self.strategy_dual):
                supplier_uid = preds[0]
                request_buckets[supplier_uid].append((buyer_uid, qty))
                demand_this_step[supplier_uid] += qty
            else:
                # Dual sourcing backup: primary first, overflow to secondary
                pids = sorted(preds)[:2]
                primary_uid = pids[0]
                secondary_uid = pids[1]
                
                primary_supplier = self._agent_by_uid(primary_uid)
                primary_available = primary_supplier.inventory
                
                if primary_available >= qty:
                    # Primary đủ hàng → order hết từ primary
                    request_buckets[primary_uid].append((buyer_uid, qty))
                    demand_this_step[primary_uid] += qty
                else:
                    # Primary không đủ → lấy hết từ primary, còn lại từ secondary
                    from_primary = primary_available
                    from_secondary = qty - from_primary
                    if from_primary > 0:
                        request_buckets[primary_uid].append((buyer_uid, from_primary))
                        demand_this_step[primary_uid] += from_primary
                    if from_secondary > 0:
                        request_buckets[secondary_uid].append((buyer_uid, from_secondary))
                        demand_this_step[secondary_uid] += from_secondary

        # Allocate from inventory
        for supplier_uid, reqs in request_buckets.items():
            if not reqs:
                continue
            supplier = self._agent_by_uid(supplier_uid)
            total_req = sum(q for (_, q) in reqs)
            
            if total_req <= supplier.inventory:
                # Fulfill all
                for (buyer_uid, q) in reqs:
                    if q <= 0:
                        continue
                    buyer = self._agent_by_uid(buyer_uid)
                    supplier.inventory -= q
                    buyer.receive_shipment(q, lead_time=supplier.lead_time, from_uid=supplier_uid)
            else:
                # Shortage: apply allocation policy
                if self.dynamic_reallocation:
                    # Service-recovery-first: prioritize high backlog
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
                else:
                    # Proportional allocation
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

        for uid, demand in demand_this_step.items():
            agent = self._agent_by_uid(uid)
            if agent:
                agent.record_demand_received(demand)

    def _agent_by_uid(self, uid):
        for a in self.all_agents:
            if a.unique_id == uid:
                return a
        return None

    # -----------------------------
    # KPI computations
    # -----------------------------
    def compute_fill_rate(self):
        total_demand = sum(r.total_demand for r in self.retailers)
        total_fulfilled = sum(r.fulfilled_demand for r in self.retailers)
        return total_fulfilled / total_demand if total_demand > 0 else 1.0

    def compute_total_cost(self):
        return sum(a.holding_cost + a.backlog_cost for a in self.all_agents)

    def compute_bullwhip(self, debug=False):
        """
        Tính Bullwhip ratio ĐÚNG CÁCH:
        Tại mỗi tier: Var(orders đặt ra) / Var(demand nhận vào)
        
        Returns: Dictionary với bullwhip ratio của từng tier và total
        """
        bullwhip_ratios = {}
        
        min_samples = 10  # Cần ít nhất 10 samples để tính variance có ý nghĩa
        
        # 1. RETAILER: Var(orders đặt cho DC) / Var(customer demand)
        retailer_orders = []
        retailer_demand = []
        for r in self.retailers:
            retailer_orders.extend(r.order_history)
            retailer_demand.extend(r.demand_received_history)
        
        if len(retailer_orders) >= min_samples and len(retailer_demand) >= min_samples:
            var_orders = np.var(retailer_orders)
            var_demand = np.var(retailer_demand)
            bullwhip_ratios["retailer"] = var_orders / var_demand if var_demand > 0.1 else 1.0
        else:
            bullwhip_ratios["retailer"] = np.nan
        
        # 2. DC: Var(orders đặt cho Plant) / Var(demand từ Retailers)
        dc_orders = []
        dc_demand = []
        for dc in self.dcs:
            dc_orders.extend(dc.order_history)
            dc_demand.extend(dc.demand_received_history)
        
        if len(dc_orders) >= min_samples and len(dc_demand) >= min_samples:
            var_orders = np.var(dc_orders)
            var_demand = np.var(dc_demand)
            bullwhip_ratios["dc"] = var_orders / var_demand if var_demand > 0.1 else 1.0
        else:
            bullwhip_ratios["dc"] = np.nan
        
        # 3. PLANT: Var(orders đặt cho Supplier) / Var(demand từ DCs)
        plant_orders = []
        plant_demand = []
        for p in self.plants:
            plant_orders.extend(p.order_history)
            plant_demand.extend(p.demand_received_history)
        
        if len(plant_orders) >= min_samples and len(plant_demand) >= min_samples:
            var_orders = np.var(plant_orders)
            var_demand = np.var(plant_demand)
            bullwhip_ratios["plant"] = var_orders / var_demand if var_demand > 0.1 else 1.0
        else:
            bullwhip_ratios["plant"] = np.nan
        
        # 4. SUPPLIER: Var(production/orders) / Var(demand từ Plants)
        supplier_orders = []
        supplier_demand = []
        for s in self.suppliers:
            supplier_orders.extend(s.order_history)
            supplier_demand.extend(s.demand_received_history)
        
        if len(supplier_orders) >= min_samples and len(supplier_demand) >= min_samples:
            var_orders = np.var(supplier_orders)
            var_demand = np.var(supplier_demand)
            bullwhip_ratios["supplier"] = var_orders / var_demand if var_demand > 0.1 else 1.0
        else:
            bullwhip_ratios["supplier"] = np.nan
        
        # 5. TOTAL CHAIN: Product của tất cả ratios (hoặc end-to-end ratio)
        valid_ratios = [v for v in bullwhip_ratios.values() if not np.isnan(v)]
        if valid_ratios:
            bullwhip_ratios["total"] = np.prod(valid_ratios)
        else:
            bullwhip_ratios["total"] = np.nan
        
        # Debug output
        if debug:
            print(f"  [Bullwhip by Tier]")
            for tier, ratio in bullwhip_ratios.items():
                print(f"    {tier:10s}: {ratio:.2f}" if not np.isnan(ratio) else f"    {tier:10s}: N/A")
        
        return bullwhip_ratios

    def compute_backlog_duration(self):
        """Average backlog duration across retailers post-disruption."""
        if self.disruption_step is None:
            return np.nan
        
        durations = []
        for r in self.retailers:
            hist = r.backlog_history[self.disruption_step:]
            dur = sum(1 for x in hist if x > 0)
            durations.append(dur)
        
        return float(np.mean(durations)) if durations else np.nan

    def compute_time_to_recover(self, target_frac=0.90):  # Lower threshold
        if not self.disruption_done or self.disruption_step is None:
            return np.nan
        
        df = self.datacollector.get_model_vars_dataframe()
        
        # Use fill_rate at disruption step as reference
        if self.disruption_step in df.index:
            pre_disruption = df[df.index < self.disruption_step]
            if len(pre_disruption) >= 3:
                baseline = pre_disruption["fill_rate"].tail(3).mean()
            else:
                baseline = 0.95
        else:
            baseline = 0.95
        
        target = target_frac * baseline
        
        # Find MINIMUM fill rate after disruption
        post_disruption = df[df.index > self.disruption_step]
        if len(post_disruption) == 0:
            return np.nan
        
        min_fill_idx = post_disruption["fill_rate"].idxmin()
        
        # Search for recovery AFTER the minimum
        recovery_search = post_disruption[post_disruption.index >= min_fill_idx]
        recovery_steps = recovery_search[recovery_search["fill_rate"] >= target].index.tolist()
        
        if recovery_steps:
            ttr = int(recovery_steps[0] - self.disruption_step)
            return max(1, ttr)
        
        return float(len(post_disruption))