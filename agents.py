# agents.py

from mesa import Agent
import numpy as np

class BaseFirm(Agent):
    """
    Agent mô phỏng một firm trong multi-tier supply chain.
    Có thể là Supplier, Plant, DC hoặc Retailer.
    """
    def __init__(self, unique_id, model, tier, base_stock=50, capacity=10, lead_time=1):
        super().__init__(unique_id, model)
        self.tier = tier
        self.inventory = base_stock
        self.base_stock = base_stock
        self.capacity = capacity
        self.available_capacity = capacity  # có thể giảm khi disruption
        self.lead_time = lead_time
        self.on_order = 0
        self.backlog = 0
        self.in_transit = []  # [(qty, remaining_time)]
        self.recovery_timer = 0  # đếm bước phục hồi sau disruption

        # KPI tracking
        self.total_demand = 0           # tổng cầu (chỉ retailer)
        self.fulfilled_demand = 0       # tổng cầu được đáp ứng
        self.holding_cost = 0           # chi phí tồn kho tích lũy
        self.backlog_cost = 0           # chi phí backlog tích lũy
        self.order_history = []         # lưu số lượng order mỗi step
        self.recovery_steps = []        # lưu số step để phục hồi

        # ---------------------------
        # NumPy random generator dùng seed từ model để reproducibility
        # ---------------------------
        # Nếu model.seed không None, sử dụng seed, nếu None thì random
        if hasattr(self.model, "_seed") and self.model._seed is not None:
            self.np_random = np.random.default_rng(self.model._seed + unique_id)
        else:
            self.np_random = np.random.default_rng()

    # ---------------------------
    # Step methods
    # ---------------------------
    def step_order(self):
        """
        Retailer tạo demand theo Poisson.
        Tính order dựa trên base-stock và tồn kho hiện tại.
        """
        if self.tier == "retailer":
            demand = self.np_random.poisson(5)  # dùng NumPy RNG theo seed
            self.total_demand += demand

            # đáp ứng inventory nếu có
            fulfilled = min(self.inventory, demand)
            self.inventory -= fulfilled
            unmet = demand - fulfilled
            self.backlog += unmet
            self.fulfilled_demand += fulfilled

        # Tính order dựa trên base-stock
        net_inventory = self.inventory + self.on_order - self.backlog
        order_qty = max(0, self.base_stock - net_inventory)
        self.order_history.append(order_qty)

        # Gửi order lên model
        self.model.place_order(self, order_qty)

    def step_receive(self):
        """
        Cập nhật inventory từ các lô hàng in_transit.
        Tính chi phí tồn kho và backlog cost.
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

        # Chi phí tồn kho & backlog
        h = 1  # cost per unit inventory
        p = 5  # cost per unit backlog
        self.holding_cost += self.inventory * h
        self.backlog_cost += self.backlog * p

    def step_produce(self):
        """
        Sản xuất nếu tier là supplier hoặc plant.
        """
        if self.tier in ("supplier", "plant") and self.available_capacity > 0:
            produced = min(self.available_capacity, self.capacity)
            self.inventory += produced

    def step_recover(self):
        """
        Phục hồi năng lực sau disruption.
        """
        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.available_capacity = self.capacity
                # reset lead_time nếu bị tăng tạm thời
