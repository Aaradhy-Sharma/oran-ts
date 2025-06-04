import random

class ResourceBlockPool:
    def __init__(self, num_rbs):
        self.num_rbs = num_rbs
        # Stores {'bs_id': <id>, 'ue_id': <id>} or None
        self.rb_status = {i: {"bs_id": None, "ue_id": None} for i in range(num_rbs)}

    def get_available_rbs_for_bs(self, bs_id, count):
        # RBs not used by *any* BS <<
        available = [
            rb_id for rb_id, status in self.rb_status.items() if status["bs_id"] is None
        ]
        random.shuffle(available)
        return available[:count]

    def mark_allocated(self, rb_id, bs_id, ue_id):
        if 0 <= rb_id < self.num_rbs:
            self.rb_status[rb_id]["bs_id"] = bs_id
            self.rb_status[rb_id]["ue_id"] = ue_id

    def release_rb(self, rb_id):
        if 0 <= rb_id < self.num_rbs:
            self.rb_status[rb_id] = {"bs_id": None, "ue_id": None}

    def release_rbs_for_ue(self, ue_id):
        for rb_id, status in list(self.rb_status.items()):
            if status["ue_id"] == ue_id:
                self.release_rb(rb_id)

    def release_rbs_for_bs(self, bs_id):
        # If a BS goes down, etc.
        for rb_id, status in list(self.rb_status.items()):
            if status["bs_id"] == bs_id:
                self.release_rb(rb_id)

    def reset(self):
        self.rb_status = {i: {"bs_id": None, "ue_id": None} for i in range(self.num_rbs)}