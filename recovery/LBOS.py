import numpy as np
from .Recovery import *
from scheduler.DQL import DQLScheduler

class LBOSRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.scheduler = DQLScheduler('energy_latency_'+str(hosts))

    def recover_decision(self, original_decision):
        containerIDs = [i[0] for i in original_decision]
        decision = self.scheduler.filter_placement(self.scheduler.placement(containerIDs))
        return decision

    def run_model(self, time_series, original_decision):
        self.scheduler.env = self.env
        return self.recover_decision(original_decision)
