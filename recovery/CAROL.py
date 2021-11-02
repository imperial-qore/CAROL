import sys
sys.path.append('recovery/CAROLSrc/')

import numpy as np
import random
from copy import deepcopy
from .Recovery import *
from .CAROLSrc.src.constants import *
from .CAROLSrc.src.utils import *
from .CAROLSrc.src.train import *

class CAROLRecovery(Recovery):
    def __init__(self, hosts, env):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.model_name = f'CAROL_{self.env_name}_{hosts}'
        self.model_loaded = False
        self.tenure = 3
        self.k = 5

    def load_model(self):
        # Load training time series and thresholds
        self.train_time_data = normalize_time_data(np.load(data_folder + self.env_name + '/' + data_filename))
        self.thresholds = np.percentile(self.train_time_data, PERCENTILES, axis=0) 
        if self.env_name == 'simulator': self.thresholds *= percentile_multiplier
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model); self.model_loaded = True

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name); norm_series = self.train_time_data
        train_time_data, train_schedule_data, anomaly_data, class_data, thresholds = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            aloss = backprop(self.epoch, self.model, self.optimizer, train_time_data, train_schedule_data, self.env.stats, norm_series, thresholds)
            anomaly_score, f1 = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, thresholds, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tAScore = {anomaly_score}')
            self.accuracy_list.append((aloss, anomaly_score, f1))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_folder, f'{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def getObjective(self, cids, hids):
        decision = [(cids[i], hids[i]) for i in range(len(cids))]
        energy = self.env.stats.runSimpleSimulation(decision)
        return energy

    def getInit(self, cids):
        return tuple(random.sample(range(self.hosts), len(cids)))

    def getRandomNeighbours(self, sol, k):
        neighbours = []
        for _ in range(k):
            newsol = list(deepcopy(sol))
            newsol[random.choice(range(len(newsol)))] = random.choice(range(self.hosts))
            neighbours.append(tuple(newsol))
        return neighbours

    def optimize_decision(self, state, schedule):
        bcel = nn.BCELoss(reduction = 'mean')
        real_label = torch.tensor([0.9]).type(torch.DoubleTensor)
        result = {}
        # get anomalies
        pred_state = gen(self.model, state, schedule, real_label, bcel, 1e-3)
        pred_state = pred_state.view(1, -1).detach().clone().numpy()
        anomaly_any_dim, _ = check_anomalies(pred_state, self.thresholds, self.env_name)
        hostlist = np.where(anomaly_any_dim[0])[0].tolist()
        # select MMT containers
        selectedContainerIDs = []
        for hostID in hostlist:
            containerIDs = self.env.getContainersOfHost(hostID)
            if containerIDs:
                containerIPS = [self.env.containerlist[cid].getRAM()[0] for cid in containerIDs]
                selectedContainerIDs.append(containerIDs[np.argmin(containerIPS)])
        # select target hosts based on tabu search (emulates node-shift)
        bestsol = self.getInit(selectedContainerIDs)
        bestobj = self.getObjective(selectedContainerIDs, bestsol)
        csol, cobj = bestsol, bestobj
        tabu = {}; it, terminate = 1, 0
        for _ in range(400):
            if terminate > 100: break
            # print('### iter {}###  Current_Objvalue: {}, Best_Objvalue: {}'.format(it, cobj, bestobj))
            neighbours = self.getRandomNeighbours(csol, self.k)
            # searching the neighbourhood of the current solution
            for n in neighbours:
                if n in tabu: continue
                tabu[n] = {'val': self.getObjective(selectedContainerIDs, n), 'time': 0}
            # admissible move
            for _ in range(400):
                # select the move with the lowest objective
                bestn = min(tabu, key =lambda x: tabu[x]['val'])
                bval, btime = tabu[bestn]['val'], tabu[bestn]['time']
                # not tabu
                if btime < it:
                    csol, cobj = bestn, bval
                    if bval < bestobj:
                        bestsol, bestobj = bestn, bval
                        terminate = 0
                    else:
                        terminate += 1
                    tabu[bestn]['time'] = it + self.tenure
                    it += 1; break
                # tabu
                else:
                    # aspiration
                    if bval < bestobj:
                        csol, cobj = bestn, bval
                        bestsol, bestobj = bestn, bval
                        terminate = 0; it += 1; break
                    else:
                        tabu[bestn]['val'] = float('inf')
                        continue
        return [(selectedContainerIDs[i], bestsol[i]) for i in range(len(selectedContainerIDs))]

    def get_data(self):
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return time_data, schedule_data

    def run_model(self, time_series, original_decision):
        if not self.model_loaded: self.load_model()
        state, schedule = self.get_data()
        result = self.optimize_decision(state, schedule)
        prev_alloc = {}
        for c in self.env.containerlist:
            oneHot = [0] * len(self.env.hostlist)
            if c: prev_alloc[c.id] = c.getHostID()
        decision = []
        for cid, hid in original_decision:
            new_host = result[cid] if cid in result else hid
            if prev_alloc[cid] != new_host: decision.append((cid, new_host))
        return decision

