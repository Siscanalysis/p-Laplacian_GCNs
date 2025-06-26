import os
import torch
import numpy as np
from torch.nn import  Module
import time
from utils.utils_method import prepare_log_files
import torch.nn.functional as F
import hashlib
import re

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class modelImplementation_nodeClassificator(Module):
    def __init__(self, model, criterion, device=None):
        super(modelImplementation_nodeClassificator, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

    def set_optimizer(self, lr, weight_decay=0):
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=lr, weight_decay=weight_decay)


    def set_optimizer_reddit(self, lr):
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)

    def evaluate(self, features, labels,mask):
        self.model.eval()
        with torch.no_grad():
            model_out, logits = self.model(features)
            set_labels = labels[mask]
            set_logits = logits[mask]
            set_model_out = model_out[mask]
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            loss = self.criterion(set_model_out, set_labels)
            acc = correct.item() * 1.0 / len(set_labels)
            return loss, acc
        
    def train_test_model(self, input_features,labels, train_mask, test_mask, valid_mask, n_epochs, test_epoch,
                         test_name="", log_path=".", patience=30):
        os.makedirs(log_path, exist_ok=True)
        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)

        dur = []
        best_val_acc = 0.0
        best_val_loss = 100000.0
        no_improv=0

        input_features=input_features.to(self.device)
        labels=labels.to(self.device)
        for epoch in range(n_epochs):
            if no_improv > patience:
                break
            self.model.train()
            epoch_start_time = time.time()
            self.optimizer.zero_grad()
            model_out, logits = self.model(input_features)
            loss= self.criterion(logits[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

            cur_epoch_time=time.time() - epoch_start_time
            dur.append(cur_epoch_time)

            if epoch % test_epoch == 0:
                print("epoch : ", epoch, " -- loss: ", loss.item(), "-- time: ",cur_epoch_time)

                train_loss, train_acc = self.evaluate(input_features, labels, train_mask)
                val_loss, val_acc = self.evaluate(input_features, labels, valid_mask)
                test_loss, test_acc = self.evaluate(input_features, labels, test_mask)


                print("training acc : ", train_acc, " -- test_acc : ", test_acc," -- valid_acc : ", val_acc)
                print("training loss : ", train_loss.item(), " -- test_acc : ", test_loss.item()," -- valid_acc : ", val_loss.item())
                print("------")
                mean_epoch_time = np.mean(np.asarray(dur))
                train_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        train_loss.item(),
                        train_acc,
                        mean_epoch_time,
                        loss))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        test_loss.item(),
                        test_acc,
                        mean_epoch_time,
                        loss))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        val_loss.item(),
                        val_acc,
                        mean_epoch_time,
                        loss))

                valid_log.flush()

            #early stopping
            no_improv += 1
            if val_acc > best_val_acc:
                no_improv = 0
                best_val_loss = val_loss.item()
                best_val_acc = val_acc
                print("--ES--")
                print("save_new_best_model, with acc:", val_acc)
                print("------")
                self.save_model(test_name, log_path)

        print("Best val acc:", best_val_acc)
        print("Best val loss:", best_val_loss)
        self.load_model(test_name, log_path)
        print("-----BEST EPOCH RESULT-----")
        _, train_acc = self.evaluate(input_features, labels, train_mask)
        _, val_acc = self.evaluate(input_features, labels, valid_mask)
        _, test_acc = self.evaluate(input_features, labels, test_mask)
        print("training acc : ", train_acc, " -- test_acc : ", test_acc," -- valid_acc : ", val_acc)


    def save_model(self, test_name, log_folder='./'):
        # Ensure base directory exists
        os.makedirs(log_folder, exist_ok=True)

        # Shorten + sanitize filename to avoid Windows file path issues
        short_hash = hashlib.md5(test_name.encode()).hexdigest()[:8]
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', test_name)[:50]  # shorter base name
        filename = f"{safe_name}_{short_hash}.pt"

        model_path = os.path.join(log_folder, filename)

        try:
            torch.save(self.model.state_dict(), model_path)
        except Exception as e:
            print(f"[ERROR] Failed to save model to: {model_path}")
            print(e)
    #originale versione
    #def save_model(self,test_name, log_folder='./'):
    #    torch.save(self.model.state_dict(), os.path.join(log_folder,test_name+'.pt'))

    #def load_model(self,test_name, log_folder):
    #    self.model.load_state_dict(torch.load(os.path.join(log_folder,test_name+'.pt'),map_location=torch.device('cpu')))
    def load_model(self, test_name, log_folder='./'):
        # Match the save_model naming logic exactly
        short_hash = hashlib.md5(test_name.encode()).hexdigest()[:8]
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', test_name)[:50]
        filename = f"{safe_name}_{short_hash}.pt"
        model_path = os.path.join(log_folder, filename)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            print(f"[Warning] Model file not found at {model_path}. Skipping load.")



    def train_test_model_reddit(self, input_features,labels, train_mask, test_mask, valid_mask, n_epochs, test_epoch,
                         test_name="", log_path=".", patience=30):

        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)

        dur = []
        best_val_acc = 0.0
        best_val_loss = 100000.0
        no_improv=0



        input_features=input_features.to(self.device)
        labels=labels.to(self.device)

        def closure():
            self.optimizer.zero_grad()
            _, logits = self.model(input_features)
            loss_train = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss_train.backward()
            return loss_train


        for epoch in range(n_epochs):
            if no_improv > patience:
                break


            self.model.train()
            epoch_start_time = time.time()
            self.optimizer.step(closure)
            cur_epoch_time=time.time() - epoch_start_time
            dur.append(cur_epoch_time)

            if epoch % test_epoch == 0:
                print("epoch : ", epoch, "-- time: ",cur_epoch_time)

                train_loss, train_acc = self.evaluate(input_features, labels, train_mask)
                val_loss, val_acc = self.evaluate(input_features, labels, valid_mask)
                test_loss, test_acc = self.evaluate(input_features, labels, test_mask)


                print("training acc : ", train_acc, " -- test_acc : ", test_acc," -- valid_acc : ", val_acc)
                print("training loss : ", train_loss.item(), " -- test_acc : ", test_loss.item()," -- valid_acc : ", val_loss.item())
                print("------")
                mean_epoch_time = np.mean(np.asarray(dur))
                train_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        train_loss.item(),
                        train_acc,
                        mean_epoch_time,
                        train_loss.item()))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        test_loss.item(),
                        test_acc,
                        mean_epoch_time,
                        train_loss.item()))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        val_loss.item(),
                        val_acc,
                        mean_epoch_time,
                        train_loss.item()))

                valid_log.flush()

            #early stopping
            no_improv += 1
            torch.cuda.empty_cache()
            if val_acc > best_val_acc:
                no_improv = 0
                best_val_loss = val_loss.item()
                best_val_acc = val_acc
                print("--ES--")
                print("save_new_best_model, with acc:", val_acc)
                print("------")
                self.save_model(test_name, log_path)

        print("Best val acc:", best_val_acc)
        print("Best val loss:", best_val_loss)
        self.load_model(test_name, log_path)
        print("-----BEST EPOCH RESULT-----")
        _, train_acc = self.evaluate(input_features, labels, train_mask)
        _, val_acc = self.evaluate(input_features, labels, valid_mask)
        _, test_acc = self.evaluate(input_features, labels, test_mask)
        print("training acc : ", train_acc, " -- test_acc : ", test_acc," -- valid_acc : ", val_acc)


def evaluate(self, features, labels,mask):
        self.model.eval()
        with torch.no_grad():
            model_out, logits = self.model(features)
            set_labels = labels[mask]
            set_logits = logits[mask]
            set_model_out = model_out[mask]
            _, indices = torch.max(set_logits, dim=1)
            correct = torch.sum(indices == set_labels)
            loss = self.criterion(set_model_out, set_labels)
            acc = correct.item() * 1.0 / len(set_labels)
            return loss, acc

    


class modelImplementation_nodeClassificator_ogbn(modelImplementation_nodeClassificator):

    def train_test_model(self, input_features,labels, train_mask, test_mask, valid_mask, n_epochs, test_epoch,evaluator,
                         test_name="", log_path=".", patience=30):

        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
        )["acc"]

        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)

        dur = []
        best_val_acc = 0.0
        best_val_loss = 100000.0
        no_improv=0

        input_features=input_features.to(self.device)
        labels=labels.to(self.device)
        for epoch in range(n_epochs):
            if no_improv > patience:
                break
            self.model.train()
            epoch_start_time = time.time()
            self.optimizer.zero_grad()
            model_out, logits = self.model(input_features)
            loss= self.criterion(logits[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

            cur_epoch_time=time.time() - epoch_start_time
            dur.append(cur_epoch_time)

            if epoch % test_epoch == 0:
                print("epoch : ", epoch, " -- loss: ", loss.item(), "-- time: ",cur_epoch_time)

                train_loss, train_acc = self.evaluate(input_features, labels, train_mask,evaluator_wrapper)
                val_loss, val_acc = self.evaluate(input_features, labels, valid_mask,evaluator_wrapper)
                test_loss, test_acc = self.evaluate(input_features, labels, test_mask,evaluator_wrapper)


                print("training acc : ", train_acc, " -- test_acc : ", test_acc," -- valid_acc : ", val_acc)
                print("training loss : ", train_loss.item(), " -- test_acc : ", test_loss.item()," -- valid_acc : ", val_loss.item())
                print("------")
                mean_epoch_time = np.mean(np.asarray(dur))
                train_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        train_loss.item(),
                        train_acc,
                        mean_epoch_time,
                        loss))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        test_loss.item(),
                        test_acc,
                        mean_epoch_time,
                        loss))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        val_loss.item(),
                        val_acc,
                        mean_epoch_time,
                        loss))

                valid_log.flush()

            #early stopping
            no_improv += 1
            if val_acc > best_val_acc:
                no_improv = 0
                best_val_loss = val_loss.item()
                best_val_acc = val_acc
                print("--ES--")
                print("save_new_best_model, with acc:", val_acc)
                print("------")
                self.save_model(test_name, log_path)

        print("Best val acc:", best_val_acc)
        print("Best val loss:", best_val_loss)
        self.load_model(test_name, log_path)
        print("-----BEST EPOCH RESULT-----")
        train_acc = self.evaluate(input_features, labels, train_mask, evaluator_wrapper)
        val_acc = self.evaluate(input_features, labels, valid_mask,evaluator_wrapper)
        test_acc = self.evaluate(input_features, labels, test_mask,evaluator_wrapper)
        print("training acc : ", train_acc, " -- test_acc : ", test_acc," -- valid_acc : ", val_acc)

    def evaluate(self, features, labels, mask, evaluator):
        self.model.eval()
        with torch.no_grad():
            model_out, logits = self.model(features)
            set_labels = labels[mask]
            set_model_out = model_out[mask]
            loss = self.criterion(set_model_out, set_labels)
            return loss,evaluator(logits[mask].log_softmax(dim=-1), torch.atleast_2d(labels[mask]).T)

