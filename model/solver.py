# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import trange
from model.layers.summarizer import SD_VSum
from model.utils.evaluation_metrics import evaluate_summary


class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None):
        """
        Class builds, trains, and evaluates the SD-VSum model.
        :param argparse.Namespace config: Configuration object with hyperparameters and file paths.
        :param torch.utils.data.DataLoader train_loader: DataLoader for training data.
        :param torch.utils.data.DataLoader val_loader: DataLoader for validation data.
        :param torch.utils.data.DataLoader test_loader: DataLoader for test data.
        """
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.BCELoss()

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)


    def build(self):
        """
        Builds and initializes the SD-VSum summarization model and optimizer.
        Loads pretrained weights if available; otherwise applies weight initialization.
        :return None
        """

        self.model = SD_VSum(input_size=self.config.input_size,
                                          text_size=self.config.text_size,
                                          output_size=self.config.input_size,
                                          heads=self.config.heads,
                                          pos_enc=self.config.pos_enc).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)

        pretrained_model_path = self.config.ckpt_path
        if os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model from {pretrained_model_path}")
            with open(pretrained_model_path, 'rb') as f:
                state_dict = torch.load(f)
                self.model.load_state_dict(state_dict)
        else:
            print(f"No pretrained model found at {pretrained_model_path}, training from scratch.")
            if self.config.init_type is not None:
                self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)


    def train(self):
        """
        Trains the summarization model over multiple epochs, tracking validation performance.
        :return str: Path to the checkpoint with the best validation F1 score.
        """
        best_f1score = -1.0
        best_f1score_epoch = 0

        loss_total = []
        f1score_total = []

        val_f1score = self.evaluate(dataloader=self.val_loader)
        f1score_total.append(val_f1score)

        f_score_path = os.path.join(self.config.save_dir_root, "val_f1score.txt")
        with open(f_score_path, "w") as file:
            file.write(f"{val_f1score}\n")

        for epoch_i in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i) + "/" + str(self.config.epochs)))
            self.model.train()
            loss_history = []
            num_batches = int(len(self.train_loader) / self.config.batch_size)
            iterator = iter(self.train_loader)

            for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
                self.optimizer.zero_grad()

                for _ in trange(self.config.batch_size, desc='Video', ncols=80, leave=False):
                    video_embeddings, text_embeddings_full, gtscores = next(iterator)
                    video_embeddings = video_embeddings.to(self.config.device)
                    video_embeddings = video_embeddings.squeeze()
                    text_embeddings_full = text_embeddings_full.to(self.config.device)
                    text_embeddings_full = text_embeddings_full.squeeze(0)
                    gtscores = gtscores.to(self.config.device)
                    gtscores = gtscores.squeeze(0)

                    for i in range(self.config.annotations):

                        if text_embeddings_full.ndim == 2: # in the case of S_NewsVSum
                            text_embeddings_full = text_embeddings_full.unsqueeze(0)
                            gtscores = gtscores.unsqueeze(0)

                        gtscore = gtscores[i]
                        text_embeddings = text_embeddings_full[i, :, :]
                        mask = (text_embeddings.abs().sum(dim=1) != 0)  # remove zero padding
                        text_embeddings = text_embeddings[mask]
                        if text_embeddings.shape[0] == 0:
                            continue
                        score = self.model(video_embeddings, text_embeddings)

                        loss = self.criterion(score.squeeze(0), gtscore.squeeze(0))
                        loss.backward()
                        loss_history.append(loss.data)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            loss = torch.stack(loss_history).mean()
            val_f1score = self.evaluate(dataloader=self.val_loader)
            loss_total.append(loss)
            f1score_total.append(val_f1score)
            with open(f_score_path, "a") as file:
                file.write(f"{val_f1score}\n")

            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                torch.save(self.model.state_dict(), f1_save_ckpt_path)


            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('    VAL  F-score {0:0.5} '.format(val_f1score))

        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch))
        f.flush()
        f.close()

        return f1_save_ckpt_path

    def evaluate(self, dataloader=None):
        """"
        Evaluates the model on a given DataLoader using the F-score metric.
        :param torch.utils.data.DataLoader dataloader: DataLoader for evaluation.
        :return float: Mean F1-score across all evaluated videos.
        """
        self.model.eval()

        fscore_history = []
        dataloader = iter(dataloader)

        for video_embeddings, text_embeddings_full, gtscores in dataloader:
            video_embeddings = video_embeddings.to(self.config.device)
            text_embeddings_full = text_embeddings_full.to(self.config.device)
            f_socre_video = []
            for i in range(self.config.annotations):
                gtscore = gtscores[i]

                if text_embeddings_full.ndim == 2:  # in the case of S_NewsVSum
                    text_embeddings_full = text_embeddings_full.unsqueeze(0)

                text_embeddings = text_embeddings_full[i, :, :]
                mask = (text_embeddings.abs().sum(dim=1) != 0)
                text_embeddings = text_embeddings[mask]
                if text_embeddings.shape[0] == 0:
                    continue
                with torch.no_grad():
                    score = self.model(video_embeddings, text_embeddings)

                # Summarization metric
                score = score.squeeze().cpu()
                f_score = evaluate_summary(score, gtscore)
                f_socre_video.append(f_score)

            if f_socre_video:
                fscore_history.append(np.mean(f_socre_video))

        final_f_score = np.mean(fscore_history)
        return final_f_score

    def test(self, ckpt_path):
        """
        Tests the summarization model using a specified checkpoint and report test performance.
        :param str ckpt_path: Path to the saved model checkpoint.
        :return None
        """
        if ckpt_path is not None:
            print("Testing Model: ", ckpt_path)
            print("Device: ", self.config.device)
            self.model.load_state_dict(torch.load(ckpt_path))

        test_fscore = self.evaluate(dataloader=self.test_loader)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRHiSum F-score {0:0.5}'.format(test_fscore))
        print("------------------------------------------------------")

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')

        f.flush()

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))  # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)


if __name__ == '__main__':
    pass
