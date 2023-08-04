from hash_model import MITH
import time

from einops import rearrange
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn.functional as F
from optimization import BertAdam
from utils.calc_utils import calc_neighbor, calc_map_k
from load_data import generate_dataset
from utils import get_logger


dataset_root_path = "./dataset"


class TrainerAsym:
    def __init__(self, args):

        self.args = args

        torch.random.manual_seed(seed=self.args.seed)
        torch.autograd.set_detect_anomaly(True)

        os.makedirs(self.args.save_dir, exist_ok=True)
        self._init_writer()

        self.logger.info('Start logging...')
        
        if self.args.is_train:
            log_str_args = "\n"
            for para in self.args.__dict__:
                log_str_args += " " * (30 - len(para)) + str(para) + "=" + str(self.args.__dict__[para]) + "\n"
            self.logger.info(log_str_args)
        else:
            self.logger.info(f"pretrained: {self.args.pretrained}")

        self.rank = self.args.rank  # gpu rank

        self._init_dataset()
        self._init_model()

        self.max_mapi2t = 0
        self.best_epoch_i2t = 0
        self.max_mapt2i = 0
        self.best_epoch_t2i = 0
        self.max_map = {'i2t': 0, "t2i": 0}
        self.best_epoch = 0

        self.logger.info("Train dataset len: {}".format(len(self.train_loader.dataset)))
        self.k_bits = self.args.k_bits

        self.img_buffer_tokens = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        self.img_buffer_cls = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        
        self.txt_buffer_tokens = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        self.txt_buffer_cls = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)

        self.device = self.txt_buffer_cls.device
        self.run()

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))

        # with open(os.path.join(self.args.save_dir, "description.txt"), 'w') as f:
        #     f.close()

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("Using ViT & GPT2...")

        self.model = MITH(args=self.args).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info(f"load pretrained model at {self.args.pretrained}")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.optimizer = BertAdam(
            [
                {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                {'params': self.model.hash.parameters(), 'lr': self.args.lr},
            ],
            lr=self.args.lr,
            warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset...")

        global dataset_root_path
        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                  indexFile=self.args.index_file,
                                                                  labelFile=self.args.label_file,
                                                                  maxWords=self.args.max_words,
                                                                  imageResolution=self.args.resolution,
                                                                  query_num=self.args.query_num,
                                                                  train_num=self.args.train_num,
                                                                  seed=self.args.seed)

        self.train_labels = train_data.get_all_label().float()
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()

        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )

    def change_state(self, mode):
        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model.pth"))
        self.logger.info(f"{self.args.dataset}_{self.args.k_bits}. Epoch: {epoch}, save model to {os.path.join(self.args.save_dir, 'model.pth')}")

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info("\n\n\n")
        self.logger.info(
            "####################### Train epochs: %d/%d #######################" % (epoch, self.args.epochs))
        epoch_avg_loss_dict = {'all_loss': 0}

        for image, text, key_padding_mask, label, index in self.train_loader:

            image = image.float().to(self.rank, non_blocking=True)
            label = label.float().to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)

            output_dict = self.model(image, text, key_padding_mask)

            img_cls_hash = output_dict['img_cls_hash']
            txt_cls_hash = output_dict['txt_cls_hash']
            self.img_buffer_cls[index] = img_cls_hash.detach()
            self.txt_buffer_cls[index] = txt_cls_hash.detach()

            img_tokens_hash = output_dict['img_tokens_hash']
            txt_tokens_hash = output_dict['txt_tokens_hash']
            self.img_buffer_tokens[index] = img_tokens_hash.detach()
            self.txt_buffer_tokens[index] = txt_tokens_hash.detach()

            hyper_lambda = self.args.hyper_lambda
            B = torch.sign(
                (img_cls_hash.detach() * hyper_lambda + img_tokens_hash.detach() * (1 - hyper_lambda)) + \
                (txt_cls_hash.detach() * hyper_lambda + txt_tokens_hash.detach() * (1 - hyper_lambda)))

            ALL_LOSS_DICT = self.compute_loss(output_dict, label, B)

            loss = 0
            for key in ALL_LOSS_DICT:
                loss += ALL_LOSS_DICT[key]
                if key in epoch_avg_loss_dict:
                    epoch_avg_loss_dict[key] += ALL_LOSS_DICT[key]
                else:
                    epoch_avg_loss_dict[key] = ALL_LOSS_DICT[key]
            epoch_avg_loss_dict['all_loss'] += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] all loss avg: {epoch_avg_loss_dict['all_loss'].data / (len(self.train_loader))}")
        self.logger.info(f"lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train...")

        for epoch in range(self.args.epochs):
            time1 = time.time()
            self.train_epoch(epoch)

            time2 = time.time()
            spend_time = int(time2 - time1)
            self.logger.info(f"{self.args.dataset}_{self.args.k_bits}. Train epoch [{epoch}], spend {spend_time // 60} min, {spend_time % 60} sec")

            if (epoch + 1) % self.args.valid_freq == 0:
                self.valid(epoch)

            time3 = time.time()
            spend_time = int(time3 - time2)
            self.logger.info(f"{self.args.dataset}_{self.args.k_bits}. Valid epoch [{epoch}], spend {spend_time // 60} min, {spend_time % 60} sec")

        self.logger.info(f">>>>>>> FINISHED {self.args.dataset}_{self.args.k_bits}. <<<<<<<")
        self.logger.info(f"Best epoch: {self.best_epoch}, i2t = {self.max_map['i2t']} t2i = {self.max_map['t2i']}")
        self.logger.info(f"I-T Best epoch: {self.best_epoch_i2t}, mAP: {self.max_mapi2t}, T-I Best epoch: {self.best_epoch_t2i}, mAP: {self.max_mapt2i}")

    def valid(self, epoch):
        self.logger.info("\n")
        self.logger.info(" Valid: %d/%d " % (epoch, self.args.epochs))
        self.change_state(mode="valid")

        q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _k_ = None
        mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device), self.retrieval_labels.to(self.device), _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device), self.retrieval_labels.to(self.device), _k_).item()

        if mAPi2t + mAPt2i > self.max_map['i2t'] + self.max_map['t2i']:
            self.best_epoch = epoch
            self.max_map['i2t'] = mAPi2t
            self.max_map['t2i'] = mAPt2i
            self.logger.info("$$$$$$$$$$$$$$$$$$$$ Best maps. $$$$$$$$$$$$$$$$$$$$$$$$")
            self.save_model(epoch)
            self.save_mat(q_i, q_t, r_i, r_t)

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i2t = epoch
            self.max_mapi2t = max(self.max_mapi2t, mAPi2t)

        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t2i = epoch
            self.max_mapt2i = max(self.max_mapt2i, mAPt2i)

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}]")
        self.logger.info(f"MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")
        self.logger.info(f"MAX [{self.best_epoch}/{self.args.epochs}] MAP(i->t): {round(self.max_map['i2t'], 5)}, MAP(t->i): {round(self.max_map['t2i'], 5)}")

    def compute_loss(self, output_dict, label, B):
        ALL_LOSS = {}

        label_sim = calc_neighbor(self.train_labels.float().to(self.rank, non_blocking=True), label)

        img_tokens_hash = output_dict['img_tokens_hash']
        txt_tokens_hash = output_dict['txt_tokens_hash']

        img_cls_hash = output_dict['img_cls_hash']
        txt_cls_hash = output_dict['txt_cls_hash']

        res_img_cls = output_dict['res_img_cls']
        res_txt_eos = output_dict['res_txt_cls']

        trans_tokens_i = output_dict['trans_tokens_i']
        trans_tokens_t = output_dict['trans_tokens_t']
        
        # Token Intra 
        hyper_tokens_intra = self.args.hyper_tokens_intra
        ALL_LOSS['tokens_intra_likelihood'] = hyper_tokens_intra * \
            (self.bayesian_loss(self.img_buffer_tokens, img_tokens_hash, label_sim) + \
             self.bayesian_loss(self.txt_buffer_tokens, txt_tokens_hash, label_sim))

        # CLS Inter
        hyper_cls_inter = self.args.hyper_cls_inter
        ALL_LOSS['cls_inter_likelihood'] = hyper_cls_inter * \
                                      (self.bayesian_loss(self.img_buffer_cls, txt_cls_hash, label_sim) + \
                                       self.bayesian_loss(self.txt_buffer_cls, img_cls_hash, label_sim))

        # hash feature
        H_i = img_cls_hash * 0.5 + img_tokens_hash * 0.5
        H_t = txt_cls_hash * 0.5 + txt_tokens_hash * 0.5
        # quantization loss
        hyper_quan = self.args.hyper_quan
        ALL_LOSS['quantization'] = hyper_quan * (self.quantization_loss_2(H_i, B) + self.quantization_loss_2(H_t, B))

        # Contrastive Alignment loss
        hyper_info_nce = self.args.hyper_info_nce
        hyper_alpha = self.args.hyper_alpha
        ALL_LOSS['infoNCE'] = hyper_info_nce * \
            (self.info_nce_loss(res_img_cls, res_txt_eos) + \
            hyper_alpha * self.info_nce_loss_bmm(trans_tokens_i, trans_tokens_t))

        # 1*gradient back to student.
        item_1 = (F.mse_loss(img_cls_hash.detach(), img_tokens_hash, reduction='sum') + \
                  F.mse_loss(txt_cls_hash.detach(), txt_tokens_hash, reduction='sum'))
        # 0.1*gradient back to teacher.
        item_2 = 0.1 * (F.mse_loss(img_cls_hash, img_tokens_hash.detach(), reduction='sum') + \
                        F.mse_loss(txt_cls_hash, txt_tokens_hash.detach(), reduction='sum'))
        # distillation loss
        hyper_distill = self.args.hyper_distill
        ALL_LOSS['distillation'] = hyper_distill * (item_1 + item_2) / (img_cls_hash.shape[0])
        return ALL_LOSS

    def get_code(self, data_loader, length: int):
        k_bits = self.k_bits

        img_buffer = torch.empty(length, k_bits, dtype=torch.float).to(self.device)
        text_buffer = torch.empty(length, k_bits, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            index = index.numpy()
            output_dict = self.model(image, text, key_padding_mask)

            img_hash_cls = output_dict['img_cls_hash'].detach()
            txt_hash_cls = output_dict['txt_cls_hash'].detach()

            img_tokens_hash = output_dict['img_tokens_hash'].detach()
            txt_tokens_hash = output_dict['txt_tokens_hash'].detach()

            img_buffer[index, :] = torch.sign(img_tokens_hash + img_hash_cls)
            text_buffer[index, :] = torch.sign(txt_tokens_hash + txt_hash_cls)

        return img_buffer, text_buffer

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt):

        save_dir = os.path.join(self.args.save_dir, "PR_curve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.cpu().detach().numpy()
        retrieval_labels = self.retrieval_labels.cpu().detach().numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }

        scio.savemat(
            os.path.join(save_dir, f"MITH-" + self.args.dataset + "-" + str(self.args.k_bits) + ".mat"),
            result_dict)
        self.logger.info(f">>>>>> save best *.mat data!")

    def info_nce_loss(self, out_1, out_2, temperature=0.07):
        # out_*: ND
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)

        return 0.5 * (loss0 + loss1)

    def info_nce_loss_bmm(self, out_1, out_2, temperature=0.07):
        # out1: L,N,D
        # out2: L,N,D
        out_1 = out_1.permute(1, 0, 2)  # NLD
        out_2 = out_2.permute(1, 0, 2)  # NLD
        bz = out_1.size(0)

        sim = torch.bmm(out_1, out_2.permute(0, 2, 1))
        sim /= temperature

        word_num = sim.shape[1]

        sim_1 = rearrange(sim, "b n1 n2 -> (b n1) n2")
        sim_2 = rearrange(sim, "b n1 n2 -> (b n2) n1")

        targets = torch.arange(word_num).type_as(out_1).long().repeat(bz)

        loss_1 = F.cross_entropy(sim_1, targets)
        loss_2 = F.cross_entropy(sim_2, targets)

        return 0.5 * (loss_1 + loss_2)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss

    def quantization_loss_2(self, hash_feature, B):
        return F.mse_loss(hash_feature, B, reduction='sum') / (hash_feature.shape[0]) / self.k_bits

    def test(self):
        if self.args.pretrained == "" or self.args.pretrained == "MODEL_PATH":
            self.logger.error("test step must load a model! please set the --pretrained argument.")
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")

        self.change_state(mode="valid")

        q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _k_ = None
        mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()

        self.save_mat(q_i, q_t, r_i, r_t)

        self.logger.info(f"MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")
        self.logger.info(">>>>>> Save *.mat data! Exit...")
