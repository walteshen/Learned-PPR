from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model_linear import PPR_Transformer


##################################################################
##################################################################
def get_generate_matrix(n, P_matrix):
    eye_matrix = np.eye(n, dtype=np.int8)
    g_matrix = np.vstack([eye_matrix, P_matrix])

    return g_matrix


def get_pc_matrix(n, P_matrix):
    eye_matrix = np.eye(n, dtype=np.int8)
    h_matrix = np.hstack([P_matrix, eye_matrix])
    return h_matrix


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_mask_from_pc_matrix(pc_matrix):
    mask_nk_nk = pc_matrix @ pc_matrix.T % 2
    mask_n_n = pc_matrix.T @ pc_matrix % 2
    tmp1 = torch.cat([mask_n_n, pc_matrix.T], 1)
    tmp2 = torch.cat([pc_matrix, mask_nk_nk], 1)
    return torch.cat([tmp1, tmp2], 0).unsqueeze(-1).squeeze()


def build_mask(code, pc_matrix):
    mask_size = code.n + pc_matrix.size(0)
    mask = torch.eye(mask_size, mask_size)
    for ii in range(pc_matrix.size(0)):
        idx = torch.where(pc_matrix[ii] > 0)[0]
        for jj in idx:
            for kk in idx:
                if jj != kk:
                    mask[jj, kk] += 1
                    mask[kk, jj] += 1
                    mask[code.n + ii, jj] += 1
                    mask[jj, code.n + ii] += 1
    src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
    return src_mask


##################################################################


class PPR_Dataset(data.Dataset):
    def __init__(self, code, len, GE_channel=False):
        self.code = code
        self.len = len

        self.GE_channel = GE_channel

        # P_matrix = np.random.randint(0, 2, (self.code.n - self.code.k, self.code.k))
        P_matrix = np.loadtxt("./Results_PPR/P_12/P_20_12.txt", dtype=np.int8)
        self.generator_matrix = torch.Tensor(
            get_generate_matrix(self.code.k, P_matrix).T
        ).long()

        self.pc_matrix = (
            torch.Tensor(get_pc_matrix(self.code.n - self.code.k, P_matrix))
            .transpose(0, 1)
            .long()
        )

        np.savetxt(
            os.path.join(model_dir, f"P_{self.code.n}_{self.code.k}.txt"),
            P_matrix,
            fmt="%d",
        )

    def __len__(self):
        return self.len

    def get_mask(self):

        mask_length = 0

        mask = torch.cat(
            (torch.ones((1, self.code.n - mask_length)), torch.zeros((1, mask_length))),
            1,
        )
        mask = mask.repeat(self.code.l, 1)
        return mask

    def __getitem__(self, index):

        m = torch.randint(0, 2, (self.code.l, self.code.k))
        x = torch.matmul(m, self.generator_matrix) % 2
        mask = self.get_mask()

        prob = 1 - 0.7 ** (1 / self.code.l)
        z = torch.Tensor(np.random.binomial(1, prob, (self.code.l, self.code.n)))
        z = torch.bernoulli(z)
        z = bin_to_sign(z)
        y = bin_to_sign(x) * z

        magnitude = y

        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)

        x = x.reshape(self.code.l * self.code.n)
        y = y.reshape(self.code.l * self.code.n)

        magnitude_T = magnitude.T
        syndrome_T = syndrome.T

        return (
            x.float(),
            y.float(),
            magnitude.float(),
            syndrome.float(),
            magnitude_T.float(),
            syndrome_T.float(),
            mask.bool(),
        )


##################################################################
##################################################################


def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()
    for batch_idx, (
        x,
        y,
        magnitude,
        syndrome,
        magnitude_T,
        syndrome_T,
        mask,
    ) in enumerate(train_loader):
        z_mul = y * bin_to_sign(x)
        z_pred = model(
            magnitude.to(device),
            syndrome.to(device),
            magnitude_T.to(device),
            syndrome_T.to(device),
        )
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx + 1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f"Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}"
            )
    logging.info(f"Epoch {epoch} Train Time {time.time() - t}s\n")
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################


def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        test_bler = real_ber = test_ber = cum_count = 0.0
        for batch_idx, (
            x,
            y,
            magnitude,
            syndrome,
            magnitude_T,
            syndrome_T,
            mask,
        ) in enumerate(test_loader):

            z_mul = y * bin_to_sign(x)
            z_pred = model(
                magnitude.to(device),
                syndrome.to(device),
                magnitude_T.to(device),
                syndrome_T.to(device),
            )
            loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
            real_ber += BER(sign_to_bin(y).to(device), x.to(device))
            test_ber += BER(x_pred, x.to(device))
            cum_count += 1
        # print("The pred BLER: ", test_bler / cum_count)
        # print("The pred BER: ", test_ber / cum_count)
        # print("The real BER: ", real_ber / cum_count)
        logging.info(
            f"Test: Real BER={real_ber / cum_count:.2e} Pred BER={test_ber / cum_count:.2e}"
        )
    return loss


##################################################################
##################################################################
##################################################################


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    model = PPR_Transformer(args, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(model)
    logging.info(
        f"# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}"
    )
    #################################

    train_dataloader = DataLoader(
        PPR_Dataset(
            code,
            len=args.batch_size * 5000,
            GE_channel=args.GE_channel,
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.workers,
    )

    test_dataloader = DataLoader(
        PPR_Dataset(
            code,
            len=1000,
            GE_channel=args.GE_channel,
        ),
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
    )
    #################################
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train(
            model,
            device,
            train_dataloader,
            optimizer,
            epoch,
            LR=scheduler.get_last_lr()[0],
        )
        scheduler.step()

        if epoch % 1 == 0 or epoch in [1, args.epochs]:
            test_loss = test(model, device, test_dataloader)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model, os.path.join(args.path, "best_model"))

        torch.save(model, os.path.join(args.path, f"model_{epoch}"))


##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Learned-PPR")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=str, default="-1", help="gpus ids")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # Code args
    parser.add_argument("--code_l", type=int, default=50)
    parser.add_argument("--code_k", type=int, default=26)
    parser.add_argument("--code_n", type=int, default=36)

    # model args
    parser.add_argument("--N_dec", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--h", type=int, default=8)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # set_seed(args.seed)
    ####################################################################

    class Code:
        pass

    code = Code()
    code.l = args.code_l
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    args.code = code
    ####################################################################
    model_dir = os.path.join(
        "Results_Learned_PPR",
        args.code_type
        + "__Code_n_"
        + str(args.code_n)
        + "_k_"
        + str(args.code_k)
        + "_l_"
        + str(args.code_l)
        + "__"
        + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
    )
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [logging.FileHandler(os.path.join(model_dir, "logging.txt"))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args)
