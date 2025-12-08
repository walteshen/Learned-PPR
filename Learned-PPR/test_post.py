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


##################################################################
##################################################################
def get_generate_matrix(n, P_matrix):
    eye_matrix = np.eye(n, dtype=np.int8)
    g_matrix = np.vstack([eye_matrix, P_matrix])

    return g_matrix


def find_correct(binary_noise):
    binary_noise = binary_noise.view(args.code.l, args.code.n).cpu().numpy()
    detect = np.sum(binary_noise, axis=0)
    correct_index = detect == 0
    return np.where(correct_index == True)[0]


def get_pc_matrix(n, P_matrix):
    eye_matrix = np.eye(n, dtype=np.int8)
    h_matrix = np.hstack([P_matrix, eye_matrix])
    return h_matrix


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def BLER(x_pred, x_gt, generate_matrix, correct_indices, N_prime):
    x_pred = x_pred.view(args.code_l, args.code_n).cpu().numpy()
    x_gt = x_gt.view(args.code_l, args.code_n).cpu().numpy()

    x_pred = x_pred[:, :N_prime]
    x_gt = x_gt[:, :N_prime]
    generate_matrix = generate_matrix[:, :N_prime]

    out = x_pred != x_gt
    detect = np.sum(out, axis=0)
    correct_index_ = detect == 0
    for i in correct_indices:
        if i < N_prime:
            correct_index_[i] = True
    decode_matrix = generate_matrix[:, correct_index_]
    if decode_matrix.size(1) < decode_matrix.size(0):
        return False
    else:
        return full_rank(decode_matrix)


##################################################################


class PPR_Dataset(data.Dataset):
    def __init__(self, code, len, GE_channel=False):
        self.code = code
        self.len = len

        self.GE_channel = GE_channel

        P_matrix = np.loadtxt(
            "./Results_PPR/P_8/P_14_8.txt",
            dtype=np.int8,
        )
        self.generator_matrix = torch.Tensor(
            get_generate_matrix(self.code.k, P_matrix).T
        ).long()

        self.pc_matrix = (
            torch.Tensor(get_pc_matrix(self.code.n - self.code.k, P_matrix))
            .transpose(0, 1)
            .long()
        )

    def __len__(self):
        return self.len

    def get_mask(self, coded_number):

        mask_length = int((self.code.n - self.code.k - coded_number))
        mask = torch.cat(
            (torch.ones((1, self.code.n - mask_length)), torch.zeros((1, mask_length))),
            1,
        )
        mask = mask.repeat(self.code.l, 1)
        return mask, self.code.k + coded_number

    def __getitem__(self, index):

        m = torch.randint(0, 2, (self.code.l, self.code.k))
        x = torch.matmul(m, self.generator_matrix) % 2
        mask, N_prime = self.get_mask(coded_number=self.code.n - self.code.k)

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
            self.generator_matrix,
            N_prime,
        )


##################################################################


def test(model, device, test_loader_list, min_FER=100):
    model.eval()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_bler = test_ber = test_fer = cum_count = 0.0
            for batch_idx, (
                x,
                y,
                magnitude,
                syndrome,
                magnitude_T,
                syndrome_T,
                mask,
                G,
                N_prime,
            ) in enumerate(test_loader):

                z_mul = y * bin_to_sign(x)
                z_pred = model(
                    magnitude.to(device),
                    syndrome.to(device),
                    magnitude_T.to(device),
                    syndrome_T.to(device),
                )
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
                test_ber += BER(sign_to_bin(y).to(device), x.to(device))
                test_fer += BER(x_pred, x.to(device))
                correct_index = find_correct(sign_to_bin(z_mul))

                test_bler += BLER(
                    x_pred, x.to(device), G.squeeze(), correct_index, N_prime
                )
                cum_count += 1

            print("The test count", cum_count)
            print("The pred BLER: ", test_bler / cum_count)
            print("The pred BER: ", test_fer / cum_count)
            print("The real BER: ", test_ber / cum_count)


##################################################################
##################################################################
##################################################################


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    model = torch.load(args.model_path).to(device)
    logging.info(model)
    logging.info(
        f"# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}"
    )
    #################################

    test_dataloader_list = [
        DataLoader(
            PPR_Dataset(
                code,
                len=int(args.test_batch_size) * 2000,
                GE_channel=args.GE_channel,
            ),
            batch_size=int(args.test_batch_size),
            shuffle=False,
            num_workers=args.workers,
        )
    ]

    # if epoch % 300 == 0 or epoch in [1, args.epochs]:
    # for i in range(50):
    test(model, device, test_dataloader_list)


##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Learned-PPR")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", type=str, default="-1", help="gpus ids")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--erasure_prob", type=float, default=0.3)
    parser.add_argument("--GE_channel", type=bool, default=False)
    # Code args
    parser.add_argument("--code_l", type=int, default=50)
    parser.add_argument("--code_k", type=int, default=26)
    parser.add_argument("--code_n", type=int, default=36)

    # model args
    parser.add_argument("--N_dec", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--h", type=int, default=16)

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
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
    main(args)
