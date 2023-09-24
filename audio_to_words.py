import argparse
import pickle

import torch
from config import pickle_file, device, input_dim,LFR_m,LFR_n
from data_gen import build_LFR_features
from transformer.transformer import Transformer
from utils import extract_feature


def parse_args():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding."
    )
    # decode
    parser.add_argument("--beam_size", default=5, type=int, help="Beam size")
    parser.add_argument("--nbest", default=5, type=int, help="Nbest size")
    parser.add_argument(
        "--decode_max_len",
        default=100,
        type=int,
        help="Max output length. If ==0 (default), it uses a "
        "end-detect function to automatically find maximum "
        "hypothesis lengths",
    )
    parser.add_argument("voice", type=str, help="sound for test")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    # with open("char_list.pkl", "rb") as file:
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)
    samples = data["test"]
    char_list = data["IVOCAB"]

    filename = "./BEST_checkpoint.tar"
    wave: str = ""

    cp = torch.load(filename)
    model: Transformer = cp["model"]
    model.to(device)
    model.eval()

    # valid
    feature = extract_feature(
        input_file=wave, feature="fbank", dim=input_dim, cmvn=True
    )
    feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
    
    input = torch.from_numpy(feature).to(device)
    input_length = [input[0].shape[0]]
    input_length = torch.LongTensor(input_length).to(device)
    
    nbest_hyps = model.recognize(input,input_length,char_list,args)
    
    out_list = []
    for hyp in nbest_hyps:
        out = hyp['yseq']
        out = [char_list[idx] for idx in out]
        out = ''.join(out)
        out_list.append(out)
    print('OUT_LIST: {}'.format(out_list))