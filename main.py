from argparse import Namespace
import argparse
import functools
import pickle
from io import BytesIO
import torch
import config
from data_gen import build_LFR_features
from transformer.transformer import Transformer
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger
from typing import BinaryIO

from utils import extract_feature


class Config:
    def __init__(self) -> None:
        self.beam_size = 5
        self.nbest = 10
        self.decode_max_len = 100


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
    args = Config()
    return args


class Model:
    def __init__(
        self, args: Namespace, check_point: str = "./BEST_checkpoint.tar"
    ) -> None:
        self.args = args
        self.check_point = torch.load(check_point)
        with open(config.pickle_file, "rb") as file:
            data = pickle.load(file)
        self.char_list = data["IVOCAB"]
        self.model: Transformer = self.check_point["model"]

        self.model.to(config.device)
        self.model.eval()

    def recognize(self, input: BinaryIO, skip_start_end: bool = False):
        """对wav文件进行语音识别

        Args:
            input (BinaryIO): 输入的音频,可以是打开的文件,也可以是内存中的binary Stream
            skip_start_end (bool, optional): 隐藏输出的标记`sos` 与`eos`. Defaults to False.

        Returns:
            List[String]: 模型推理给出的后验概率最高的推理结果
        """
        feature = extract_feature(
            input_file=input, feature="fbank", dim=config.input_dim, cmvn=True
        )
        feature = build_LFR_features(feature, m=config.LFR_m, n=config.LFR_n)

        input = torch.from_numpy(feature).to(config.device)
        input_length = [input[0].shape[0]]
        input_length = torch.LongTensor(input_length).to(config.device)

        nbest_hyps = self.model.recognize(
            input, input_length, self.char_list, self.args
        )

        out_list = []
        for hyp in nbest_hyps:
            out = hyp["yseq"]
            out = [
                self.char_list[idx]
                for idx in out
                if not (
                    skip_start_end
                    and (
                        self.char_list[idx] == "<sos>" or self.char_list[idx] == "<eos>"
                    )
                )
            ]
            out = "".join(out)
            out_list.append(out)

        return out_list


args = parse_args()
model = Model(args)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
)


@app.get("/", response_class=HTMLResponse)
def uploader():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload voice</title>
</head>
<body>
    <form action="/recognize" method="post" enctype="multipart/form-data">
        <input type="file" name="voice" id="voice" accept="voice/wav">
        <input type="submit" value="upload">
    </form>
</body>
</html>"""


import pydantic


class InputItem(pydantic.BaseModel):
    base64_voice: str
    voice_type: str
    is_egg: bool = True


@app.post("/raw_recognize")
async def raw_recognize(req: Request):
    resp = await req.body()

    return ["A", "B"]


@app.post("/recognize")
def recognize(item: InputItem):
    import base64, os
    from moviepy.editor import AudioFileClip

    print(item.is_egg)
    print(item.base64_voice)

    if item.is_egg:
        with open("tmp/temp.egg", "wb") as f:
            f.write(base64.b64decode(item.base64_voice.encode("utf-8")))
            f.flush()

        audio = AudioFileClip(
            "tmp/temp.egg",
        )
        audio.write_audiofile(
            "tmp/temp.wav",
        )
    else:
        f = open("tmp/temp.wav", "wb")
        f.write(base64.b64decode(item.base64_voice.encode("utf-8")))
        f.flush()
        os.fsync(f.fileno())
        f.close()

    ret: list[str] = model.recognize("tmp/temp.wav", skip_start_end=True)

    return ret


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
