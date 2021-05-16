from sconf import Config
import torch

from model.mxfont.models import Generator
from model.mxfont.train import setup_transforms


from model.mxfont.datasets.ttf_utils import render, read_font
from model.mxfont.utils.visualize import refine, normalize
from PIL import Image

import cv2

class FontGenerator:

    def __init__(self):
        cfg = Config('model/mxfont/cfgs/defaults.yaml')
        g_kwargs = cfg.get('g_args', {})

        self.gen = Generator(1, cfg.C, 1, **g_kwargs)

        weight = torch.load('model/mxfont/fontgen.pth', map_location='cpu')

        if "generator_ema" in weight:
            weight = weight["generator_ema"]
        self.gen.load_state_dict(weight)

        _, self.val_transform = setup_transforms(cfg)

    def get_font_tensor(self, font_path, char):
        src_font = read_font(font_path)
        src_img = render(src_font, char)
        src_tensor = self.val_transform(src_img)
        return src_tensor

    def get_font_tensor_from_img(self, font_image):
        src_tensor = self.val_transform(font_image)
        return src_tensor

    def get_gen_font(self, src, trg):
        out = self.gen.gen_from_style_char(trg.unsqueeze(1).cuda(), src.unsqueeze(1).cuda())
        out = normalize(refine(out))
        out = out.mul(255).clamp(0, 255).byte()
        out = out.squeeze(0).squeeze(0).detach().cpu()
        font_image = Image.fromarray(out[2].squeeze(0).numpy(), mode='L')

        return font_image

    def get_libary(self, trg):
        all_chars = []
        for char in 'sกขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืฺุูเแโใไๅๆ็่้๊๋์ํ๎๐๑๒๓๔๕๖๗๘๙':
            src = self.get_font_tensor('model/mxfont/THSarabunNew.ttf', char).unsqueeze(0)
            src = torch.stack((src, src, src)).squeeze(1)
            font_char_image = self.get_gen_font(src, trg)
            all_chars.append(font_char_image[35:128, 25:90])
        return all_chars

    def to_tile(self, list_of_chars):
        res = []
        for i in range((len(list_of_chars) // 12)):
            res.append(list_of_chars[i * 12:(i * 12) + 12])
        return res

    def concat_tile(self, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    def sample_chars_to_tensor(self, char_tile_image):
        ### need to reconsider how to separate sample
        # sor = cv2.resize(char_tile_image[50:900, 0:850], (128, 128))
        # hor = cv2.resize(char_tile_image[:, 645:1455], (128, 128))
        # jor = cv2.resize(char_tile_image[200:, 1280:2100], (128, 128))
        #
        # sor = Image.fromarray(sor, mode='L')
        # hor = Image.fromarray(hor, mode='L')
        # jor = Image.fromarray(jor, mode='L')
        #
        # trg1 = self.get_font_tensor_from_img(sor).unsqueeze(0)
        # trg2 = self.get_font_tensor_from_img(hor).unsqueeze(0)
        # trg3 = self.get_font_tensor_from_img(jor).unsqueeze(0)
        #
        # trg = torch.stack((trg1, trg2, trg3)).squeeze(1)
        trg_path = 'model/Garuda-Book.ttf'
        trg1 = self.get_font_tensor(trg_path, 'ส').unsqueeze(0)
        trg2 = self.get_font_tensor(trg_path, 'ฮ').unsqueeze(0)
        trg3 = self.get_font_tensor(trg_path, 'จ').unsqueeze(0)
        trg = torch.stack((trg1, trg2, trg3)).squeeze(1)

        return trg

    def generate_font_from_sample_image(self, sample_char_image=None):

        sample_chars_tensor = self.sample_chars_to_tensor(sample_char_image)

        all_gen_chars = self.get_libary(sample_chars_tensor)

        all_char_tiles = self.to_tile(all_gen_chars)

        all_char_panel = self.concat_tile(all_char_tiles)

        return all_char_panel
        # cv2.imwrite('/content/fontTest_fix.jpg', all_char_panel)




