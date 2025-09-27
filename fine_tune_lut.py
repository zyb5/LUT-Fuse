import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.o_fusion_dataset import DistillDataSet
from data.o_fusion_dataset import RandomCropPair
import datetime
import os

import transforms as T
from scripts.loss_lut import fusion_loss
from itertools import chain
from scripts.calculate import OptimizableLUT, Generator_for_info, apply_fusion_4d_with_interpolation

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class TV_4D(nn.Module):
    def __init__(self, dim=16, output_channels=3):
        super(TV_4D, self).__init__()

        self.weight_r = torch.ones(dim, dim, dim, dim - 1, output_channels, dtype=torch.float)
        self.weight_r[..., (0, dim - 2), :] *= 2.0

        self.weight_g = torch.ones(dim, dim, dim - 1, dim, output_channels, dtype=torch.float)
        self.weight_g[..., (0, dim - 2), :, :] *= 2.0

        self.weight_b = torch.ones(dim, dim - 1, dim, dim, output_channels, dtype=torch.float)
        self.weight_b[..., (0, dim - 2), :, :, :] *= 2.0

        self.weight_ir = torch.ones(dim - 1, dim, dim, dim, output_channels, dtype=torch.float)
        self.weight_ir[(0, dim - 2), :, :, :, :] *= 2.0

        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        device = LUT.device

        self.weight_r = self.weight_r.to(device)

        self.weight_g = self.weight_g.to(device)
        self.weight_b = self.weight_b.to(device)
        self.weight_ir = self.weight_ir.to(device)

        dif_r = LUT[ :, :, :, :-1, :] - LUT[ :, :, :, 1:, :]  
        dif_g = LUT[ :, :, :-1, :, :] - LUT[ :, :, 1:, :, :]  
        dif_b = LUT[ :, :-1, :, :, :] - LUT[ :, 1:, :, :, :]  
        dif_ir = LUT[ :-1, :, :, :, :] - LUT[ 1:, :, :, :, :]  

        tv = (torch.mean(torch.mul(dif_r ** 2, self.weight_r)) + torch.mean(torch.mul(dif_g ** 2, self.weight_g)) +
              torch.mean(torch.mul(dif_b ** 2, self.weight_b)) + torch.mean(torch.mul(dif_ir ** 2, self.weight_ir)))

        mn = (torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) +
              torch.mean(self.relu(dif_b)) + torch.mean(self.relu(dif_ir)))

        return tv, mn


def fine_tune_lut(lut_model, Generator_context, train_loader, val_loader, device, epochs, learning_rate, save_dir="ww"):
    TV4 = TV_4D().to(device)
    best_val_loss = 1e5
    Generator_context.train()
    loss_fuction = fusion_loss()
    optimizer = optim.Adam(chain(lut_model.parameters(), Generator_context.parameters()), lr=learning_rate)
    # optimizer = optim.Adam(lut_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        lut_model.train()

        train_loss = 0
        # train_loss_max = 0
        # train_loss_text = 0
        train_loss_l1 = 0
        train_loss_ssim = 0
        train_loss_tv0 = 0
        train_loss_mn0 = 0

        for step, data in enumerate(train_loader):
            I_A, I_B, fuse, _ = data
            # optimizer.zero_grad()

            if torch.cuda.is_available():
                I_A = I_A.to(device)
                I_B = I_B.to(device)
                high_quality = fuse.to(device)
                loss_fuction = loss_fuction.to(device)

            lut = lut_model()

            tv0, mn0 = TV4(lut)
            loss_tv0 = tv0
            loss_mn0 = mn0

            outputs = apply_fusion_4d_with_interpolation(I_A * 255., I_B * 255., lut, Generator_context)

            l1 = F.l1_loss(outputs, high_quality)
            ssim = loss_fuction(I_A, I_B, outputs)
            loss_all = l1 + ssim + 10.0 * loss_mn0 + 0.0001 * loss_tv0 #+ text_loss + loss_max

            loss_all.backward()
            optimizer.step()

            train_loss += loss_all.item()
            train_loss_l1 += l1.item()
            train_loss_ssim += ssim.item()
            # train_loss_text += text_loss.item()
            # train_loss_max += loss_max.item()
            train_loss_tv0 += loss_tv0.item()
            train_loss_mn0 += loss_mn0.item()
            # train_loss_color += loss_color.item()

        tb_writer.add_scalar("train_total_loss", train_loss/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_l1", train_loss_l1/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_ssim", train_loss_ssim / len(train_loader), epoch)
        # tb_writer.add_scalar("train_loss_text", train_loss_text / len(train_loader), epoch)
        # tb_writer.add_scalar("train_loss_max", train_loss_max / len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_tv0", train_loss_tv0/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_mn0", train_loss_mn0/len(train_loader), epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - loss_l1: {train_loss_l1 / len(train_loader):.6f} - loss_ssim: {train_loss_ssim / len(train_loader):.6f} - loss_tv: {train_loss_tv0 / len(train_loader):.6f} - loss_tv: {train_loss_tv0 / len(train_loader):.6f} ")
        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - l1: {train_loss_l1 / len(train_loader):.6f} - loss_text: {train_loss_text / len(train_loader):.6f} - loss_max: {train_loss_max / len(train_loader):.6f}")
        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - l1: {train_loss_l1 / len(train_loader):.6f} - tv: {train_loss_tv0 / len(train_loader):.6f} - mn: {train_loss_mn0 / len(train_loader):.6f}")

        if (epoch + 1) % 10 == 0:
            val_loss, val_loss_l1, val_loss_ssim, val_loss_tv0, val_loss_mn0 = validate_lut(lut_model, val_loader, device)
            tb_writer.add_scalar("val_total_loss", val_loss/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_l1", val_loss_l1/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_ssim", val_loss_ssim / len(val_loader), epoch)
            # tb_writer.add_scalar("val_loss_text", val_loss_text / len(val_loader), epoch)
            # tb_writer.add_scalar("val_loss_max", val_loss_max / len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_tv0", val_loss_tv0/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_mn0", val_loss_mn0/len(val_loader), epoch)
            # print(f"Validation - Epoch {epoch} - Loss: {val_loss / len(val_loader):.6f} - l1: {val_loss_l1 / len(val_loader):.6f} - tv: {val_loss_tv0 / len(val_loader):.6f} - mn: {val_loss_mn0 / len(val_loader):.6f}")

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                filename = f"fine_tuned_ygcy_epoch{epoch}_valloss{val_loss:.6f}.npy"
                full_path = os.path.join(filefold_path, filename)
                save_lut(lut_model, full_path)

                context_filename = f"generator_context_epoch{epoch}_valloss{val_loss:.6f}.pth"
                generator_context_save_path = os.path.join(filefold_path, context_filename)
                save_generator_context(Generator_context, save_path=generator_context_save_path)

            print(f"Validation - Epoch {epoch} - Loss: {val_loss / len(val_loader):.6f} - l1: {val_loss_l1 / len(val_loader):.6f} - loss_ssim: {val_loss_ssim / len(val_loader):.6f} - loss_tv0: {val_loss_tv0 / len(val_loader):.6f} - loss_mn0: {val_loss_mn0 / len(val_loader):.6f}")


def save_lut(lut_module, path):

    lut_weights = lut_module().detach().cpu().numpy()
    np.save(path, lut_weights)
    print(f"Fine-tuned LUT saved to {path}")


def validate_lut(lut_module, val_loader, device):
    train_loss = 0
    train_loss_mn0 = 0
    train_loss_tv0 = 0
    train_loss_ssim = 0
    train_loss_l1 = 0
    # train_loss_text = 0
    TV4 = TV_4D().to(device)

    loss_fuction = fusion_loss()
    Generator_context.eval()

    lut = lut_module()
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            I_A, I_B, fuse, task = data
            if torch.cuda.is_available():
                I_A = I_A.to(device)
                I_B = I_B.to(device)
                high_quality = fuse.to(device)
                loss_fuction = loss_fuction.to(device)

            outputs = apply_fusion_4d_with_interpolation(I_A * 255., I_B * 255., lut, Generator_context)
            tv0, mn0 = TV4(lut)
            loss_tv0 = tv0
            loss_mn0 = mn0
            l1 = F.l1_loss(outputs, high_quality)
            loss_ssim = loss_fuction(I_A, I_B, outputs)
            loss_all = l1 + loss_ssim + 0.1 * loss_mn0 + 10.0 * loss_tv0 #+ text_loss + max_loss

            train_loss += loss_all.item()
            train_loss_l1 += l1.item()
            train_loss_ssim += loss_ssim.item()
            # train_loss_text += text_loss.item()
            # train_loss_max += max_loss.item()
            train_loss_tv0 += loss_tv0.item()
            train_loss_mn0 += loss_mn0.item()

    return train_loss, train_loss_l1 , train_loss_tv0, train_loss_mn0


def save_generator_context(generator_context, save_path="generator_context.pth"):
    torch.save(generator_context.state_dict(), save_path)
    print(f"Generator_for_info weights saved to {save_path}")

if __name__ == "__main__":

    if os.path.exists("./finetune_lut_exp") is False:                       
        os.makedirs("./finetune_lut_exp")
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./finetune_lut_exp/finetune_lut_{}".format(file_name)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lut_filepath = "ckpts/fine_tuned_lut_original.npy"   
    lut_tensor = torch.tensor(np.load(lut_filepath).astype(np.float32), device=DEVICE) 
    lut = OptimizableLUT(lut_tensor) 

    context_file = "ckpts/generator_context_original.pth"
    Generator_context = Generator_for_info().to(DEVICE)        
    Generator_context.load_state_dict(torch.load(context_file))
    # Generator_context.eval()

    batch_size = 6
    visible_path = " "
    infrared_path = " "
    train_fusion_path = " "
    test_visible_path = " "
    test_infrared_path = " "
    test_fusion_path = " "
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    data_transform = {
        "train": RandomCropPair(size=(128, 128)),
        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = DistillDataSet(visible_path=visible_path,
                                  infrared_path=infrared_path,
                                  other_fuse_path=train_fusion_path,
                                  phase="train",
                                  transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_dataset = DistillDataSet(visible_path=test_visible_path,
                                infrared_path=test_infrared_path,
                                other_fuse_path=test_fusion_path,
                                phase="val",
                                transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    fine_tune_lut(lut, Generator_context, train_loader, val_loader, DEVICE, epochs=496, learning_rate=5e-5)

