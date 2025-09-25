import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import time
from data.simple_dataset import RandomCropPair
from data.simple_dataset import SimpleDataSet
import torch.nn as nn
import transforms as T
import os
from scripts.calculate import load_lookup_table, Generator_for_info, apply_fusion_4d_with_interpolation


def main():
    lut_filepath = " "
    context_file = " "
    infrared_dir = " "
    visible_dir = " "
    save_dir = " "
    os.makedirs(save_dir, exist_ok=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lut = load_lookup_table(lut_filepath).to(device)
    if lut is None:
        return

    get_context = Generator_for_info()
    get_context.load_state_dict(torch.load(context_file))
    get_context = get_context.to(device)
    get_context.eval()

    data_transform = {
        "train": RandomCropPair(size=(96, 96)),
        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    val_dataset = SimpleDataSet(visible_path=visible_dir,
                                infrared_path=infrared_dir,
                                phase="val",
                                transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)

    infrared_files = sorted(os.listdir(infrared_dir))
    visible_files = sorted(os.listdir(visible_dir))

    assert len(infrared_files) == len(visible_files), "红外和可见光文件夹中的图像数量不一致！"
    target_size = (128, 128)
    times = []

    for step, data in enumerate(val_loader):
        I_A, I_B, task = data

        if torch.cuda.is_available():
            I_A = I_A.to("cuda")
            I_B = I_B.to("cuda")

        torch.cuda.synchronize()
        start_time = time.time()
        outputs = apply_fusion_4d_with_interpolation(I_A * 255., I_B * 255., lut, get_context)
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        if not os.path.splitext(task[0])[1]:
            task_with_extension = task[0] + ".png"
        else:
            task_with_extension = task[0]
        save_path = os.path.join(save_dir, task_with_extension)
        fusion_result = outputs.squeeze(0).clamp(0, 1).cpu()
        fusion_result_image = ToPILImage()(fusion_result)
        fusion_result_image.save(save_path)


    warmup_skip = 25
    if len(times) > warmup_skip:
        times_after_warmup = times[warmup_skip:]
        avg_time = np.mean(times_after_warmup)
        std_time = np.std(times_after_warmup)
        print(f"处理完成！跳过前 {warmup_skip} 张图像后，平均时间: {avg_time:.4f} 秒，时间标准差: {std_time:.4f} 秒")
    else:
        print(f"图像数量不足以跳过前 {warmup_skip} 张！总图像数: {len(times)}")

if __name__ == "__main__":
    main()

