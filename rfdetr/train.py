from rfdetr import RFDETRBase
import argparse

def get_args():
    parser = argparse.ArgumentParser("Training Script")

    parser.add_argument("--epochs", default=15, type=int, help="number of training epochs")
    parser.add_argument("--data_path", default="../../../data/BOArDING_Dataset/BOArDING_Det/for_detr", type=str, help="path to the dataset directory")
    parser.add_argument("--name", default="default", type=str, help="name of the output directory")

    return parser.parse_args()

args = get_args()
print("Training with the following parameters:")
print(args)


model = RFDETRBase()

model.train(dataset_dir=args.data_path,
            epochs=args.epochs,
            batch_size=4,
            grad_accum_steps=4,
            lr=1e-4,
            wandb=True,
            project="rfdetr",
            run=args.name,
            output_dir=f"../runs/train/{args.name}")