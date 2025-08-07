from rfdetr import RFDETRBase
import argparse

def get_args():
    parser = argparse.ArgumentParser("Training Script")

    parser.add_argument("--epochs", default=1, type=int, help="number of training epochs")
    parser.add_argument("--data_path", default="../../../data/debug_data_Head/for_detr", type=str, help="path to the dataset directory")
    parser.add_argument("--name", default="test", type=str, help="name of the output directory")
    parser.add_argument("--freeze_encoder", action='store_true', help="freeze the encoder during training")
    parser.add_argument("--pretrain_weights", default="", type=str, help="path to pre-trained weights")
    return parser.parse_args()

if __name__ == "__main__":
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
                freeze_encoder=args.freeze_encoder,
                pretrain_weights=args.pretrain_weights,
                output_dir=f"../runs/train/{args.name}")