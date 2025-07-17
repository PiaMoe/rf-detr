from rfdetr import RFDETRBase

data_path = "../../../data/BOArDING_Dataset/BOArDING_Det/for_detr"
output_path = "../test_run"

model = RFDETRBase()

model.train(dataset_dir=data_path,
            epochs=1,
            batch_size=4,
            grad_accum_steps=4,
            lr=1e-4,
            wandb=True,
            project="rfdetr",
            run="test",
            output_dir=output_path)