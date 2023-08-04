import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k-bits", type=int, default=64, help="length of hash codes.")
    parser.add_argument("--top-k-label", type=int, default=8, help="the top-k operation in proposed localized token aggregation (LTA) module.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="the number of transformer layers in local concept transforming (LCT) module.")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--res-mlp-layers", type=int, default=2, help="the number of ResMLP blocks in global concept learning (GCL) module.")

    parser.add_argument("--valid-freq", type=int, default=1, help="To valid every $valid-freq$ epochs.")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-lr", type=float, default=0.000002, help="learning rate for CLIP in MITH.")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for other modules in MITH.")

    parser.add_argument("--is-train", action="store_true")
    
    parser.add_argument("--hyper-tokens-intra", type=float, default=1, help="weight of the intra-modal similarity preservation loss, which is set to 1.")
    # hyper parameters
    parser.add_argument("--hyper-distill", type=float, default=1, help="weight of the distillation loss, which is labeled as $\ mu$ in paper.")
    parser.add_argument("--hyper-info-nce", type=float, default=50, help="weight of the contrastive inter-modal alignment loss, which is labeled as $\ delta$ in paper.")
    parser.add_argument("--hyper-cls-inter", type=float, default=10, help="weight of the inter-modal similarity preservation loss, which is labeled as $\ beta$ in paper.")
    parser.add_argument("--hyper-quan", type=float, default=8, help="weight of the quantization loss, which is labeled as $\ gamma$ in paper.")

    parser.add_argument("--hyper-alpha", type=float, default=0.01, help="$\ alpha$ in paper.")
    parser.add_argument("--hyper-lambda", type=float, default=0.99, help="$\ lambda$ in paper.")

    # other
    parser.add_argument("--clip-path", type=str, default="./cache/ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--dataset", type=str, default="coco", help="choose from [coco, flickr25k, nuswide]")
    parser.add_argument("--query-num", type=int, default=5000)
    parser.add_argument("--train-num", type=int, default=10000)

    parser.add_argument("--pretrained", type=str, default="", help="pretrained model path.")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--result-name", type=str, default="result", help="result dir name.")

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-proportion", type=float, default=0.05, help="Proportion of training to perform learning rate warmup.")

    args = parser.parse_args()

    import datetime
    _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.is_train:
        _time += "_test"
    k_bits = args.k_bits
    parser.add_argument("--save-dir", type=str, default=f"./{args.result_name}/{args.dataset}_{k_bits}/{_time}")
    args = parser.parse_args()

    return args
