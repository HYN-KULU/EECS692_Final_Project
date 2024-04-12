import torch
from unimatch.unimatch import UniMatch
from argparse import ArgumentParser

def get_flow_model():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    parser.add_argument('--feature_channels', type=int, default=128)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--upsample_factor', type=int, default=4)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--ffn_dim_expansion', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--reg_refine', type=bool, default=True)
    parser.add_argument('--task', type=str, default='flow')
    args = parser.parse_args(args=[])
    DEVICE = 'cuda:0'
    print(DEVICE)
    model = UniMatch(feature_channels=args.feature_channels,
                        num_scales=args.num_scales,
                        upsample_factor=args.upsample_factor,
                        num_head=args.num_head,
                        ffn_dim_expansion=args.ffn_dim_expansion,
                        num_transformer_layers=args.num_transformer_layers,
                        reg_refine=args.reg_refine,
                        task=args.task).to(DEVICE)

    checkpoint = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

    model.to(DEVICE)
    model.eval()
    model._requires_grad = False
    return model

if __name__=="__main__":
    model=get_flow_model()