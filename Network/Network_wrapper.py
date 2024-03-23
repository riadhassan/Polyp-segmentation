from Network.proposed import Our_proposed
# from Network.Network_UNet import UNet
# from Network.Network_R2Unet import R2U_Net
# from Network.Network_attenU import AttU_Net
# from Network.Network_BASNet import BASNet
# from Network.Network_UnetPP import NestedUNet
# from Network.Network_DenseNet import FCDenseNet
# from Network.Network_Basic_Unet import Basic_UNet
# from Network.Network_Our_without_uncertainty import Our_without_uncertainty

def model_wrapper(conf):
    in_channel = 3
    out_channel = 2

    architecture = {
        "our": Our_proposed(in_channels=in_channel, out_channels=out_channel),
        # "our_without_uncertainty": Our_without_uncertainty(in_channels=in_channel, out_channels=out_channel),
        # "UNet": UNet(in_channels=in_channel, out_channels=out_channel),
        # "R2UNet": R2U_Net(img_ch=in_channel, output_ch=out_channel),
        # "Attention_Unet": AttU_Net(input_channels=in_channel, num_classes=out_channel),
        # "BASNet": BASNet(n_channels=in_channel, n_classes=out_channel),
        # "Unet++": NestedUNet(num_classes=out_channel, input_channels=in_channel),
        # "DenseNet": FCDenseNet(n_classes=out_channel),
        # "Basic_Unet": Basic_UNet(n_channels=in_channel, n_classes=out_channel)
    }

    model = architecture[conf.model_name]

    return model