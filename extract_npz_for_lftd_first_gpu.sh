# SeResNet101-IBN-a
python test.py --config_file configs/stage2/se_resnet101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/se_resnet101a_384/v1/se_resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/se_resnet101a_384/v1' > ./logs/se_resnet101a_384_npz_extract_v1.log

# ResNext101-IBN-a
python test.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v1/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/v1' > ./logs/resnext101a_384_npz_extract_v1.log

# ResNet101-IBN-a
python test.py --config_file configs/stage2/101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/101a_384/v1/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384/v1' > ./logs/101a_384_npz_extract_v1.log

# ResNet101-IBN-a (recrop)
python test.py --config_file configs/stage2/101a_384_recrop.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/101a_384_recrop/v1/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384_recrop/v1' > ./logs/101a_384_recrop_npz_extract_v1.log

# ResNet101-IBN-a (spgan)
python test.py --config_file configs/stage2/101a_384_spgan.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/101a_384_spgan/v1/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384_spgan/v1' > ./logs/101a_384_spgan_npz_extract_v1.log

# DenseNet169-IBN-a
python test.py --config_file configs/stage2/densenet169a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/densenet169a_384/v1/densenet169_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/densenet169a_384/v1' > ./logs/densenet169a_384_npz_extract_v1.log

# ResNest101
python test.py --config_file configs/stage2/s101_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/s101_384/v1/resnest101_2.pth' OUTPUT_DIR './logs/stage2/s101_384/v1' > ./logs/s101_384_npz_extract_v1.log

# Swin transformer 384 pth 29
python test.py --config_file configs/stage2/swin_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/swin_transformer_384/v1/swin_transformer_29.pth' OUTPUT_DIR './logs/stage2/swin_transformer_384/v1/pth29' > ./logs/swin_backbone_384_v2_pth29_test1.log

# Swin transformer 224
python test.py --config_file configs/stage2/swin.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/swin_transformer/v1/swin_transformer_2.pth' OUTPUT_DIR './logs/stage2/swin_transformer/v1' > ./logs/swin_backbone_224_npz_extract_v1.log

# TransReID
python test.py --config_file configs/stage2/transreid_256.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/transreid_256/v1/transformer_2.pth' OUTPUT_DIR './logs/stage2/transreid_256/v1' > ./logs/transreid_npz_extract_v1.log