python train.py --config_file=configs/stage1/lftd.yml
# python test.py --config_file configs/stage1/lftd.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage1/lftd/lftd_10.pth' OUTPUT_DIR './logs/stage1/lftd/'

python train_stage2_v1.py --config_file=configs/stage2/lftd.yml
python test.py --config_file configs/stage2/lftd.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/lftd/v1/lftd_2.pth' OUTPUT_DIR './logs/stage2/lftd/v1/'
python ensemble.py

