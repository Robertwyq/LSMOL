name="$1"

python optimization.py --dataset WaymoSceneFlowDataset \
--dataset_path /data/yuqi_wang/waymo_v1.2/waymo_lsmol/$name \
--exp_name $name \
--batch_size 1 \
 --iters 5000 \
 --use_all_points \
 --compute_metrics \
 --hidden_units 128 \
 --lr 0.001  \
 --backward_flow  \
 --early_patience 300