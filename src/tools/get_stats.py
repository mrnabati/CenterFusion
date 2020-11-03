import _init_paths
import torch
from dataset.datasets.nuscenes import nuScenes
from opts import opts
from tqdm import tqdm


def main(opt):
    split =  'val'
    batch_size = opt.batch_size
    # attributes = [5,8,9]
    attributes = list(range(18))

    dataset = nuScenes(opt, split)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    mean = torch.zeros((1,len(attributes)))
    var = torch.zeros((1,len(attributes)))
    maximum = torch.zeros((1,len(attributes)))
    minimum = torch.zeros((1,len(attributes)))
    num_batches = 0

    max_rcs = 0
    for data in tqdm(loader):
        pc = data['pc_2d']  # [batch_size, attr, 1000]
        N = data['pc_N']    # [batch_size]
        b_mean = torch.zeros((1,len(attributes))) # [1, attr]
        b_var = torch.zeros((1,len(attributes)))  # [1, attr]
        batch_samples = pc.shape[0]
        num_batches += 1
        
        for b in range(batch_samples):
            b_N = N[b]      # [attr, 1000]
            if b_N > 1:
                b_mean += pc[b, attributes, :b_N].mean(1).unsqueeze(0) # [1, attr]
                b_var += pc[b, attributes, :b_N].var(1).unsqueeze(0) # [1, attr]

                b_max = torch.max(pc[b,5,:b_N], dim=0)[0]
                if max_rcs < b_max:
                    max_rcs = b_max

            else:
                batch_samples -=1
        mean += b_mean/batch_samples
        var += b_var/batch_samples

        # b_max = torch.max(pc[:,5,:b_N], dim=1)[0]
        # b_max = torch.max(b_max, dim=0)[0]
        

    mean /= num_batches
    var /= num_batches
    std = torch.sqrt(var)
    print('mean: ', mean)
    print('std: ', std)
    print('max_rcs: ', max_rcs)

    with open("pc_stats.txt","w") as f:
        f.write("mean:" + ','.join([str(m) for m in mean.tolist()]))
        f.write("\nstd:" + ','.join([str(s) for s in std.tolist()]))

    # pre-calculated results:
    # pc_mean = np.array([407.1079, 255.5098, 41.8673, 1.8379, 36.2591, 6.6596, 
    #                 -0.4997, -0.8596, 0.0240, 0.0165, 1.0, 3.0, 19.5380, 
    #                 19.6979, 0.0, 1.0233, 16.4908, 3.0]).reshape(18,1)
    # pc_std = np.array([166.4957, 16.4833, 25.2681, 1.3892, 25.9181, 6.7230, 
    #                 2.7588, 2.0672, 1.4910, 0.5632, 0.0, 0.0, 0.7581, 
    #                 1.02991, 0.0, 0.2116, 0.5794, 0.0]).reshape(18,1)


if __name__ == '__main__':
    args = ['--pointcloud --nuscenes_att --velocity --dataset nuscenes --batch_size 10']
    opt = opts().parse()
    opt.pointcloud = True
    opt.dataset = 'nuscenes'
    opt.batch_size = 50
    main(opt)