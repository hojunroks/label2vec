import torch
import torch.nn as nn

def ours_loss(temp):
    sim_ = nn.CosineSimilarity(2)
    sim_f = nn.CosineSimilarity(1)
    def criterion_(feature_batch, label_batch, vectors):
        batch_size, feature_size = feature_batch.shape
        tiled_f = feature_batch.repeat(1,10).view(batch_size, -1, feature_size) # 256 * 10 * 512
        tiled_l = vectors[label_batch].repeat(1,10).view(batch_size, -1, feature_size) # 256 * 10 * 512
        class_num, feature_size = vectors.shape
        tiled_v = vectors.repeat(batch_size, 1).view(batch_size, class_num, feature_size) # 256 * 10 * 512
        tiled_sim_f = sim_(tiled_f, tiled_v) # 256 * 10
        tiled_sim_l = sim_(tiled_l, tiled_v) # 256 * 10
        e_sims_f = torch.exp(tiled_sim_f/temp) # 256 * 10
        e_sims_l = torch.exp(tiled_sim_l/temp) # 256 * 10
        e_sum_f = torch.sum(e_sims_f, dim=1).unsqueeze(0).transpose(0,1) # 256 * 1
        e_sum_l = torch.sum(e_sims_l, dim=1).unsqueeze(0).transpose(0,1) # 256 * 1
        loss = 1-torch.mean(sim_f(torch.log(e_sims_f/e_sum_f), torch.log(e_sims_l/e_sum_l)))
        return loss
    return criterion_
         
def contrastive_loss(temp):
    sim_ = nn.CosineSimilarity(2)
    def criterion_(feature_batch, label_batch, vectors):
        batch_size, feature_size = feature_batch.shape
        tiled_f = feature_batch.repeat(1,10).view(batch_size, -1, feature_size) # 256 * 10 * 512
        class_num, feature_size = vectors.shape
        tiled_v = vectors.repeat(batch_size, 1).view(batch_size, class_num, feature_size)
        tiled_sim = sim_(tiled_f, tiled_v)
        e_sims = torch.exp(tiled_sim/temp) # 256 * 10
        e_lab = torch.gather(e_sims.transpose(0,1), 0, label_batch.unsqueeze(0))[0]
        e_sum = torch.sum(e_sims, dim=1)
        loss = torch.mean(-torch.log(e_lab/e_sum))
        tiled_sim_f = sim_(tiled_f, tiled_v) # 256 * 10
        chu = torch.argmax(tiled_sim_f, dim=1)
        acc = torch.sum(torch.eq(chu, label_batch))/batch_size
        return loss, acc
    return criterion_