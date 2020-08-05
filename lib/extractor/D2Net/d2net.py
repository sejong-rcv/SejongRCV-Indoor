import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, finetune_feature_extraction=False, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        model = models.vgg16()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')

        self.model = nn.Sequential(
            *list(model.features.children())[: conv4_3_idx + 1]
        )

        self.num_channels = 512

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            for param in list(self.model.parameters())[-2 :]:
                param.requires_grad = True
        import pdb;pdb.set_trace()
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        ext_out = self.model(batch)
        return ext_out


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score


class D2Net(nn.Module):
    def __init__(self, model_file=None, use_cuda=True, finetune=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=finetune,
            use_cuda=use_cuda
        )

        self.detection = SoftDetectionModule()

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)['model'])
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

            
    def forward(self, batch):

        b = batch['image1'].size(0)

        dense_features = self.dense_feature_extraction(
            torch.cat([batch['image1'], batch['image2']], dim=0)
        )

        scores = self.detection(dense_features)

        dense_features1 = dense_features[: b, :, :, :]
        dense_features2 = dense_features[b :, :, :, :]

        scores1 = scores[: b, :, :]
        scores2 = scores[b :, :, :]

        return {
            'dense_features1': dense_features1,
            'scores1': scores1,
            'dense_features2': dense_features2,
            'scores2': scores2
        }
class D2Net_local_extractor():
    def __init__(self,  model_file, use_relu=True, max_edge=1600, max_sum_edges=2800, preprocessing='caffe', multiscale=False):
        super(D2Net_local_extractor, self).__init__()

        self.max_edge = max_edge
        self.max_sum_edges = max_sum_edges
        self.preprocessing = preprocessing
        self.multiscale = multiscale
        self.model = d2test(
                            model_file=model_file,
                            use_relu=use_relu,
                            use_cuda=torch.cuda.is_available()
                        )

    def extract(self, image):
        
        resized_image = self.preproc_img(image)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=self.preprocessing
        )

        with torch.no_grad():
            if self.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    self.model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    self.model,
                    scales=[1]
                )
        
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        return {"keypoints" : keypoints,
                "scores" : scores,
                "descriptors" : descriptors}

    def preproc_img(self, image):

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=2)

        resized_image = image
        
        if max(resized_image.shape) > self.max_edge:
            ori_shape = np.asarray(resized_image.shape[:2])
            weight = self.max_edge / max(resized_image.shape)
            new_shape = (ori_shape * weight).astype("int").tolist()
            resize_fn = db.Resize(new_shape)
            resized_image = resize_fn(resized_image).astype('float')

        if sum(resized_image.shape[: 2]) > self.max_sum_edges:
            ori_shape = np.asarray(resized_image.shape[:2])
            weight = self.max_sum_edges / sum(resized_image.shape[: 2])
            new_shape = (ori_shape * weight).astype("int").tolist()
            resize_fn = db.Resize(new_shape)
            resized_image = resize_fn(resized_image).astype('float')

        return resized_image
