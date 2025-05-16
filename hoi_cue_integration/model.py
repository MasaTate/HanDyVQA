import torch
import torch.nn as nn
import torch.nn.functional as F


class HandPoseEncoder(nn.Module):
    def __init__(
        self,
        input_dim=63,
        output_dim=512,
        hidden_size=128,
        num_linear_layers=1,
        num_lstm_layers=1,
    ):
        super(HandPoseEncoder, self).__init__()

        layers = []
        input_size = input_dim
        for _ in range(num_linear_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = (
                hidden_size  # The output of the previous layer is the input of the next
            )

        # LSTM
        layers.append(
            nn.LSTM(hidden_size, output_dim, num_lstm_layers, batch_first=True)
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, hand_pose):
        # hand_pose: (B, T, 21, 3)
        B, T, J, C = hand_pose.shape
        x = hand_pose.view(B, T, -1)  # (B, T, 63)
        output, (h_n, c_n) = self.encoder(x)
        return h_n[-1]  # (B, output_dim)


class ObjectEncoder(nn.Module):
    def __init__(
        self,
        input_dim=4,
        output_dim=512,
        hidden_size=128,
        num_linear_layers=1,
        num_lstm_layers=1,
        object_num=5,
    ):
        super(ObjectEncoder, self).__init__()

        layers = []
        input_size = input_dim
        for _ in range(num_linear_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = (
                hidden_size  # The output of the previous layer is the input of the next
            )

        # LSTM
        layers.append(
            nn.LSTM(hidden_size, output_dim, num_lstm_layers, batch_first=True)
        )
        self.encoder = nn.Sequential(*layers)

        self.conv_pooling = nn.Conv1d(
            in_channels=output_dim, out_channels=output_dim, kernel_size=object_num
        )

    def forward(self, bbox):
        # bbox: (B, T, n, 4)
        feat_list = []
        for i in range(bbox.shape[2]):
            output, (h_n, c_n) = self.encoder(bbox[:, :, i, :])
            feat_list.append(h_n[-1])
        feat = torch.stack(feat_list, dim=2)

        feat = self.conv_pooling(feat).squeeze(
            2
        )  # (B, output_dim, 1) → (B, output_dim)

        return feat


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        hand_dim=63,
        bbox_dim=4,
        obj_dim=768,
        vis_dim=512,
        embed_dim=512,
        num_frame=16,
        object_num=5,
        projection_layers: int = 1,
        projection_dropout: float = 0.5,
    ):
        super(MultiModalEncoder, self).__init__()

        self.object_num = object_num

        hidden_accumulate_dim = 0
        if hand_dim is not None:
            self.hand_encoder = HandPoseEncoder(
                input_dim=hand_dim,
                output_dim=128,
                hidden_size=128,
                num_linear_layers=1,
                num_lstm_layers=1,
            )
            hidden_accumulate_dim += 128 * 2
        else:
            self.hand_encoder = None

        if bbox_dim is not None:
            self.bbox_encoder = ObjectEncoder(
                input_dim=bbox_dim,
                output_dim=128,
                hidden_size=128,
                num_linear_layers=1,
                num_lstm_layers=1,
                object_num=self.object_num,
            )
            hidden_accumulate_dim += 128 * 2
        else:
            self.bbox_encoder = None

        if obj_dim is not None:
            self.object_encoder = ObjectEncoder(
                input_dim=obj_dim,
                output_dim=128,
                hidden_size=128,
                num_linear_layers=1,
                num_lstm_layers=1,
                object_num=self.object_num,
            )
            hidden_accumulate_dim += 128 * 2
        else:
            self.object_encoder = None

        hidden_accumulate_dim += vis_dim

        self.vis_dropout = nn.Dropout(p=projection_dropout)

        assert projection_layers >= 1, "projection_layers は 1 以上を指定してください"
        layers = []
        in_dim = hidden_accumulate_dim
        for i in range(projection_layers):

            layers.append(nn.Linear(in_dim, embed_dim))

            if i < projection_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=projection_dropout))
            in_dim = embed_dim

        self.projection = nn.Sequential(*layers)

    def forward(
        self,
        left_hand_pose,
        right_hand_pose,
        left_bbox,
        right_bbox,
        left_object_feat,
        right_object_feat,
        visual_feat,
    ):
        """
        right_hand_pose: (B, T, 21, 3)
        left_hand_pose:  (B, T, 21, 3)
        bbox:     (B, T, n, 4)
        object:  (B, T, n, D)
        """

        final_feat = []
        if self.hand_encoder is not None:
            left_hand_feat = self.hand_encoder(left_hand_pose)
            right_hand_feat = self.hand_encoder(right_hand_pose)
            # print(left_hand_feat.shape)
            # print(right_hand_feat.shape)
            final_feat.append(left_hand_feat)
            final_feat.append(right_hand_feat)

        if self.bbox_encoder is not None:
            left_bbox_feat = self.bbox_encoder(left_bbox)
            right_bbox_feat = self.bbox_encoder(right_bbox)
            # print(left_bbox_feat.shape)
            # print(right_bbox_feat.shape)
            final_feat.append(left_bbox_feat)
            final_feat.append(right_bbox_feat)

        if left_object_feat is not None:
            left_object_feat = self.object_encoder(left_object_feat)
            right_object_feat = self.object_encoder(right_object_feat)
            # print(left_object_feat.shape)
            # print(right_object_feat.shape)
            final_feat.append(left_object_feat)
            final_feat.append(right_object_feat)

        if len(visual_feat.shape) == 3:
            visual_feat = visual_feat.squeeze(1)

        visual_feat = self.vis_dropout(visual_feat)
        final_feat.append(visual_feat)

        final_feat = torch.cat(final_feat, dim=-1)
        final_feat = self.projection(final_feat)

        return final_feat


NUM_FRAME = 16
VIS_D = 512
HAND_D = (21, 3)
OBJ_D = 768
MAX_OBJ_NUM = 8

model = MultiModalEncoder(
    hand_dim=HAND_D[0] * HAND_D[1],
    bbox_dim=4,
    obj_dim=OBJ_D,
    vis_dim=VIS_D,
    embed_dim=VIS_D,
    num_frame=NUM_FRAME,
    object_num=MAX_OBJ_NUM,
    projection_layers=2,
    projection_dropout=0.1,
)

BATCH_SIZE = 1

# Input
video_feats = torch.randn(BATCH_SIZE, VIS_D)
text_feats = torch.randn(BATCH_SIZE, VIS_D)
left_hand_feats = torch.randn(BATCH_SIZE, NUM_FRAME, *HAND_D)
right_hand_feats = torch.randn(BATCH_SIZE, NUM_FRAME, *HAND_D)
left_bbox_feats = torch.randn(BATCH_SIZE, NUM_FRAME, MAX_OBJ_NUM, 4)
right_bbox_feats = torch.randn(BATCH_SIZE, NUM_FRAME, MAX_OBJ_NUM, 4)
left_obj_feats = torch.randn(BATCH_SIZE, NUM_FRAME, MAX_OBJ_NUM, OBJ_D)
right_obj_feats = torch.randn(BATCH_SIZE, NUM_FRAME, MAX_OBJ_NUM, OBJ_D)
print()
print("========Input Features========")
print("video_feats:", video_feats.shape)
print("left_hand_feats:", left_hand_feats.shape)
print("right_hand_feats:", right_hand_feats.shape)
print("left_bbox_feats:", left_bbox_feats.shape)
print("right_bbox_feats:", right_bbox_feats.shape)
print("left_obj_feats:", left_obj_feats.shape)
print("right_obj_feats:", right_obj_feats.shape)
print()


# Integrated Feature
print("========Integrated Feature========")
mm_features = model(
    left_hand_feats,
    right_hand_feats,
    left_bbox_feats,
    right_bbox_feats,
    left_obj_feats,
    right_obj_feats,
    video_feats,
)

mm_features = mm_features / torch.norm(mm_features, dim=1, keepdim=True)
text_feats = text_feats / torch.norm(text_feats, dim=1, keepdim=True)

print("Integrated Feature:", mm_features.shape)
print()

print("========Simialrity with Text Feature========")
similarity_matrix = mm_features @ text_feats.T
print("Similarity matrix:", similarity_matrix)
print()
