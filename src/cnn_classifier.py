

from torch import nn


class CNNMLP(nn.Module):

    def __init__(self, input_shape: int, flattened_dim):
        super().__init__()

        self.image_features = nn.Sequential(
            nn.Conv3d(in_channels=input_shape, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10) #CHANGE OUTPUT TO MATCH CLASSES IN TRAIN SET - use softmax and logrank metric
        )

    def forward(self, image):
        cnn_features = self.image_features(image)     
        flattened_image = cnn_features.view(cnn_features.size(0), -1)

        predicted_age = self.mlp(flattened_image)     
        predicted_class = self.classifier(flattened_image)

        return predicted_age, predicted_class
    
