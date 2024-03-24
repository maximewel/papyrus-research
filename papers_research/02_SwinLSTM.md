* Paper: SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM
* Link: https://arxiv.org/pdf/2308.09891.pdf

Rationale: Combine LSTM and transformers if spatio-temporal data is important (as in, online image is a timeseries) ?


LSTM + CNNs show good performance, but CNNs are limited in their 'vision' or 'scope' to local dependancies. However, ViT show very good results and a generalisation capacity to the whole image dependancies. SwinLST adds tranformers to LSTM cells (directly embeded inside the cell) on the basis of Swin transformers

--> Reading Swin transformers

Swin-LSTM focuses on capturing large scale spatiotemporal dependencies, on the whole image.
Used to do segmentation or next frame prediction in videos. Capture dependencies in the input rather than the output.